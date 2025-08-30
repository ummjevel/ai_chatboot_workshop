#!/usr/bin/env python3
"""
1차시: 실무 환경 구축 & 빠른 프로토타입
Author: AI Chatbot Workshop
Date: 2024-08-30
Description: OpenAI API 연동 기본 챗봇과 스트리밍 응답 구현, Streamlit 인터페이스
"""

import os
import time
import json
import logging
import asyncio
from typing import Iterator, Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from functools import wraps
import streamlit as st
from openai import OpenAI
import redis

# 로컬 설정 임포트
from config import get_config

# 설정 로드
config = get_config()

# 로깅 설정
logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    """채팅 메시지 구조"""
    role: str  # "user" or "assistant" or "system"
    content: str
    timestamp: datetime
    tokens_used: int = 0
    processing_time: float = 0.0
    model: str = ""

@dataclass
class APIUsageStats:
    """API 사용량 통계"""
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_response_time: float = 0.0
    error_count: int = 0
    last_reset: datetime = None

class TokenMonitor:
    """토큰 사용량 모니터링 클래스"""
    
    # 모델별 토큰당 가격 (2024년 8월 기준)
    TOKEN_PRICES = {
        "gpt-4o-mini": {"input": 0.00015 / 1000, "output": 0.0006 / 1000},
        "gpt-4o": {"input": 0.005 / 1000, "output": 0.015 / 1000},
        "gpt-3.5-turbo": {"input": 0.001 / 1000, "output": 0.002 / 1000},
    }
    
    def __init__(self):
        self.redis_client = None
        self._init_redis()
    
    def _init_redis(self):
        """Redis 연결 초기화"""
        try:
            self.redis_client = redis.from_url(config.get_redis_url())
            self.redis_client.ping()
            logger.info("Redis 연결 성공")
        except Exception as e:
            logger.warning(f"Redis 연결 실패, 메모리 모드로 전환: {e}")
            self.redis_client = None
    
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """토큰 사용량 기반 비용 계산"""
        if model not in self.TOKEN_PRICES:
            logger.warning(f"알 수 없는 모델: {model}, 기본 요금 적용")
            model = "gpt-4o-mini"
        
        prices = self.TOKEN_PRICES[model]
        input_cost = input_tokens * prices["input"]
        output_cost = output_tokens * prices["output"]
        
        return round(input_cost + output_cost, 6)
    
    def track_usage(self, user_id: str, model: str, input_tokens: int, 
                   output_tokens: int, processing_time: float):
        """사용량 추적 및 저장"""
        total_tokens = input_tokens + output_tokens
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        
        # 사용량 데이터 생성
        usage_data = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost": cost,
            "processing_time": processing_time
        }
        
        try:
            if self.redis_client:
                # Redis에 저장
                key = f"usage:{user_id}:{datetime.now().strftime('%Y%m%d')}"
                self.redis_client.lpush(key, json.dumps(usage_data))
                self.redis_client.expire(key, 86400 * 30)  # 30일 보관
            
            # 로그에 기록
            logger.info(
                f"토큰 사용량 - 사용자: {user_id}, 모델: {model}, "
                f"토큰: {total_tokens}, 비용: ${cost:.4f}, 시간: {processing_time:.2f}s"
            )
            
        except Exception as e:
            logger.error(f"사용량 추적 중 오류: {e}")
    
    def get_daily_usage(self, user_id: str) -> Dict[str, Any]:
        """일일 사용량 조회"""
        today = datetime.now().strftime('%Y%m%d')
        key = f"usage:{user_id}:{today}"
        
        try:
            if self.redis_client:
                usage_list = self.redis_client.lrange(key, 0, -1)
                total_tokens = 0
                total_cost = 0.0
                request_count = len(usage_list)
                
                for usage_json in usage_list:
                    usage = json.loads(usage_json.decode())
                    total_tokens += usage.get("total_tokens", 0)
                    total_cost += usage.get("cost", 0.0)
                
                return {
                    "date": today,
                    "request_count": request_count,
                    "total_tokens": total_tokens,
                    "total_cost": round(total_cost, 4),
                    "limit_remaining": max(0, config.performance.max_tokens_per_user_day - total_tokens)
                }
        except Exception as e:
            logger.error(f"사용량 조회 중 오류: {e}")
        
        # 기본값 반환
        return {
            "date": today,
            "request_count": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "limit_remaining": config.performance.max_tokens_per_user_day
        }

class APIKeyRotator:
    """API 키 로테이션 관리 클래스"""
    
    def __init__(self):
        self.primary_key = config.llm.openai_api_key
        self.backup_key = config.llm.backup_api_key
        self.current_key = self.primary_key
        self.failure_count = 0
        self.last_rotation = datetime.now()
    
    def get_current_key(self) -> str:
        """현재 사용 중인 API 키 반환"""
        return self.current_key
    
    def handle_api_error(self, error: Exception) -> bool:
        """
        API 에러 처리 및 키 로테이션
        
        Returns:
            bool: 로테이션 성공 여부
        """
        self.failure_count += 1
        logger.warning(f"API 호출 실패 (횟수: {self.failure_count}): {error}")
        
        # 3번 실패 시 백업 키로 전환
        if self.failure_count >= 3 and self.backup_key and self.current_key != self.backup_key:
            logger.info("백업 API 키로 전환")
            self.current_key = self.backup_key
            self.failure_count = 0
            self.last_rotation = datetime.now()
            return True
        
        # 백업 키도 실패 시 primary로 다시 전환 (쿨다운 후)
        if self.current_key == self.backup_key and self.failure_count >= 3:
            cooldown = timedelta(minutes=5)
            if datetime.now() - self.last_rotation > cooldown:
                logger.info("Primary API 키로 복원 시도")
                self.current_key = self.primary_key
                self.failure_count = 0
                self.last_rotation = datetime.now()
                return True
        
        return False
    
    def reset_failure_count(self):
        """성공 시 실패 카운트 리셋"""
        self.failure_count = 0

def log_function_call(func):
    """함수 호출 로깅 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.debug(f"[ENTER] {func.__name__}", extra={
            "function": func.__name__, 
            "args_count": len(args),
            "kwargs_keys": list(kwargs.keys())
        })
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.debug(
                f"[EXIT] {func.__name__} - SUCCESS (소요시간: {elapsed:.2f}s)",
                extra={"function": func.__name__, "elapsed_time": elapsed}
            )
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"[EXIT] {func.__name__} - ERROR: {str(e)} (소요시간: {elapsed:.2f}s)", 
                extra={
                    "error": str(e), 
                    "function": func.__name__,
                    "elapsed_time": elapsed
                }
            )
            raise
    return wrapper

class BasicChatbot:
    """기본 챗봇 클래스"""
    
    def __init__(self):
        self.key_rotator = APIKeyRotator()
        self.token_monitor = TokenMonitor()
        self.client = None
        self._init_openai_client()
    
    def _init_openai_client(self):
        """OpenAI 클라이언트 초기화"""
        try:
            client_config = config.get_openai_client_config()
            client_config["api_key"] = self.key_rotator.get_current_key()
            self.client = OpenAI(**client_config)
            logger.info("OpenAI 클라이언트 초기화 완료")
        except Exception as e:
            logger.error(f"OpenAI 클라이언트 초기화 실패: {e}")
            raise
    
    def _refresh_client_if_needed(self):
        """필요시 클라이언트 재생성 (키 로테이션 후)"""
        current_key = self.key_rotator.get_current_key()
        if self.client.api_key != current_key:
            self.client.api_key = current_key
            logger.info("API 키 갱신됨")
    
    @log_function_call
    def generate_response(self, messages: List[Dict[str, str]], 
                         user_id: str = "anonymous") -> ChatMessage:
        """
        일반 응답 생성 (비스트리밍)
        
        Args:
            messages: 대화 메시지 리스트
            user_id: 사용자 ID (사용량 추적용)
            
        Returns:
            ChatMessage: 생성된 응답 메시지
            
        Raises:
            Exception: API 호출 실패 시
        """
        start_time = time.time()
        
        # 사용량 제한 체크
        daily_usage = self.token_monitor.get_daily_usage(user_id)
        if daily_usage["limit_remaining"] <= 0:
            raise Exception(f"일일 토큰 한도 초과 ({daily_usage['total_tokens']} 토큰)")
        
        try:
            self._refresh_client_if_needed()
            
            # API 요청 로깅
            logger.info(
                f"OpenAI API 요청 시작 - 사용자: {user_id}, 모델: {config.llm.openai_model}",
                extra={
                    "user_id": user_id,
                    "model": config.llm.openai_model,
                    "message_count": len(messages)
                }
            )
            
            # OpenAI API 호출
            response = self.client.chat.completions.create(
                model=config.llm.openai_model,
                messages=messages,
                max_tokens=config.llm.max_tokens,
                temperature=config.llm.temperature,
                stream=False
            )
            
            # 응답 처리
            processing_time = time.time() - start_time
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            content = response.choices[0].message.content
            
            # 사용량 추적
            self.token_monitor.track_usage(
                user_id, config.llm.openai_model, 
                input_tokens, output_tokens, processing_time
            )
            
            # 성공 시 실패 카운트 리셋
            self.key_rotator.reset_failure_count()
            
            # ChatMessage 생성
            message = ChatMessage(
                role="assistant",
                content=content,
                timestamp=datetime.now(),
                tokens_used=response.usage.total_tokens,
                processing_time=processing_time,
                model=config.llm.openai_model
            )
            
            logger.info(
                f"OpenAI API 응답 완료 - 토큰: {response.usage.total_tokens}, "
                f"시간: {processing_time:.2f}s",
                extra={
                    "user_id": user_id,
                    "tokens_used": response.usage.total_tokens,
                    "processing_time": processing_time
                }
            )
            
            return message
            
        except Exception as e:
            # API 에러 처리 및 키 로테이션
            if self.key_rotator.handle_api_error(e):
                logger.info("키 로테이션 후 재시도")
                return self.generate_response(messages, user_id)
            
            logger.error(f"응답 생성 실패: {e}")
            raise
    
    @log_function_call
    def generate_streaming_response(self, messages: List[Dict[str, str]], 
                                  user_id: str = "anonymous") -> Iterator[Dict[str, Any]]:
        """
        스트리밍 응답 생성
        
        Args:
            messages: 대화 메시지 리스트
            user_id: 사용자 ID
            
        Yields:
            Dict: 스트리밍 청크 데이터
        """
        start_time = time.time()
        accumulated_content = ""
        total_tokens = 0
        
        # 사용량 제한 체크
        daily_usage = self.token_monitor.get_daily_usage(user_id)
        if daily_usage["limit_remaining"] <= 0:
            yield {
                "type": "error",
                "content": f"일일 토큰 한도 초과 ({daily_usage['total_tokens']} 토큰)",
                "finished": True
            }
            return
        
        try:
            self._refresh_client_if_needed()
            
            logger.info(f"스트리밍 응답 시작 - 사용자: {user_id}")
            
            # OpenAI 스트리밍 API 호출
            stream = self.client.chat.completions.create(
                model=config.llm.openai_model,
                messages=messages,
                max_tokens=config.llm.max_tokens,
                temperature=config.llm.temperature,
                stream=True
            )
            
            # 스트리밍 처리
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content_chunk = chunk.choices[0].delta.content
                    accumulated_content += content_chunk
                    
                    yield {
                        "type": "content",
                        "content": content_chunk,
                        "accumulated": accumulated_content,
                        "finished": False
                    }
            
            # 완료 처리
            processing_time = time.time() - start_time
            
            # TODO: 스트리밍에서는 정확한 토큰 수를 실시간으로 알 수 없음
            # 근사치 계산 (4 chars ≈ 1 token)
            estimated_tokens = len(accumulated_content) // 4
            
            # 사용량 추적 (추정값)
            self.token_monitor.track_usage(
                user_id, config.llm.openai_model,
                estimated_tokens // 2,  # 추정 input
                estimated_tokens // 2,  # 추정 output
                processing_time
            )
            
            self.key_rotator.reset_failure_count()
            
            yield {
                "type": "complete",
                "content": accumulated_content,
                "finished": True,
                "processing_time": processing_time,
                "estimated_tokens": estimated_tokens
            }
            
            logger.info(f"스트리밍 완료 - 시간: {processing_time:.2f}s, 추정 토큰: {estimated_tokens}")
            
        except Exception as e:
            if self.key_rotator.handle_api_error(e):
                logger.info("키 로테이션 후 스트리밍 재시도")
                yield from self.generate_streaming_response(messages, user_id)
                return
            
            logger.error(f"스트리밍 응답 실패: {e}")
            yield {
                "type": "error",
                "content": f"오류가 발생했습니다: {str(e)}",
                "finished": True
            }

def create_streamlit_ui():
    """Streamlit 웹 인터페이스 생성"""
    st.set_page_config(
        page_title="AI 챗봇 멘토링 - 1차시",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 AI 챗봇 멘토링 - 1차시: 기본 챗봇")
    st.caption("OpenAI API 연동과 스트리밍 응답 구현")
    
    # 사이드바 - 설정 및 통계
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 사용자 ID 입력
        user_id = st.text_input("사용자 ID", value="user123", key="user_id")
        
        # 모델 선택
        model_options = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
        selected_model = st.selectbox("모델 선택", model_options, index=0)
        
        # 스트리밍 모드
        streaming_mode = st.checkbox("스트리밍 모드", value=True)
        
        # Temperature 조정
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        
        st.divider()
        
        # 사용량 통계
        if st.button("📊 사용량 조회"):
            chatbot = BasicChatbot()
            daily_usage = chatbot.token_monitor.get_daily_usage(user_id)
            
            st.subheader("오늘의 사용량")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("요청 수", daily_usage["request_count"])
                st.metric("사용 토큰", daily_usage["total_tokens"])
            with col2:
                st.metric("예상 비용", f"${daily_usage['total_cost']:.4f}")
                st.metric("남은 한도", daily_usage["limit_remaining"])
    
    # 메인 영역
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 채팅")
        
        # 채팅 히스토리 초기화
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "system", 
                    "content": "당신은 도움이 되는 AI 어시스턴트입니다. 친근하고 정확한 답변을 제공해주세요."
                }
            ]
            st.session_state.display_messages = []
        
        # 이전 메시지 표시
        for msg in st.session_state.display_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if "metadata" in msg:
                    st.caption(f"🕐 {msg['metadata']['processing_time']:.2f}초 | "
                              f"🎯 {msg['metadata']['tokens_used']} 토큰")
        
        # 사용자 입력
        if prompt := st.chat_input("메시지를 입력하세요..."):
            # 사용자 메시지 추가
            user_msg = {"role": "user", "content": prompt}
            st.session_state.messages.append(user_msg)
            st.session_state.display_messages.append(user_msg)
            
            with st.chat_message("user"):
                st.write(prompt)
            
            # AI 응답 생성
            with st.chat_message("assistant"):
                try:
                    chatbot = BasicChatbot()
                    
                    if streaming_mode:
                        # 스트리밍 응답
                        message_placeholder = st.empty()
                        full_response = ""
                        
                        for chunk in chatbot.generate_streaming_response(
                            st.session_state.messages, user_id
                        ):
                            if chunk["type"] == "content":
                                full_response += chunk["content"]
                                message_placeholder.markdown(full_response + "▌")
                            elif chunk["type"] == "complete":
                                message_placeholder.markdown(full_response)
                                st.caption(f"🕐 {chunk['processing_time']:.2f}초 | "
                                          f"🎯 ~{chunk['estimated_tokens']} 토큰 (추정)")
                            elif chunk["type"] == "error":
                                st.error(chunk["content"])
                                break
                        
                        if full_response:
                            assistant_msg = {"role": "assistant", "content": full_response}
                            st.session_state.messages.append(assistant_msg)
                            st.session_state.display_messages.append({
                                **assistant_msg,
                                "metadata": {
                                    "processing_time": chunk.get("processing_time", 0),
                                    "tokens_used": chunk.get("estimated_tokens", 0)
                                }
                            })
                    else:
                        # 일반 응답
                        response = chatbot.generate_response(st.session_state.messages, user_id)
                        st.write(response.content)
                        st.caption(f"🕐 {response.processing_time:.2f}초 | "
                                  f"🎯 {response.tokens_used} 토큰")
                        
                        assistant_msg = {"role": "assistant", "content": response.content}
                        st.session_state.messages.append(assistant_msg)
                        st.session_state.display_messages.append({
                            **assistant_msg,
                            "metadata": {
                                "processing_time": response.processing_time,
                                "tokens_used": response.tokens_used
                            }
                        })
                
                except Exception as e:
                    st.error(f"오류가 발생했습니다: {str(e)}")
    
    with col2:
        st.header("📋 실습 가이드")
        
        with st.expander("🎯 1차시 목표", expanded=True):
            st.write("""
            - OpenAI API 연동 이해
            - 스트리밍 응답 구현
            - 토큰 사용량 모니터링
            - API 키 로테이션 로직
            - 기본 웹 인터페이스
            """)
        
        with st.expander("💡 실습 포인트"):
            st.write("""
            1. **스트리밍 vs 일반 응답** 비교
            2. **토큰 사용량** 실시간 확인
            3. **에러 처리** 및 재시도 로직
            4. **사용량 제한** 동작 확인
            5. **로깅** 메시지 확인 (터미널)
            """)
        
        with st.expander("🚀 확장 아이디어"):
            st.write("""
            - 대화 히스토리 저장
            - 다양한 LLM 모델 지원
            - 사용자별 설정 저장
            - 응답 품질 평가
            - 비용 분석 대시보드
            """)
        
        # 시스템 정보
        st.divider()
        st.subheader("🔧 시스템 정보")
        st.code(f"""
환경: {config.app.env}
모델: {config.llm.openai_model}
최대 토큰: {config.llm.max_tokens}
Temperature: {config.llm.temperature}
        """)

def run_cli_demo():
    """CLI 데모 실행"""
    print("=== AI 챗봇 멘토링 1차시 CLI 데모 ===")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.\n")
    
    chatbot = BasicChatbot()
    messages = [
        {
            "role": "system", 
            "content": "당신은 도움이 되는 AI 어시스턴트입니다. 친근하고 정확한 답변을 제공해주세요."
        }
    ]
    
    while True:
        try:
            user_input = input("\n사용자: ")
            if user_input.lower() in ['quit', 'exit']:
                print("챗봇을 종료합니다.")
                break
            
            if not user_input.strip():
                continue
            
            messages.append({"role": "user", "content": user_input})
            
            print("\nAI (스트리밍): ", end="", flush=True)
            
            # 스트리밍 응답 처리
            full_response = ""
            for chunk in chatbot.generate_streaming_response(messages, "cli_user"):
                if chunk["type"] == "content":
                    print(chunk["content"], end="", flush=True)
                    full_response += chunk["content"]
                elif chunk["type"] == "complete":
                    print(f"\n\n📊 처리시간: {chunk['processing_time']:.2f}초, "
                          f"추정 토큰: {chunk['estimated_tokens']}")
                elif chunk["type"] == "error":
                    print(f"\n❌ 오류: {chunk['content']}")
                    break
            
            if full_response:
                messages.append({"role": "assistant", "content": full_response})
        
        except KeyboardInterrupt:
            print("\n\n챗봇을 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        # CLI 모드
        run_cli_demo()
    else:
        # Streamlit 모드 (기본)
        print("Streamlit 웹 인터페이스를 시작합니다...")
        print("브라우저에서 http://localhost:8501 에 접속하세요.")
        print("CLI 모드로 실행하려면: python lesson1_basic_chatbot.py cli")
        create_streamlit_ui()