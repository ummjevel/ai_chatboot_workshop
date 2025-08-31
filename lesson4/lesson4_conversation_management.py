#!/usr/bin/env python3
"""
4차시: 대화 상태 관리 & 멀티턴 최적화
Author: AI Chatbot Workshop
Date: 2024-08-30
Description: Redis 기반 세션 관리, 컨텍스트 윈도우 최적화, 대화 흐름 제어
"""

import os
import sys
import json
import time
import uuid
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import asyncio

# 외부 라이브러리
import streamlit as st
import redis
from openai import OpenAI
import tiktoken
from collections import deque, defaultdict

# 로컬 모듈
sys.path.append('..')
from config import get_config

# 설정 및 로깅
config = get_config()
logger = logging.getLogger(__name__)

class ConversationState(Enum):
    """대화 상태 정의"""
    GREETING = "greeting"
    INFORMATION_SEEKING = "information_seeking"
    TASK_EXECUTION = "task_execution"
    PROBLEM_SOLVING = "problem_solving"
    CASUAL_CHAT = "casual_chat"
    ENDING = "ending"
    UNKNOWN = "unknown"

class MessageType(Enum):
    """메시지 유형"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"

@dataclass
class ConversationMessage:
    """대화 메시지 구조"""
    id: str
    role: MessageType
    content: str
    timestamp: datetime
    metadata: Dict[str, Any]
    tokens: int = 0
    importance_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'id': self.id,
            'role': self.role.value,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'tokens': self.tokens,
            'importance_score': self.importance_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationMessage':
        """딕셔너리에서 생성"""
        return cls(
            id=data['id'],
            role=MessageType(data['role']),
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {}),
            tokens=data.get('tokens', 0),
            importance_score=data.get('importance_score', 0.0)
        )

@dataclass
class ConversationSession:
    """대화 세션 구조"""
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    messages: List[ConversationMessage]
    current_state: ConversationState
    context_metadata: Dict[str, Any]
    total_tokens: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'messages': [msg.to_dict() for msg in self.messages],
            'current_state': self.current_state.value,
            'context_metadata': self.context_metadata,
            'total_tokens': self.total_tokens
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationSession':
        """딕셔너리에서 생성"""
        return cls(
            session_id=data['session_id'],
            user_id=data['user_id'],
            created_at=datetime.fromisoformat(data['created_at']),
            last_activity=datetime.fromisoformat(data['last_activity']),
            messages=[ConversationMessage.from_dict(msg) for msg in data['messages']],
            current_state=ConversationState(data['current_state']),
            context_metadata=data.get('context_metadata', {}),
            total_tokens=data.get('total_tokens', 0)
        )

class SessionManager:
    """Redis 기반 세션 관리자"""
    
    def __init__(self, redis_url: str = None, session_timeout: int = 3600):
        """
        세션 매니저 초기화
        
        Args:
            redis_url: Redis 연결 URL
            session_timeout: 세션 만료 시간(초)
        """
        self.redis_url = redis_url or config.get_redis_url()
        self.session_timeout = session_timeout
        self.client = None
        self.fallback_sessions = {}  # Redis 실패 시 메모리 백업
        
        self._init_redis()
        logger.info(f"세션 매니저 초기화 완료 - 만료시간: {session_timeout}초")
    
    def _init_redis(self):
        """Redis 연결 초기화"""
        try:
            self.client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            self.client.ping()
            logger.info("Redis 연결 성공")
        except Exception as e:
            logger.warning(f"Redis 연결 실패, 메모리 모드로 전환: {e}")
            self.client = None
    
    def create_session(self, user_id: str) -> ConversationSession:
        """
        새 대화 세션 생성
        
        Args:
            user_id: 사용자 ID
            
        Returns:
            ConversationSession: 생성된 세션
        """
        session_id = str(uuid.uuid4())
        now = datetime.now()
        
        session = ConversationSession(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_activity=now,
            messages=[],
            current_state=ConversationState.GREETING,
            context_metadata={
                'topic_history': [],
                'user_preferences': {},
                'conversation_summary': ""
            }
        )
        
        self._save_session(session)
        logger.info(f"새 세션 생성: {session_id} (사용자: {user_id})")
        
        return session
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """
        세션 조회
        
        Args:
            session_id: 세션 ID
            
        Returns:
            Optional[ConversationSession]: 조회된 세션
        """
        logger.debug(f"세션 조회: {session_id}")
        
        try:
            if self.client:
                # Redis에서 조회
                session_data = self.client.get(f"session:{session_id}")
                if session_data:
                    session = ConversationSession.from_dict(json.loads(session_data))
                    logger.debug(f"Redis에서 세션 로드: {session_id}")
                    return session
            else:
                # 메모리에서 조회
                if session_id in self.fallback_sessions:
                    logger.debug(f"메모리에서 세션 로드: {session_id}")
                    return self.fallback_sessions[session_id]
            
            logger.debug(f"세션을 찾을 수 없음: {session_id}")
            return None
            
        except Exception as e:
            logger.error(f"세션 조회 실패: {e}")
            return None
    
    def update_session(self, session: ConversationSession):
        """
        세션 업데이트
        
        Args:
            session: 업데이트할 세션
        """
        session.last_activity = datetime.now()
        self._save_session(session)
        logger.debug(f"세션 업데이트: {session.session_id}")
    
    def _save_session(self, session: ConversationSession):
        """세션 저장 (Redis 또는 메모리)"""
        try:
            session_data = json.dumps(session.to_dict(), ensure_ascii=False)
            
            if self.client:
                # Redis에 저장
                self.client.setex(
                    f"session:{session.session_id}",
                    self.session_timeout,
                    session_data
                )
                logger.debug(f"Redis에 세션 저장: {session.session_id}")
            else:
                # 메모리에 저장
                self.fallback_sessions[session.session_id] = session
                logger.debug(f"메모리에 세션 저장: {session.session_id}")
        
        except Exception as e:
            logger.error(f"세션 저장 실패: {e}")
    
    def add_message(self, session_id: str, message: ConversationMessage) -> bool:
        """
        세션에 메시지 추가
        
        Args:
            session_id: 세션 ID
            message: 추가할 메시지
            
        Returns:
            bool: 성공 여부
        """
        session = self.get_session(session_id)
        if not session:
            logger.warning(f"세션을 찾을 수 없음: {session_id}")
            return False
        
        # 토큰 수 계산
        if not message.tokens:
            message.tokens = self._count_tokens(message.content)
        
        session.messages.append(message)
        session.total_tokens += message.tokens
        
        # 중요도 점수 계산
        message.importance_score = self._calculate_importance(message, session)
        
        self.update_session(session)
        logger.debug(f"메시지 추가: {session_id}, 토큰: {message.tokens}")
        
        return True
    
    def _count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """토큰 수 계산"""
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception:
            # 대략적 계산 (4글자 ≈ 1토큰)
            return len(text) // 4
    
    def _calculate_importance(self, message: ConversationMessage, session: ConversationSession) -> float:
        """
        메시지 중요도 점수 계산
        
        Args:
            message: 평가할 메시지
            session: 현재 세션
            
        Returns:
            float: 중요도 점수 (0.0 ~ 1.0)
        """
        score = 0.5  # 기본 점수
        
        # 1. 메시지 길이 (긴 메시지일수록 중요)
        length_factor = min(len(message.content) / 200, 1.0) * 0.2
        score += length_factor
        
        # 2. 질문 포함 여부
        if '?' in message.content or '질문' in message.content:
            score += 0.2
        
        # 3. 시스템 메시지나 중요 키워드
        important_keywords = ['오류', '문제', '도움', '중요', '긴급']
        if any(keyword in message.content for keyword in important_keywords):
            score += 0.2
        
        # 4. 대화 초반 메시지 (컨텍스트 설정에 중요)
        if len(session.messages) < 5:
            score += 0.1
        
        return min(score, 1.0)
    
    def get_user_sessions(self, user_id: str, limit: int = 10) -> List[str]:
        """
        사용자의 세션 목록 조회
        
        Args:
            user_id: 사용자 ID
            limit: 반환할 세션 수 제한
            
        Returns:
            List[str]: 세션 ID 목록
        """
        # TODO: Redis에서 사용자별 세션 인덱스 관리
        # 현재는 기본 구현만 제공
        logger.info(f"사용자 {user_id}의 세션 목록 조회 요청")
        return []
    
    def cleanup_expired_sessions(self):
        """만료된 세션 정리"""
        if not self.client:
            # 메모리 모드에서 만료된 세션 정리
            now = datetime.now()
            expired_sessions = []
            
            for session_id, session in self.fallback_sessions.items():
                if now - session.last_activity > timedelta(seconds=self.session_timeout):
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self.fallback_sessions[session_id]
                logger.debug(f"만료된 세션 정리: {session_id}")

class ContextWindowOptimizer:
    """컨텍스트 윈도우 최적화"""
    
    def __init__(self, max_tokens: int = 4000, compression_ratio: float = 0.5):
        """
        컨텍스트 최적화기 초기화
        
        Args:
            max_tokens: 최대 토큰 수
            compression_ratio: 압축 시 유지할 비율
        """
        self.max_tokens = max_tokens
        self.compression_ratio = compression_ratio
        self.client = OpenAI(api_key=config.llm.openai_api_key)
        
        logger.info(f"컨텍스트 최적화기 초기화 - 최대 토큰: {max_tokens}")
    
    def optimize_context(self, session: ConversationSession) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        컨텍스트 윈도우 최적화
        
        Args:
            session: 최적화할 세션
            
        Returns:
            Tuple[List[Dict[str, str]], Dict[str, Any]]: 최적화된 메시지 목록과 메타데이터
        """
        logger.debug(f"컨텍스트 최적화 시작 - 현재 토큰: {session.total_tokens}")
        
        if session.total_tokens <= self.max_tokens:
            # 최적화 불필요
            messages = self._convert_to_openai_format(session.messages)
            metadata = {'optimization': 'none', 'original_tokens': session.total_tokens}
            return messages, metadata
        
        # 최적화 전략 결정
        if session.total_tokens > self.max_tokens * 2:
            # 대폭 압축 필요 - 요약 기반
            return self._summarize_context(session)
        else:
            # 중요도 기반 필터링
            return self._filter_by_importance(session)
    
    def _convert_to_openai_format(self, messages: List[ConversationMessage]) -> List[Dict[str, str]]:
        """OpenAI API 형식으로 변환"""
        return [
            {'role': msg.role.value, 'content': msg.content}
            for msg in messages
            if msg.role in [MessageType.USER, MessageType.ASSISTANT, MessageType.SYSTEM]
        ]
    
    def _filter_by_importance(self, session: ConversationSession) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        중요도 기반 메시지 필터링
        
        Args:
            session: 필터링할 세션
            
        Returns:
            Tuple: 필터링된 메시지와 메타데이터
        """
        logger.debug("중요도 기반 메시지 필터링 수행")
        
        # 메시지를 중요도 순으로 정렬
        sorted_messages = sorted(session.messages, key=lambda m: m.importance_score, reverse=True)
        
        # 최근 메시지는 항상 포함
        recent_messages = session.messages[-5:] if len(session.messages) > 5 else session.messages
        
        # 중요한 메시지 선별
        selected_messages = []
        current_tokens = 0
        target_tokens = int(self.max_tokens * self.compression_ratio)
        
        # 최근 메시지 먼저 추가
        for msg in reversed(recent_messages):
            if current_tokens + msg.tokens <= target_tokens:
                selected_messages.insert(0, msg)
                current_tokens += msg.tokens
        
        # 중요한 이전 메시지 추가
        for msg in sorted_messages:
            if msg not in selected_messages and current_tokens + msg.tokens <= target_tokens:
                # 시간순 위치 찾아서 삽입
                insert_pos = 0
                for i, selected_msg in enumerate(selected_messages):
                    if msg.timestamp < selected_msg.timestamp:
                        insert_pos = i
                        break
                    insert_pos = i + 1
                
                selected_messages.insert(insert_pos, msg)
                current_tokens += msg.tokens
        
        messages = self._convert_to_openai_format(selected_messages)
        metadata = {
            'optimization': 'importance_filtering',
            'original_count': len(session.messages),
            'filtered_count': len(selected_messages),
            'original_tokens': session.total_tokens,
            'filtered_tokens': current_tokens
        }
        
        logger.info(f"중요도 필터링 완료 - {len(session.messages)} -> {len(selected_messages)} 메시지")
        return messages, metadata
    
    def _summarize_context(self, session: ConversationSession) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        대화 내용 요약을 통한 컨텍스트 압축
        
        Args:
            session: 요약할 세션
            
        Returns:
            Tuple: 요약된 메시지와 메타데이터
        """
        logger.debug("대화 요약 기반 컨텍스트 압축 수행")
        
        try:
            # 요약할 메시지들 (최근 메시지 제외)
            messages_to_summarize = session.messages[:-3] if len(session.messages) > 3 else []
            recent_messages = session.messages[-3:] if len(session.messages) > 3 else session.messages
            
            if not messages_to_summarize:
                # 요약할 내용이 없음
                messages = self._convert_to_openai_format(session.messages)
                metadata = {'optimization': 'no_summary_needed'}
                return messages, metadata
            
            # 요약 프롬프트 구성
            conversation_text = "\n".join([
                f"{msg.role.value}: {msg.content}"
                for msg in messages_to_summarize
            ])
            
            summary_prompt = f"""다음 대화 내용을 간결하게 요약해주세요. 중요한 정보와 맥락을 유지하되, 
불필요한 세부사항은 제거해주세요.

대화 내용:
{conversation_text}

요약:"""
            
            # 요약 생성
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "당신은 대화 내용을 효율적으로 요약하는 전문가입니다."},
                    {"role": "user", "content": summary_prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content
            
            # 요약 메시지 생성
            summary_message = ConversationMessage(
                id=str(uuid.uuid4()),
                role=MessageType.SYSTEM,
                content=f"[이전 대화 요약]: {summary}",
                timestamp=datetime.now(),
                metadata={'type': 'conversation_summary'},
                tokens=len(summary) // 4
            )
            
            # 최종 메시지 구성 (요약 + 최근 메시지)
            final_messages = [summary_message] + recent_messages
            
            messages = self._convert_to_openai_format(final_messages)
            metadata = {
                'optimization': 'summarization',
                'original_count': len(session.messages),
                'summarized_count': len(messages_to_summarize),
                'final_count': len(final_messages),
                'original_tokens': session.total_tokens,
                'summary_tokens': summary_message.tokens + sum(msg.tokens for msg in recent_messages)
            }
            
            logger.info(f"대화 요약 완료 - {len(session.messages)} -> {len(final_messages)} 메시지")
            return messages, metadata
            
        except Exception as e:
            logger.error(f"대화 요약 실패: {e}")
            # 실패 시 중요도 필터링으로 폴백
            return self._filter_by_importance(session)

class IntentClassifier:
    """의도 분류기"""
    
    def __init__(self):
        """의도 분류기 초기화"""
        self.client = OpenAI(api_key=config.llm.openai_api_key)
        
        # 의도별 키워드 정의
        self.intent_keywords = {
            ConversationState.GREETING: ['안녕', '안녕하세요', '반갑', '처음', 'hello', 'hi'],
            ConversationState.INFORMATION_SEEKING: ['알려', '설명', '뭐야', '무엇', '어떻게', '왜', '언제', '어디서'],
            ConversationState.TASK_EXECUTION: ['해줘', '만들어', '생성', '작성', '실행', '처리'],
            ConversationState.PROBLEM_SOLVING: ['문제', '오류', '에러', '안돼', '도움', '해결'],
            ConversationState.ENDING: ['고마워', '감사', '잘가', '안녕히', 'bye', '끝'],
            ConversationState.CASUAL_CHAT: ['재미있', '웃긴', '농담', '이야기']
        }
        
        logger.info("의도 분류기 초기화 완료")
    
    def classify_intent(self, message: str, context_messages: List[ConversationMessage] = None) -> ConversationState:
        """
        메시지의 의도 분류
        
        Args:
            message: 분류할 메시지
            context_messages: 컨텍스트 메시지들
            
        Returns:
            ConversationState: 분류된 의도
        """
        logger.debug(f"의도 분류 시작: {message[:50]}...")
        
        # 1단계: 키워드 기반 빠른 분류
        quick_intent = self._classify_by_keywords(message)
        if quick_intent != ConversationState.UNKNOWN:
            logger.debug(f"키워드 기반 분류: {quick_intent.value}")
            return quick_intent
        
        # 2단계: LLM 기반 정확한 분류
        return self._classify_by_llm(message, context_messages)
    
    def _classify_by_keywords(self, message: str) -> ConversationState:
        """키워드 기반 빠른 의도 분류"""
        message_lower = message.lower()
        
        for intent, keywords in self.intent_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return intent
        
        return ConversationState.UNKNOWN
    
    def _classify_by_llm(self, message: str, context_messages: List[ConversationMessage] = None) -> ConversationState:
        """LLM 기반 정확한 의도 분류"""
        try:
            # 컨텍스트 구성
            context = ""
            if context_messages:
                recent_context = context_messages[-3:] if len(context_messages) > 3 else context_messages
                context = "\n".join([f"{msg.role.value}: {msg.content}" for msg in recent_context])
            
            prompt = f"""다음 메시지의 의도를 분류해주세요.

가능한 의도:
- greeting: 인사, 첫 만남
- information_seeking: 정보 요청, 질문
- task_execution: 작업 요청, 실행 명령
- problem_solving: 문제 해결, 도움 요청
- casual_chat: 일상 대화, 잡담
- ending: 대화 종료, 작별인사
- unknown: 분류 불가

{f'이전 대화 맥락: {context}' if context else ''}

분류할 메시지: {message}

의도 (위 카테고리 중 하나만 답변):"""
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "당신은 대화 의도를 정확하게 분류하는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip().lower()
            
            # 결과를 ConversationState로 변환
            intent_mapping = {
                'greeting': ConversationState.GREETING,
                'information_seeking': ConversationState.INFORMATION_SEEKING,
                'task_execution': ConversationState.TASK_EXECUTION,
                'problem_solving': ConversationState.PROBLEM_SOLVING,
                'casual_chat': ConversationState.CASUAL_CHAT,
                'ending': ConversationState.ENDING,
                'unknown': ConversationState.UNKNOWN
            }
            
            classified_intent = intent_mapping.get(result, ConversationState.UNKNOWN)
            logger.debug(f"LLM 기반 분류: {classified_intent.value}")
            
            return classified_intent
            
        except Exception as e:
            logger.error(f"LLM 의도 분류 실패: {e}")
            return ConversationState.UNKNOWN

class ConversationFlowController:
    """대화 흐름 제어기"""
    
    def __init__(self, session_manager: SessionManager, intent_classifier: IntentClassifier):
        """
        대화 흐름 제어기 초기화
        
        Args:
            session_manager: 세션 매니저
            intent_classifier: 의도 분류기
        """
        self.session_manager = session_manager
        self.intent_classifier = intent_classifier
        self.topic_change_threshold = 0.7  # 주제 변경 감지 임계값
        
        logger.info("대화 흐름 제어기 초기화 완료")
    
    def process_user_message(self, session_id: str, user_message: str) -> Dict[str, Any]:
        """
        사용자 메시지 처리 및 상태 전환
        
        Args:
            session_id: 세션 ID
            user_message: 사용자 메시지
            
        Returns:
            Dict[str, Any]: 처리 결과
        """
        logger.info(f"사용자 메시지 처리 시작: {session_id}")
        
        # 세션 조회
        session = self.session_manager.get_session(session_id)
        if not session:
            logger.error(f"세션을 찾을 수 없음: {session_id}")
            return {'error': '세션을 찾을 수 없습니다'}
        
        # 사용자 메시지 객체 생성
        user_msg = ConversationMessage(
            id=str(uuid.uuid4()),
            role=MessageType.USER,
            content=user_message,
            timestamp=datetime.now(),
            metadata={'processed_by': 'flow_controller'}
        )
        
        # 의도 분류
        current_intent = self.intent_classifier.classify_intent(user_message, session.messages)
        
        # 주제 변경 감지
        topic_changed = self._detect_topic_change(user_message, session)
        
        # 상태 전환 결정
        previous_state = session.current_state
        new_state = self._determine_state_transition(current_intent, session.current_state, topic_changed)
        
        # 세션 상태 업데이트
        session.current_state = new_state
        
        # 컨텍스트 메타데이터 업데이트
        if topic_changed:
            session.context_metadata['topic_history'].append({
                'timestamp': datetime.now().isoformat(),
                'previous_topic': session.context_metadata.get('current_topic', ''),
                'new_topic': self._extract_topic(user_message)
            })
            session.context_metadata['current_topic'] = self._extract_topic(user_message)
        
        # 메시지 추가
        self.session_manager.add_message(session_id, user_msg)
        
        # 처리 결과 반환
        result = {
            'session_id': session_id,
            'message_id': user_msg.id,
            'classified_intent': current_intent.value,
            'previous_state': previous_state.value,
            'current_state': new_state.value,
            'topic_changed': topic_changed,
            'processing_metadata': {
                'message_count': len(session.messages),
                'total_tokens': session.total_tokens,
                'session_duration': (datetime.now() - session.created_at).total_seconds()
            }
        }
        
        logger.info(f"메시지 처리 완료 - 의도: {current_intent.value}, 상태: {previous_state.value} -> {new_state.value}")
        
        return result
    
    def _determine_state_transition(self, intent: ConversationState, current_state: ConversationState, topic_changed: bool) -> ConversationState:
        """
        상태 전환 결정
        
        Args:
            intent: 분류된 의도
            current_state: 현재 상태
            topic_changed: 주제 변경 여부
            
        Returns:
            ConversationState: 새로운 상태
        """
        # 명확한 의도가 있는 경우 해당 상태로 전환
        if intent in [ConversationState.GREETING, ConversationState.ENDING]:
            return intent
        
        # 주제가 변경되었고 정보 요청이면 정보 탐색 상태로
        if topic_changed and intent == ConversationState.INFORMATION_SEEKING:
            return ConversationState.INFORMATION_SEEKING
        
        # 작업 실행 요청은 즉시 전환
        if intent == ConversationState.TASK_EXECUTION:
            return ConversationState.TASK_EXECUTION
        
        # 문제 해결 요청은 우선순위 높음
        if intent == ConversationState.PROBLEM_SOLVING:
            return ConversationState.PROBLEM_SOLVING
        
        # 그외 경우는 현재 상태 유지 또는 자연스러운 전환
        transition_rules = {
            ConversationState.GREETING: ConversationState.INFORMATION_SEEKING,
            ConversationState.INFORMATION_SEEKING: intent if intent != ConversationState.UNKNOWN else current_state,
            ConversationState.TASK_EXECUTION: intent if intent != ConversationState.UNKNOWN else ConversationState.INFORMATION_SEEKING,
            ConversationState.PROBLEM_SOLVING: intent if intent != ConversationState.UNKNOWN else current_state,
            ConversationState.CASUAL_CHAT: intent if intent != ConversationState.UNKNOWN else current_state
        }
        
        return transition_rules.get(current_state, intent if intent != ConversationState.UNKNOWN else current_state)
    
    def _detect_topic_change(self, message: str, session: ConversationSession) -> bool:
        """
        주제 변경 감지
        
        Args:
            message: 새 메시지
            session: 현재 세션
            
        Returns:
            bool: 주제 변경 여부
        """
        if len(session.messages) < 3:
            return False  # 초기 메시지들은 주제 변경으로 간주하지 않음
        
        # 최근 메시지들과의 유사성 계산 (간단한 키워드 기반)
        recent_messages = session.messages[-3:]
        recent_text = " ".join([msg.content for msg in recent_messages if msg.role == MessageType.USER])
        
        # 키워드 추출 및 비교 (실제 구현에서는 더 정교한 방법 사용)
        current_keywords = set(message.lower().split())
        recent_keywords = set(recent_text.lower().split())
        
        # 공통 키워드 비율 계산
        if not recent_keywords:
            return False
        
        common_ratio = len(current_keywords & recent_keywords) / len(current_keywords | recent_keywords)
        
        logger.debug(f"주제 유사도: {common_ratio:.3f}")
        
        return common_ratio < (1 - self.topic_change_threshold)
    
    def _extract_topic(self, message: str) -> str:
        """
        메시지에서 주제 추출
        
        Args:
            message: 분석할 메시지
            
        Returns:
            str: 추출된 주제
        """
        # 간단한 주제 추출 (실제 구현에서는 NER, 키워드 추출 등 사용)
        words = message.split()
        
        # 명사류 키워드 추출 (한글 기준 간단 휴리스틱)
        topic_candidates = []
        for word in words:
            if len(word) > 1 and not any(char in word for char in ['?', '!', '.']):
                topic_candidates.append(word)
        
        return " ".join(topic_candidates[:3]) if topic_candidates else "일반 대화"

class MultiTurnChatbot:
    """멀티턴 대화 최적화 챗봇"""
    
    def __init__(self):
        """멀티턴 챗봇 초기화"""
        self.session_manager = SessionManager()
        self.context_optimizer = ContextWindowOptimizer()
        self.intent_classifier = IntentClassifier()
        self.flow_controller = ConversationFlowController(
            self.session_manager, 
            self.intent_classifier
        )
        self.client = OpenAI(api_key=config.llm.openai_api_key)
        
        logger.info("멀티턴 챗봇 초기화 완료")
    
    def start_conversation(self, user_id: str) -> str:
        """
        새 대화 시작
        
        Args:
            user_id: 사용자 ID
            
        Returns:
            str: 세션 ID
        """
        session = self.session_manager.create_session(user_id)
        logger.info(f"새 대화 시작: {session.session_id} (사용자: {user_id})")
        return session.session_id
    
    def chat(self, session_id: str, user_message: str) -> Dict[str, Any]:
        """
        대화 처리
        
        Args:
            session_id: 세션 ID
            user_message: 사용자 메시지
            
        Returns:
            Dict[str, Any]: 응답 및 메타데이터
        """
        start_time = time.time()
        logger.info(f"대화 처리 시작: {session_id}")
        
        try:
            # 1단계: 사용자 메시지 처리 및 상태 전환
            flow_result = self.flow_controller.process_user_message(session_id, user_message)
            
            if 'error' in flow_result:
                return flow_result
            
            # 2단계: 세션 조회 및 컨텍스트 최적화
            session = self.session_manager.get_session(session_id)
            optimized_messages, optimization_metadata = self.context_optimizer.optimize_context(session)
            
            # 3단계: 상태별 시스템 프롬프트 구성
            system_prompt = self._generate_system_prompt(session.current_state, flow_result)
            
            # 4단계: LLM 응답 생성
            messages = [{"role": "system", "content": system_prompt}] + optimized_messages
            
            response = self.client.chat.completions.create(
                model=config.llm.openai_model,
                messages=messages,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens
            )
            
            ai_response = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            # 5단계: AI 응답 메시지 저장
            ai_message = ConversationMessage(
                id=str(uuid.uuid4()),
                role=MessageType.ASSISTANT,
                content=ai_response,
                timestamp=datetime.now(),
                metadata={
                    'model': config.llm.openai_model,
                    'temperature': config.llm.temperature,
                    'conversation_state': session.current_state.value
                },
                tokens=tokens_used
            )
            
            self.session_manager.add_message(session_id, ai_message)
            
            processing_time = time.time() - start_time
            
            # 응답 구성
            result = {
                'session_id': session_id,
                'response': ai_response,
                'conversation_state': session.current_state.value,
                'processing_time': processing_time,
                'tokens_used': tokens_used,
                'flow_analysis': flow_result,
                'context_optimization': optimization_metadata,
                'session_stats': {
                    'message_count': len(session.messages),
                    'total_tokens': session.total_tokens,
                    'duration_seconds': (datetime.now() - session.created_at).total_seconds()
                }
            }
            
            logger.info(f"대화 처리 완료 - 시간: {processing_time:.2f}초, 토큰: {tokens_used}")
            
            return result
            
        except Exception as e:
            logger.error(f"대화 처리 실패: {e}")
            return {'error': f'대화 처리 중 오류 발생: {str(e)}'}
    
    def _generate_system_prompt(self, state: ConversationState, flow_result: Dict[str, Any]) -> str:
        """
        상태별 시스템 프롬프트 생성
        
        Args:
            state: 현재 대화 상태
            flow_result: 대화 흐름 분석 결과
            
        Returns:
            str: 시스템 프롬프트
        """
        base_prompt = "당신은 도움이 되는 AI 어시스턴트입니다."
        
        state_prompts = {
            ConversationState.GREETING: f"{base_prompt} 사용자와 첫 만남이므로 친근하게 인사하고 어떤 도움을 드릴지 물어보세요.",
            
            ConversationState.INFORMATION_SEEKING: f"{base_prompt} 사용자가 정보를 찾고 있습니다. 정확하고 유용한 정보를 제공하되, 확실하지 않은 내용은 그렇다고 말씀하세요.",
            
            ConversationState.TASK_EXECUTION: f"{base_prompt} 사용자가 특정 작업 수행을 요청했습니다. 단계별로 명확한 안내를 제공하고, 필요시 추가 정보를 요청하세요.",
            
            ConversationState.PROBLEM_SOLVING: f"{base_prompt} 사용자가 문제 해결을 위한 도움을 요청했습니다. 문제를 정확히 파악하고 체계적인 해결방안을 제시하세요.",
            
            ConversationState.CASUAL_CHAT: f"{base_prompt} 사용자와 일상적인 대화를 나누고 있습니다. 자연스럽고 친근한 톤으로 대화를 이어가세요.",
            
            ConversationState.ENDING: f"{base_prompt} 대화가 마무리되는 단계입니다. 따뜻한 작별인사와 함께 추가로 도움이 필요하면 언제든 연락하라고 안내하세요."
        }
        
        system_prompt = state_prompts.get(state, base_prompt)
        
        # 주제 변경 정보 추가
        if flow_result.get('topic_changed'):
            system_prompt += f"\n\n참고: 사용자가 새로운 주제로 대화를 전환했습니다. 이전 맥락을 고려하되 새 주제에 집중하세요."
        
        return system_prompt
    
    def get_conversation_history(self, session_id: str, limit: int = 20) -> Dict[str, Any]:
        """
        대화 히스토리 조회
        
        Args:
            session_id: 세션 ID
            limit: 메시지 제한 수
            
        Returns:
            Dict[str, Any]: 대화 히스토리
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            return {'error': '세션을 찾을 수 없습니다'}
        
        recent_messages = session.messages[-limit:] if len(session.messages) > limit else session.messages
        
        return {
            'session_id': session_id,
            'user_id': session.user_id,
            'current_state': session.current_state.value,
            'created_at': session.created_at.isoformat(),
            'last_activity': session.last_activity.isoformat(),
            'total_tokens': session.total_tokens,
            'message_count': len(session.messages),
            'messages': [
                {
                    'id': msg.id,
                    'role': msg.role.value,
                    'content': msg.content,
                    'timestamp': msg.timestamp.isoformat(),
                    'tokens': msg.tokens,
                    'importance_score': msg.importance_score
                }
                for msg in recent_messages
            ],
            'context_metadata': session.context_metadata
        }
    
    def cleanup_sessions(self):
        """만료된 세션 정리"""
        self.session_manager.cleanup_expired_sessions()

# Streamlit 웹 인터페이스
def streamlit_app():
    """Streamlit 기반 웹 인터페이스"""
    st.set_page_config(
        page_title="멀티턴 대화 챗봇",
        page_icon="💬",
        layout="wide"
    )
    
    st.title("💬 AI 챗봇 멘토링 - 4차시: 멀티턴 대화 관리")
    st.write("세션 관리, 컨텍스트 최적화, 대화 상태 추적을 체험해보세요!")
    
    # 세션 상태 초기화
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = MultiTurnChatbot()
    
    if 'current_session_id' not in st.session_state:
        user_id = st.sidebar.text_input("사용자 ID", value="user123")
        if st.sidebar.button("새 대화 시작"):
            session_id = st.session_state.chatbot.start_conversation(user_id)
            st.session_state.current_session_id = session_id
            st.success(f"새 대화가 시작되었습니다! 세션 ID: {session_id[:8]}...")
    
    # 사이드바 - 세션 정보
    with st.sidebar:
        if 'current_session_id' in st.session_state:
            st.header("📊 세션 정보")
            
            # 대화 히스토리 조회
            if st.button("히스토리 새로고침"):
                history = st.session_state.chatbot.get_conversation_history(
                    st.session_state.current_session_id
                )
                st.session_state.conversation_history = history
            
            if 'conversation_history' in st.session_state:
                history = st.session_state.conversation_history
                
                st.metric("현재 상태", history['current_state'])
                st.metric("메시지 수", history['message_count'])
                st.metric("총 토큰", history['total_tokens'])
                
                # 세션 지속 시간
                created = datetime.fromisoformat(history['created_at'])
                duration = datetime.now() - created
                st.metric("세션 지속시간", f"{int(duration.total_seconds())}초")
        
        st.divider()
        
        # 설정
        st.header("⚙️ 설정")
        st.info(f"최대 컨텍스트: {st.session_state.chatbot.context_optimizer.max_tokens} 토큰")
        st.info(f"세션 만료: {st.session_state.chatbot.session_manager.session_timeout}초")
        
        # 세션 정리
        if st.button("만료된 세션 정리"):
            st.session_state.chatbot.cleanup_sessions()
            st.success("세션 정리 완료!")
    
    # 메인 영역 - 채팅
    if 'current_session_id' in st.session_state:
        st.header("💬 대화")
        
        # 대화 히스토리 표시
        if 'conversation_history' in st.session_state:
            messages = st.session_state.conversation_history['messages']
            
            for msg in messages:
                if msg['role'] in ['user', 'assistant']:
                    with st.chat_message(msg['role']):
                        st.write(msg['content'])
                        
                        # 메타데이터 표시
                        if msg['role'] == 'assistant':
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.caption(f"🎯 {msg['tokens']} 토큰")
                            with col2:
                                st.caption(f"⭐ 중요도: {msg['importance_score']:.2f}")
                            with col3:
                                st.caption(f"🕐 {msg['timestamp'][11:19]}")
        
        # 새 메시지 입력
        if user_input := st.chat_input("메시지를 입력하세요..."):
            # 사용자 메시지 표시
            with st.chat_message("user"):
                st.write(user_input)
            
            # AI 응답 생성
            with st.chat_message("assistant"):
                with st.spinner("응답 생성 중..."):
                    result = st.session_state.chatbot.chat(
                        st.session_state.current_session_id, 
                        user_input
                    )
                
                if 'error' not in result:
                    st.write(result['response'])
                    
                    # 응답 메타데이터
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.caption(f"🎯 {result['tokens_used']} 토큰")
                    with col2:
                        st.caption(f"🔄 상태: {result['conversation_state']}")
                    with col3:
                        st.caption(f"⏱️ {result['processing_time']:.2f}초")
                    
                    # 상세 분석 결과
                    with st.expander("🔍 분석 결과", expanded=False):
                        st.json({
                            'flow_analysis': result['flow_analysis'],
                            'context_optimization': result['context_optimization'],
                            'session_stats': result['session_stats']
                        })
                else:
                    st.error(result['error'])
            
            # 히스토리 업데이트
            if 'error' not in result:
                history = st.session_state.chatbot.get_conversation_history(
                    st.session_state.current_session_id
                )
                st.session_state.conversation_history = history
                st.rerun()
    else:
        st.info("👈 사이드바에서 새 대화를 시작해주세요!")
    
    # 디버그 정보
    with st.expander("🔧 시스템 정보", expanded=False):
        if 'current_session_id' in st.session_state:
            st.write("**세션 ID:**", st.session_state.current_session_id)
        
        st.write("**Redis 연결 상태:**", 
                "연결됨" if st.session_state.chatbot.session_manager.client else "메모리 모드")
        
        st.write("**구성 요소:**")
        st.write("- 세션 관리: Redis/메모리 백업")
        st.write("- 컨텍스트 최적화: 토큰 기반 압축")
        st.write("- 의도 분류: 키워드 + LLM")
        st.write("- 상태 관리: 대화 흐름 추적")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        # CLI 데모 모드
        print("=== 멀티턴 대화 챗봇 CLI 데모 ===")
        
        chatbot = MultiTurnChatbot()
        user_id = input("사용자 ID를 입력하세요: ") or "cli_user"
        
        session_id = chatbot.start_conversation(user_id)
        print(f"새 대화 시작! 세션 ID: {session_id[:8]}...")
        print("종료하려면 'quit' 또는 'exit'를 입력하세요.\n")
        
        while True:
            try:
                user_input = input("\n사용자: ")
                if user_input.lower() in ['quit', 'exit']:
                    print("대화를 종료합니다.")
                    break
                
                if not user_input.strip():
                    continue
                
                result = chatbot.chat(session_id, user_input)
                
                if 'error' not in result:
                    print(f"\nAI ({result['conversation_state']}): {result['response']}")
                    print(f"📊 토큰: {result['tokens_used']}, 시간: {result['processing_time']:.2f}초")
                    
                    if result['flow_analysis'].get('topic_changed'):
                        print("🔄 주제 변경이 감지되었습니다.")
                else:
                    print(f"❌ 오류: {result['error']}")
            
            except KeyboardInterrupt:
                print("\n\n대화를 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
    else:
        # Streamlit 모드
        streamlit_app()