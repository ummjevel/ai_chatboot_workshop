#!/usr/bin/env python3
"""
2차시: 프롬프트 실무 최적화
Author: AI Chatbot Workshop
Date: 2024-08-30
Description: Jinja2 기반 프롬프트 템플릿 시스템, 페르소나 적용, 트러블슈팅 해결
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import hashlib
import random
from jinja2 import Template, Environment, FileSystemLoader
from openai import OpenAI
import streamlit as st

# 로컬 모듈
import sys
sys.path.append('..')
from config import get_config

config = get_config()
logger = logging.getLogger(__name__)

@dataclass
class PromptTemplate:
    """프롬프트 템플릿 구조"""
    name: str
    category: str
    template: str
    variables: List[str]
    description: str
    author: str
    version: str = "1.0"
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class PersonaConfig:
    """페르소나 설정"""
    name: str
    role: str
    tone: str
    expertise: List[str]
    constraints: List[str]
    examples: List[Dict[str, str]]
    brand_guidelines: Optional[Dict[str, Any]] = None

@dataclass
class PromptTestResult:
    """프롬프트 테스트 결과"""
    template_name: str
    test_input: Dict[str, Any]
    generated_prompt: str
    response: str
    tokens_used: int
    processing_time: float
    quality_score: float
    timestamp: datetime

class PromptTemplateEngine:
    """Jinja2 기반 프롬프트 템플릿 엔진"""
    
    def __init__(self, template_dir: str = "templates"):
        """
        Args:
            template_dir: 템플릿 파일이 저장된 디렉토리
        """
        self.template_dir = template_dir
        self.env = Environment(
            loader=FileSystemLoader(template_dir) if os.path.exists(template_dir) else None,
            trim_blocks=True,
            lstrip_blocks=True
        )
        self.templates: Dict[str, PromptTemplate] = {}
        self._load_default_templates()
        logger.info(f"프롬프트 엔진 초기화 완료 - 템플릿 수: {len(self.templates)}")
    
    def _load_default_templates(self):
        """기본 템플릿 로드"""
        default_templates = {
            "general_assistant": PromptTemplate(
                name="general_assistant",
                category="기본",
                template="""당신은 {{persona.role}}입니다. {{persona.tone}} 톤으로 답변해주세요.

전문분야: {{persona.expertise|join(', ')}}

제약사항:
{% for constraint in persona.constraints %}
- {{constraint}}
{% endfor %}

사용자 질문: {{user_question}}

답변 형식: {{response_format}}
답변 길이: {{length_limit}}자 이내""",
                variables=["persona", "user_question", "response_format", "length_limit"],
                description="일반적인 어시스턴트 역할을 위한 기본 템플릿",
                author="AI Workshop"
            ),
            
            "code_reviewer": PromptTemplate(
                name="code_reviewer",
                category="개발",
                template="""당신은 {{years_experience}}년 경력의 {{programming_language}} 전문가입니다.

다음 코드를 리뷰해주세요:

```{{programming_language}}
{{code_snippet}}
```

리뷰 관점:
{% for aspect in review_aspects %}
- {{aspect}}
{% endfor %}

출력 형식:
1. 코드 품질 점수 (1-10)
2. 주요 개선사항
3. 보안 이슈
4. 성능 최적화 제안
5. 개선된 코드 예시""",
                variables=["years_experience", "programming_language", "code_snippet", "review_aspects"],
                description="코드 리뷰를 위한 전문 템플릿",
                author="AI Workshop"
            ),
            
            "customer_support": PromptTemplate(
                name="customer_support",
                category="고객지원",
                template="""안녕하세요! {{company_name}} 고객지원팀의 {{agent_name}}입니다.

브랜드 가이드라인:
- 톤: {{brand_tone}}
- 핵심가치: {{brand_values|join(', ')}}
- 금지어: {{forbidden_words|join(', ')}}

고객 문의:
분류: {{inquiry_category}}
내용: {{customer_message}}
우선순위: {{priority_level}}

{% if previous_interactions %}
이전 상담 내역:
{% for interaction in previous_interactions %}
- {{interaction.date}}: {{interaction.summary}}
{% endfor %}
{% endif %}

응답 가이드라인:
1. 공감적 인사
2. 문제 파악 확인
3. 구체적 해결방안 제시
4. 추가 도움 제안

최대 {{max_length}}자로 답변해주세요.""",
                variables=["company_name", "agent_name", "brand_tone", "brand_values", 
                          "forbidden_words", "inquiry_category", "customer_message", 
                          "priority_level", "previous_interactions", "max_length"],
                description="고객지원을 위한 브랜드 가이드라인 적용 템플릿",
                author="AI Workshop"
            ),
            
            "content_creator": PromptTemplate(
                name="content_creator",
                category="마케팅",
                template="""당신은 {{platform}} 전문 콘텐츠 크리에이터입니다.

콘텐츠 요구사항:
- 주제: {{topic}}
- 타겟 오디언스: {{target_audience}}
- 콘텐츠 유형: {{content_type}}
- 목표: {{marketing_goal}}
- 키워드: {{keywords|join(', ')}}

브랜드 톤앤매너:
- 스타일: {{brand_style}}
- 가치관: {{brand_values|join(', ')}}

{% if competitor_analysis %}
경쟁사 분석:
{% for competitor in competitor_analysis %}
- {{competitor.name}}: {{competitor.strategy}}
{% endfor %}
{% endif %}

콘텐츠 구조:
1. 후크 (첫 줄)
2. 본문 ({{content_length}} 단어)
3. 콜투액션
4. 해시태그 (최대 {{max_hashtags}}개)

SEO를 고려하여 작성하되, 자연스러운 흐름을 유지하세요.""",
                variables=["platform", "topic", "target_audience", "content_type", 
                          "marketing_goal", "keywords", "brand_style", "brand_values",
                          "competitor_analysis", "content_length", "max_hashtags"],
                description="소셜미디어 마케팅 콘텐츠 생성 템플릿",
                author="AI Workshop"
            )
        }
        
        self.templates.update(default_templates)
    
    def add_template(self, template: PromptTemplate):
        """새 템플릿 추가"""
        self.templates[template.name] = template
        logger.info(f"템플릿 추가: {template.name}")
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """템플릿 조회"""
        return self.templates.get(name)
    
    def list_templates(self, category: Optional[str] = None) -> List[PromptTemplate]:
        """템플릿 목록 조회"""
        if category:
            return [t for t in self.templates.values() if t.category == category]
        return list(self.templates.values())
    
    def render_prompt(self, template_name: str, variables: Dict[str, Any]) -> str:
        """
        프롬프트 렌더링
        
        Args:
            template_name: 템플릿 이름
            variables: 템플릿 변수
            
        Returns:
            str: 렌더링된 프롬프트
            
        Raises:
            ValueError: 템플릿이 존재하지 않거나 필수 변수가 누락된 경우
        """
        template_obj = self.get_template(template_name)
        if not template_obj:
            raise ValueError(f"템플릿을 찾을 수 없습니다: {template_name}")
        
        # 필수 변수 확인
        missing_vars = set(template_obj.variables) - set(variables.keys())
        if missing_vars:
            logger.warning(f"누락된 변수: {missing_vars}")
        
        try:
            jinja_template = Template(template_obj.template)
            rendered = jinja_template.render(**variables)
            
            logger.debug(f"프롬프트 렌더링 완료 - 템플릿: {template_name}, 길이: {len(rendered)}")
            return rendered
        
        except Exception as e:
            logger.error(f"프롬프트 렌더링 실패: {e}")
            raise

class PersonaManager:
    """페르소나 관리자"""
    
    def __init__(self):
        self.personas: Dict[str, PersonaConfig] = {}
        self._load_default_personas()
        logger.info(f"페르소나 관리자 초기화 - 페르소나 수: {len(self.personas)}")
    
    def _load_default_personas(self):
        """기본 페르소나 로드"""
        default_personas = {
            "friendly_assistant": PersonaConfig(
                name="친근한 도우미",
                role="도움이 되는 AI 어시스턴트",
                tone="친근하고 따뜻한",
                expertise=["일반 상식", "문제 해결", "정보 제공"],
                constraints=[
                    "항상 정중하고 예의바르게 대답",
                    "확실하지 않은 정보는 추측하지 않음",
                    "개인정보는 절대 요청하지 않음"
                ],
                examples=[
                    {
                        "input": "안녕하세요",
                        "output": "안녕하세요! 무엇을 도와드릴까요? 궁금한 것이 있으시면 언제든 말씀해주세요."
                    }
                ]
            ),
            
            "technical_expert": PersonaConfig(
                name="기술 전문가",
                role="10년 경력의 소프트웨어 엔지니어",
                tone="전문적이면서도 이해하기 쉬운",
                expertise=["Python", "웹개발", "AI/ML", "시스템 아키텍처", "데이터베이스"],
                constraints=[
                    "정확한 기술 정보만 제공",
                    "예시 코드는 실행 가능해야 함",
                    "최신 기술 동향 반영",
                    "초보자도 이해할 수 있도록 설명"
                ],
                examples=[
                    {
                        "input": "Python으로 API를 어떻게 만들어요?",
                        "output": "FastAPI나 Flask를 사용하면 쉽게 만들 수 있습니다. 예를 들어 FastAPI로 간단한 API를 만들어보겠습니다..."
                    }
                ]
            ),
            
            "brand_ambassador": PersonaConfig(
                name="브랜드 앰배서더",
                role="기업 브랜드 대변인",
                tone="전문적이고 신뢰할 수 있는",
                expertise=["브랜드 커뮤니케이션", "고객 서비스", "마케팅"],
                constraints=[
                    "브랜드 가치와 일치하는 메시지",
                    "경쟁사 언급 금지",
                    "부정적 표현 최소화",
                    "항상 솔루션 중심적 접근"
                ],
                examples=[
                    {
                        "input": "제품에 문제가 있어요",
                        "output": "불편을 끼쳐드려 죄송합니다. 빠른 해결을 위해 구체적인 상황을 알려주시겠어요? 최선을 다해 도움드리겠습니다."
                    }
                ],
                brand_guidelines={
                    "tone": "professional_friendly",
                    "values": ["고객중심", "혁신", "신뢰", "품질"],
                    "forbidden_words": ["문제", "불가능", "안됨", "모름"],
                    "preferred_words": ["해결", "가능", "도움", "지원"]
                }
            )
        }
        
        self.personas.update(default_personas)
    
    def add_persona(self, persona: PersonaConfig):
        """새 페르소나 추가"""
        self.personas[persona.name] = persona
        logger.info(f"페르소나 추가: {persona.name}")
    
    def get_persona(self, name: str) -> Optional[PersonaConfig]:
        """페르소나 조회"""
        return self.personas.get(name)
    
    def list_personas(self) -> List[PersonaConfig]:
        """페르소나 목록"""
        return list(self.personas.values())

class PromptOptimizer:
    """프롬프트 최적화 및 A/B 테스트"""
    
    def __init__(self, openai_client: OpenAI, template_engine: PromptTemplateEngine):
        self.client = openai_client
        self.template_engine = template_engine
        self.test_results: List[PromptTestResult] = []
        logger.info("프롬프트 최적화 엔진 초기화 완료")
    
    def generate_prompt_variations(self, base_template: str, 
                                 variations_config: Dict[str, List[str]]) -> List[str]:
        """
        프롬프트 변형 생성
        
        Args:
            base_template: 기본 템플릿
            variations_config: 변형 설정 {'temperature': [0.1, 0.7, 0.9], ...}
            
        Returns:
            List[str]: 변형된 프롬프트들
        """
        variations = []
        
        # 온도 변형
        if 'temperature_hints' in variations_config:
            for temp_hint in variations_config['temperature_hints']:
                variation = f"{base_template}\n\n{temp_hint}"
                variations.append(variation)
        
        # 톤 변형
        if 'tone_variations' in variations_config:
            for tone in variations_config['tone_variations']:
                variation = base_template.replace("답변해주세요", f"{tone} 톤으로 답변해주세요")
                variations.append(variation)
        
        # 길이 제한 변형
        if 'length_limits' in variations_config:
            for length in variations_config['length_limits']:
                variation = f"{base_template}\n\n답변은 {length}자 이내로 작성해주세요."
                variations.append(variation)
        
        logger.info(f"프롬프트 변형 {len(variations)}개 생성")
        return variations
    
    def test_prompt_quality(self, prompt: str, test_inputs: List[str]) -> float:
        """
        프롬프트 품질 평가
        
        Args:
            prompt: 테스트할 프롬프트
            test_inputs: 테스트 입력들
            
        Returns:
            float: 품질 점수 (0-100)
        """
        scores = []
        
        for test_input in test_inputs:
            try:
                # API 호출
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": test_input}
                ]
                
                response = self.client.chat.completions.create(
                    model=config.llm.openai_model,
                    messages=messages,
                    max_tokens=200,
                    temperature=0.1  # 일관성을 위해 낮은 temperature 사용
                )
                
                answer = response.choices[0].message.content
                
                # 품질 평가 (간단한 휴리스틱)
                score = self._calculate_quality_score(test_input, answer, prompt)
                scores.append(score)
                
            except Exception as e:
                logger.error(f"품질 테스트 실패: {e}")
                scores.append(0)
        
        avg_score = sum(scores) / len(scores) if scores else 0
        logger.info(f"평균 품질 점수: {avg_score:.1f}")
        return avg_score
    
    def _calculate_quality_score(self, input_text: str, output_text: str, prompt: str) -> float:
        """품질 점수 계산 (휴리스틱)"""
        score = 50  # 기본 점수
        
        # 길이 적절성 (너무 짧거나 길지 않음)
        if 50 <= len(output_text) <= 500:
            score += 20
        
        # 관련성 (입력과 출력의 키워드 매칭)
        input_words = set(input_text.lower().split())
        output_words = set(output_text.lower().split())
        relevance = len(input_words & output_words) / len(input_words) if input_words else 0
        score += relevance * 20
        
        # 구조화 (문장 구조)
        if '. ' in output_text or '\n' in output_text:
            score += 10
        
        return min(100, max(0, score))
    
    def run_ab_test(self, prompt_a: str, prompt_b: str, 
                   test_inputs: List[str], test_name: str = "A/B Test") -> Dict[str, Any]:
        """
        A/B 테스트 실행
        
        Args:
            prompt_a: 프롬프트 A
            prompt_b: 프롬프트 B  
            test_inputs: 테스트 입력들
            test_name: 테스트 이름
            
        Returns:
            Dict: 테스트 결과
        """
        logger.info(f"A/B 테스트 시작: {test_name}")
        start_time = time.time()
        
        results_a = []
        results_b = []
        
        for i, test_input in enumerate(test_inputs):
            # 프롬프트 A 테스트
            result_a = self._test_single_prompt(f"A_{i}", prompt_a, test_input)
            results_a.append(result_a)
            
            # 프롬프트 B 테스트
            result_b = self._test_single_prompt(f"B_{i}", prompt_b, test_input)
            results_b.append(result_b)
            
            # 저장
            self.test_results.extend([result_a, result_b])
        
        # 결과 분석
        avg_score_a = sum(r.quality_score for r in results_a) / len(results_a)
        avg_score_b = sum(r.quality_score for r in results_b) / len(results_b)
        avg_time_a = sum(r.processing_time for r in results_a) / len(results_a)
        avg_time_b = sum(r.processing_time for r in results_b) / len(results_b)
        avg_tokens_a = sum(r.tokens_used for r in results_a) / len(results_a)
        avg_tokens_b = sum(r.tokens_used for r in results_b) / len(results_b)
        
        total_time = time.time() - start_time
        
        # 승자 결정
        winner = "A" if avg_score_a > avg_score_b else "B"
        score_diff = abs(avg_score_a - avg_score_b)
        
        result = {
            "test_name": test_name,
            "winner": winner,
            "score_difference": score_diff,
            "results": {
                "prompt_a": {
                    "avg_quality_score": avg_score_a,
                    "avg_processing_time": avg_time_a,
                    "avg_tokens": avg_tokens_a,
                    "results": results_a
                },
                "prompt_b": {
                    "avg_quality_score": avg_score_b,
                    "avg_processing_time": avg_time_b,
                    "avg_tokens": avg_tokens_b,
                    "results": results_b
                }
            },
            "test_duration": total_time,
            "total_tests": len(test_inputs) * 2,
            "timestamp": datetime.now()
        }
        
        logger.info(f"A/B 테스트 완료 - 승자: {winner}, 점수차: {score_diff:.1f}")
        return result
    
    def _test_single_prompt(self, test_id: str, prompt: str, test_input: str) -> PromptTestResult:
        """단일 프롬프트 테스트"""
        start_time = time.time()
        
        try:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": test_input}
            ]
            
            response = self.client.chat.completions.create(
                model=config.llm.openai_model,
                messages=messages,
                max_tokens=300,
                temperature=0.7,
                seed=42  # 일관성을 위한 seed 고정
            )
            
            processing_time = time.time() - start_time
            answer = response.choices[0].message.content
            tokens = response.usage.total_tokens
            quality = self._calculate_quality_score(test_input, answer, prompt)
            
            return PromptTestResult(
                template_name=test_id,
                test_input={"user_input": test_input},
                generated_prompt=prompt,
                response=answer,
                tokens_used=tokens,
                processing_time=processing_time,
                quality_score=quality,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"프롬프트 테스트 실패: {e}")
            return PromptTestResult(
                template_name=test_id,
                test_input={"user_input": test_input},
                generated_prompt=prompt,
                response=f"오류: {str(e)}",
                tokens_used=0,
                processing_time=time.time() - start_time,
                quality_score=0,
                timestamp=datetime.now()
            )

class TroubleshootingSolver:
    """프롬프트 트러블슈팅 해결사"""
    
    @staticmethod
    def ensure_consistency(base_prompt: str) -> str:
        """일관성 확보를 위한 프롬프트 개선"""
        consistency_suffix = """

[중요한 지침]
- temperature=0으로 설정하여 일관된 응답을 생성하세요
- 동일한 질문에는 항상 동일한 구조로 답변하세요
- 예측 가능하고 신뢰할 수 있는 응답을 제공하세요"""
        
        return base_prompt + consistency_suffix
    
    @staticmethod
    def compress_prompt(long_prompt: str, max_length: int = 1000) -> str:
        """프롬프트 압축 (토큰 수 최적화)"""
        if len(long_prompt) <= max_length:
            return long_prompt
        
        # 압축 전략
        compressed = long_prompt
        
        # 1. 중복 공백 제거
        import re
        compressed = re.sub(r'\s+', ' ', compressed)
        
        # 2. 불필요한 구문 제거
        unnecessary_phrases = [
            "please", "kindly", "I would like you to",
            "Could you", "Would you mind", "It would be great if"
        ]
        
        for phrase in unnecessary_phrases:
            compressed = compressed.replace(phrase, "")
        
        # 3. 핵심 내용만 유지
        if len(compressed) > max_length:
            # 문장 단위로 자르기
            sentences = compressed.split('. ')
            compressed = '. '.join(sentences[:max_length//50])  # 대략 추정
        
        logger.info(f"프롬프트 압축: {len(long_prompt)} -> {len(compressed)} 문자")
        return compressed.strip()
    
    @staticmethod
    def fix_response_format(prompt: str, desired_format: str) -> str:
        """응답 형식 제어 개선"""
        format_instructions = {
            "json": "\n\n응답을 반드시 유효한 JSON 형식으로만 제공하세요. 다른 설명은 포함하지 마세요.",
            "markdown": "\n\n응답을 마크다운 형식으로 작성하세요. 헤더, 리스트, 코드 블록을 활용하세요.",
            "bullet_points": "\n\n응답을 불릿 포인트 형태로 정리해주세요:\n- 첫 번째 요점\n- 두 번째 요점\n- ...",
            "numbered_list": "\n\n응답을 번호 매긴 목록으로 작성해주세요:\n1. 첫 번째 항목\n2. 두 번째 항목\n...",
            "paragraph": "\n\n응답을 잘 구조화된 문단 형태로 작성해주세요. 각 문단은 하나의 주요 아이디어를 다루세요."
        }
        
        format_instruction = format_instructions.get(desired_format, "")
        return prompt + format_instruction

def create_streamlit_ui():
    """Streamlit UI 생성"""
    st.set_page_config(
        page_title="AI 챗봇 멘토링 - 2차시",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 AI 챗봇 멘토링 - 2차시: 프롬프트 최적화")
    st.caption("Jinja2 템플릿, 페르소나 적용, A/B 테스트")
    
    # 초기화
    if 'template_engine' not in st.session_state:
        st.session_state.template_engine = PromptTemplateEngine()
        st.session_state.persona_manager = PersonaManager()
        st.session_state.optimizer = PromptOptimizer(
            OpenAI(api_key=config.llm.openai_api_key),
            st.session_state.template_engine
        )
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 템플릿 선택
        template_names = list(st.session_state.template_engine.templates.keys())
        selected_template = st.selectbox("템플릿 선택", template_names, index=0)
        
        # 페르소나 선택
        persona_names = list(st.session_state.persona_manager.personas.keys())
        selected_persona = st.selectbox("페르소나 선택", persona_names, index=0)
        
        st.divider()
        
        # 테스트 모드
        test_mode = st.radio(
            "테스트 모드",
            ["단일 테스트", "A/B 테스트", "품질 분석"]
        )
    
    # 메인 콘텐츠
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 프롬프트 템플릿")
        
        # 선택된 템플릿 정보
        template = st.session_state.template_engine.get_template(selected_template)
        if template:
            st.subheader(f"📋 {template.name}")
            st.write(f"**카테고리**: {template.category}")
            st.write(f"**설명**: {template.description}")
            st.write(f"**작성자**: {template.author}")
            
            # 템플릿 내용
            st.text_area("템플릿 내용", template.template, height=200, disabled=True)
            
            # 필수 변수
            st.write("**필수 변수**:")
            for var in template.variables:
                st.write(f"- `{var}`")
    
    with col2:
        st.header("🎭 페르소나 설정")
        
        # 선택된 페르소나 정보
        persona = st.session_state.persona_manager.get_persona(selected_persona)
        if persona:
            st.subheader(f"👤 {persona.name}")
            st.write(f"**역할**: {persona.role}")
            st.write(f"**톤**: {persona.tone}")
            
            # 전문분야
            st.write("**전문분야**:")
            for expertise in persona.expertise:
                st.write(f"- {expertise}")
            
            # 제약사항
            st.write("**제약사항**:")
            for constraint in persona.constraints:
                st.write(f"- {constraint}")
    
    st.divider()
    
    # 테스트 섹션
    if test_mode == "단일 테스트":
        st.header("🧪 단일 프롬프트 테스트")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # 변수 입력
            st.subheader("변수 입력")
            variables = {}
            
            if template:
                for var in template.variables:
                    if var == "persona":
                        variables[var] = persona
                    elif var in ["user_question", "customer_message", "topic"]:
                        variables[var] = st.text_area(f"{var}:", height=100)
                    elif var in ["response_format", "content_type", "brand_tone"]:
                        variables[var] = st.text_input(f"{var}:")
                    elif var in ["length_limit", "max_length", "years_experience"]:
                        variables[var] = st.number_input(f"{var}:", min_value=1, value=200)
                    elif var in ["keywords", "brand_values", "expertise"]:
                        text_input = st.text_input(f"{var} (쉼표 구분):")
                        variables[var] = [x.strip() for x in text_input.split(',') if x.strip()]
                    else:
                        variables[var] = st.text_input(f"{var}:")
            
            if st.button("프롬프트 생성"):
                try:
                    rendered_prompt = st.session_state.template_engine.render_prompt(
                        selected_template, variables
                    )
                    st.session_state.generated_prompt = rendered_prompt
                except Exception as e:
                    st.error(f"프롬프트 생성 실패: {e}")
        
        with col2:
            # 생성된 프롬프트
            st.subheader("생성된 프롬프트")
            if hasattr(st.session_state, 'generated_prompt'):
                st.text_area("", st.session_state.generated_prompt, height=300)
                
                # 테스트 실행
                test_input = st.text_input("테스트 입력:")
                if st.button("응답 생성"):
                    with st.spinner("AI 응답 생성 중..."):
                        try:
                            client = OpenAI(api_key=config.llm.openai_api_key)
                            messages = [
                                {"role": "system", "content": st.session_state.generated_prompt},
                                {"role": "user", "content": test_input}
                            ]
                            
                            response = client.chat.completions.create(
                                model=config.llm.openai_model,
                                messages=messages,
                                max_tokens=500,
                                temperature=0.7
                            )
                            
                            st.subheader("AI 응답")
                            st.write(response.choices[0].message.content)
                            
                            # 메타데이터
                            st.caption(f"토큰 사용: {response.usage.total_tokens}")
                            
                        except Exception as e:
                            st.error(f"응답 생성 실패: {e}")
    
    elif test_mode == "A/B 테스트":
        st.header("⚖️ A/B 테스트")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("프롬프트 A")
            prompt_a = st.text_area("프롬프트 A:", height=150)
        
        with col2:
            st.subheader("프롬프트 B")
            prompt_b = st.text_area("프롬프트 B:", height=150)
        
        # 테스트 입력
        st.subheader("테스트 입력 (줄바꿈으로 구분)")
        test_inputs_text = st.text_area("", height=100)
        test_inputs = [x.strip() for x in test_inputs_text.split('\n') if x.strip()]
        
        if st.button("A/B 테스트 실행") and prompt_a and prompt_b and test_inputs:
            with st.spinner("A/B 테스트 실행 중..."):
                try:
                    result = st.session_state.optimizer.run_ab_test(
                        prompt_a, prompt_b, test_inputs, "Manual A/B Test"
                    )
                    
                    # 결과 표시
                    st.subheader("🏆 테스트 결과")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("승자", result["winner"])
                    with col2:
                        st.metric("점수차", f"{result['score_difference']:.1f}")
                    with col3:
                        st.metric("총 테스트", result["total_tests"])
                    
                    # 상세 결과
                    results_a = result["results"]["prompt_a"]
                    results_b = result["results"]["prompt_b"]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("📊 프롬프트 A")
                        st.metric("평균 품질 점수", f"{results_a['avg_quality_score']:.1f}")
                        st.metric("평균 처리 시간", f"{results_a['avg_processing_time']:.2f}초")
                        st.metric("평균 토큰 수", f"{results_a['avg_tokens']:.0f}")
                    
                    with col2:
                        st.subheader("📊 프롬프트 B")
                        st.metric("평균 품질 점수", f"{results_b['avg_quality_score']:.1f}")
                        st.metric("평균 처리 시간", f"{results_b['avg_processing_time']:.2f}초")
                        st.metric("평균 토큰 수", f"{results_b['avg_tokens']:.0f}")
                    
                except Exception as e:
                    st.error(f"A/B 테스트 실패: {e}")
    
    elif test_mode == "품질 분석":
        st.header("📈 프롬프트 품질 분석")
        
        # 분석할 프롬프트
        analysis_prompt = st.text_area("분석할 프롬프트:", height=200)
        
        # 테스트 케이스
        st.subheader("테스트 케이스")
        test_cases = st.text_area("테스트 입력들 (줄바꿈으로 구분):", height=100)
        test_list = [x.strip() for x in test_cases.split('\n') if x.strip()]
        
        if st.button("품질 분석 시작") and analysis_prompt and test_list:
            with st.spinner("품질 분석 중..."):
                try:
                    quality_score = st.session_state.optimizer.test_prompt_quality(
                        analysis_prompt, test_list
                    )
                    
                    # 결과 표시
                    st.subheader("📊 품질 분석 결과")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("전체 품질 점수", f"{quality_score:.1f}/100")
                    with col2:
                        grade = "A" if quality_score >= 80 else "B" if quality_score >= 60 else "C"
                        st.metric("등급", grade)
                    with col3:
                        st.metric("테스트 케이스 수", len(test_list))
                    
                    # 개선 제안
                    st.subheader("💡 개선 제안")
                    if quality_score < 70:
                        st.warning("품질 점수가 낮습니다. 다음을 고려해보세요:")
                        st.write("- 더 구체적인 지시사항 추가")
                        st.write("- 예시 포함")
                        st.write("- 응답 형식 명시")
                        st.write("- 제약사항 명확화")
                    else:
                        st.success("좋은 품질의 프롬프트입니다!")
                    
                    # 최적화된 버전 제안
                    st.subheader("🔧 최적화된 프롬프트 제안")
                    
                    # 일관성 개선
                    consistent_prompt = TroubleshootingSolver.ensure_consistency(analysis_prompt)
                    st.text_area("일관성 개선 버전:", consistent_prompt, height=200)
                    
                    # 압축 버전
                    compressed_prompt = TroubleshootingSolver.compress_prompt(analysis_prompt)
                    st.text_area("압축 버전 (토큰 최적화):", compressed_prompt, height=150)
                    
                except Exception as e:
                    st.error(f"품질 분석 실패: {e}")
    
    # 템플릿 및 페르소나 관리
    st.divider()
    with st.expander("🛠️ 템플릿 & 페르소나 관리"):
        tab1, tab2 = st.tabs(["새 템플릿 추가", "새 페르소나 추가"])
        
        with tab1:
            st.subheader("새 템플릿 추가")
            new_template_name = st.text_input("템플릿 이름:")
            new_template_category = st.text_input("카테고리:")
            new_template_content = st.text_area("템플릿 내용:", height=200)
            new_template_vars = st.text_input("변수 목록 (쉼표 구분):")
            new_template_desc = st.text_input("설명:")
            
            if st.button("템플릿 추가"):
                if all([new_template_name, new_template_category, new_template_content]):
                    variables = [v.strip() for v in new_template_vars.split(',') if v.strip()]
                    new_template = PromptTemplate(
                        name=new_template_name,
                        category=new_template_category,
                        template=new_template_content,
                        variables=variables,
                        description=new_template_desc,
                        author="User"
                    )
                    st.session_state.template_engine.add_template(new_template)
                    st.success(f"템플릿 '{new_template_name}' 추가 완료!")
                    st.rerun()
        
        with tab2:
            st.subheader("새 페르소나 추가")
            new_persona_name = st.text_input("페르소나 이름:")
            new_persona_role = st.text_input("역할:")
            new_persona_tone = st.text_input("톤:")
            new_persona_expertise = st.text_input("전문분야 (쉼표 구분):")
            new_persona_constraints = st.text_input("제약사항 (쉼표 구분):")
            
            if st.button("페르소나 추가"):
                if all([new_persona_name, new_persona_role, new_persona_tone]):
                    expertise = [e.strip() for e in new_persona_expertise.split(',') if e.strip()]
                    constraints = [c.strip() for c in new_persona_constraints.split(',') if c.strip()]
                    
                    new_persona = PersonaConfig(
                        name=new_persona_name,
                        role=new_persona_role,
                        tone=new_persona_tone,
                        expertise=expertise,
                        constraints=constraints,
                        examples=[]
                    )
                    st.session_state.persona_manager.add_persona(new_persona)
                    st.success(f"페르소나 '{new_persona_name}' 추가 완료!")
                    st.rerun()

def run_cli_demo():
    """CLI 데모"""
    print("=== AI 챗봇 멘토링 2차시: 프롬프트 최적화 ===")
    
    # 초기화
    template_engine = PromptTemplateEngine()
    persona_manager = PersonaManager()
    
    print(f"로드된 템플릿: {len(template_engine.templates)}개")
    print(f"로드된 페르소나: {len(persona_manager.personas)}개")
    
    # 템플릿 목록 표시
    print("\n📋 사용 가능한 템플릿:")
    for i, template in enumerate(template_engine.list_templates(), 1):
        print(f"{i}. {template.name} ({template.category})")
    
    # 사용자 선택
    while True:
        try:
            choice = input("\n템플릿 번호를 선택하세요 (0: 종료): ")
            if choice == '0':
                break
            
            template_idx = int(choice) - 1
            templates_list = template_engine.list_templates()
            
            if 0 <= template_idx < len(templates_list):
                selected_template = templates_list[template_idx]
                print(f"\n선택된 템플릿: {selected_template.name}")
                print(f"설명: {selected_template.description}")
                print(f"필수 변수: {', '.join(selected_template.variables)}")
                
                # 변수 입력
                variables = {}
                for var in selected_template.variables:
                    if var == "persona":
                        print("\n페르소나 목록:")
                        personas = persona_manager.list_personas()
                        for i, p in enumerate(personas):
                            print(f"{i+1}. {p.name}")
                        
                        persona_choice = int(input("페르소나 선택: ")) - 1
                        if 0 <= persona_choice < len(personas):
                            variables[var] = personas[persona_choice]
                    else:
                        value = input(f"{var}: ")
                        # 타입 추정
                        if var.endswith('_limit') or var.startswith('max_') or 'years' in var:
                            try:
                                variables[var] = int(value)
                            except:
                                variables[var] = value
                        elif ',' in value:
                            variables[var] = [x.strip() for x in value.split(',')]
                        else:
                            variables[var] = value
                
                # 프롬프트 렌더링
                try:
                    rendered = template_engine.render_prompt(selected_template.name, variables)
                    print(f"\n🎯 생성된 프롬프트:")
                    print("=" * 50)
                    print(rendered)
                    print("=" * 50)
                    
                    # 테스트 여부
                    if input("\nAI 응답을 테스트해보시겠습니까? (y/n): ").lower() == 'y':
                        test_input = input("테스트 입력: ")
                        
                        client = OpenAI(api_key=config.llm.openai_api_key)
                        messages = [
                            {"role": "system", "content": rendered},
                            {"role": "user", "content": test_input}
                        ]
                        
                        print("\n🤖 AI 응답:")
                        response = client.chat.completions.create(
                            model=config.llm.openai_model,
                            messages=messages,
                            max_tokens=500,
                            temperature=0.7
                        )
                        
                        print(response.choices[0].message.content)
                        print(f"\n사용된 토큰: {response.usage.total_tokens}")
                
                except Exception as e:
                    print(f"오류: {e}")
            else:
                print("잘못된 선택입니다.")
        
        except (ValueError, KeyboardInterrupt):
            print("프로그램을 종료합니다.")
            break

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        run_cli_demo()
    else:
        create_streamlit_ui()