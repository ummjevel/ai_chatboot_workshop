# 🤖 AI 챗봇 멘토링 워크샵

> **7차시 실무형 AI 챗봇 개발 완전 가이드**  
> 개발자를 위한 핸즈온 중심(90%) 실무 멘토링 프로그램

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-green.svg)](https://openai.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📚 워크샵 개요

이 워크샵은 **실무에서 바로 사용 가능한** AI 챗봇 시스템을 구축하는 방법을 학습합니다. 단순한 API 호출이 아닌, **프로덕션 레디** 수준의 코드를 작성하여 실제 서비스에 적용할 수 있도록 설계되었습니다.

### 🎯 대상
- **중급 개발자** (Python 기본 문법 이해)
- **컴공과 재학생** (프로그래밍 경험 있음)
- **AI/ML에 관심 있는 개발자**
- **실무 적용을 목표로 하는 분**

### ⭐ 특징
- **핸즈온 중심** - 90% 코딩 실습
- **실무 코드** - 프로덕션에서 사용 가능한 품질
- **완전한 구현** - 뼈대가 아닌 동작하는 완성품
- **확장 가능한 설계** - 기업 서비스 수준의 아키텍처

---

## 🗓️ 커리큘럼 (7차시)

| 차시 | 주제 | 핵심 학습 내용 | 결과물 |
|------|------|---------------|--------|
| **1차시** | **[기본 챗봇 구현](lesson1_basic_chatbot.py)** | OpenAI API 연동, 스트리밍 응답, 토큰 모니터링 | 동작하는 웹 챗봇 |
| **2차시** | **프롬프트 최적화** | 프롬프트 템플릿, 페르소나 적용, A/B 테스트 | 프롬프트 관리 시스템 |
| **3차시** | **RAG 구현** | 문서 검색, 벡터 DB, 지식 기반 답변 | PDF 업로드 QA 봇 |
| **4차시** | **대화 상태 관리** | 멀티턴 대화, 컨텍스트 윈도우, 세션 관리 | 연속 대화 챗봇 |
| **5차시** | **외부 연동** | Tool Calling, 실시간 데이터, API 통합 | 기능 확장 챗봇 |
| **6차시** | **성능 최적화** | 캐싱, 비동기, 비용 최적화, 모니터링 | 고성능 챗봇 시스템 |
| **7차시** | **배포 및 통합** | FastAPI, Docker, 클라우드 배포 | 프로덕션 서비스 |

---

## 🚀 빠른 시작

### 1️⃣ 환경 설정

```bash
# 저장소 복제
git clone https://github.com/your-repo/ai_chatbot_workshop.git
cd ai_chatbot_workshop

# Python 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2️⃣ 환경변수 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일을 열어 API 키 입력
# OPENAI_API_KEY=sk-your-openai-api-key-here
```

### 3️⃣ 1차시 챗봇 실행

#### 웹 인터페이스 (권장)
```bash
streamlit run lesson1_basic_chatbot.py
```
브라우저에서 `http://localhost:8501` 접속

#### CLI 인터페이스
```bash
python lesson1_basic_chatbot.py cli
```

### 4️⃣ Jupyter 노트북 실습
```bash
jupyter lab lesson1_basic_chatbot.ipynb
```

---

## 📁 프로젝트 구조

```
ai_chatbot_workshop/
├── 📄 README.md                    # 이 파일
├── 📄 requirements.txt              # Python 의존성
├── 📄 .env.example                  # 환경변수 템플릿
├── 📄 config.py                     # 설정 관리 모듈
├── 📄 LICENSE                       # MIT 라이선스
│
├── 📂 doc/                          # 문서 및 요구사항
│   ├── REQUIREMENTS.md
│   └── REQUIREMENTS2.md
│
├── 📂 lesson1_기본_챗봇/
│   ├── 🐍 lesson1_basic_chatbot.py      # 통합 실행 파일
│   └── 📓 lesson1_basic_chatbot.ipynb   # 단계별 학습 노트북
│
├── 📂 lesson2_프롬프트_최적화/      # (2차시에 생성)
├── 📂 lesson3_RAG_구현/             # (3차시에 생성)
├── 📂 lesson4_대화_관리/            # (4차시에 생성)
├── 📂 lesson5_외부_연동/            # (5차시에 생성)
├── 📂 lesson6_성능_최적화/          # (6차시에 생성)
├── 📂 lesson7_배포_통합/            # (7차시에 생성)
│
├── 📂 data/                         # 데이터 저장소
│   ├── chroma_db/                   # 벡터 DB (3차시)
│   └── uploads/                     # 업로드 파일
│
└── 📂 logs/                         # 로그 파일
    └── app.log
```

---

## 🛠️ 주요 기술 스택

### 🤖 AI/LLM
- **OpenAI GPT API** - 주요 LLM 서비스
- **Anthropic Claude** - 대체 LLM (선택사항)
- **LangChain** - LLM 애플리케이션 프레임워크
- **LlamaIndex** - RAG 구현 도구

### 🌐 웹 프레임워크
- **Streamlit** - 빠른 프로토타입 웹 UI
- **FastAPI** - 프로덕션 REST API
- **Uvicorn** - ASGI 웹 서버

### 🗄️ 데이터베이스
- **Redis** - 세션 관리, 캐싱
- **ChromaDB** - 벡터 데이터베이스
- **FAISS** - 고성능 벡터 검색

### 🔧 개발 도구
- **Pydantic** - 데이터 검증
- **Pytest** - 테스트 프레임워크
- **Black** - 코드 포매터
- **Ruff** - 고속 린터

---

## 💡 1차시 상세 가이드

### 🎯 학습 목표
- [x] OpenAI API 연동 및 클라이언트 설정
- [x] 스트리밍 vs 일반 응답 구현 비교
- [x] 실무용 토큰 사용량 모니터링
- [x] API 키 로테이션 및 장애 처리
- [x] 구조화된 로깅 시스템 구축
- [x] 웹 인터페이스와 CLI 모드 지원

### 🏗️ 구현된 주요 클래스

#### `BasicChatbot`
```python
# 기본 사용법
chatbot = BasicChatbot(token_monitor, key_rotator)

# 일반 응답
response = chatbot.generate_response(messages, user_id="user123")

# 스트리밍 응답
for chunk in chatbot.generate_streaming_response(messages, user_id="user123"):
    if chunk["type"] == "content":
        print(chunk["content"], end="")
```

#### `TokenMonitor`
```python
# 사용량 추적
monitor = TokenMonitor()
monitor.track_usage("user123", "gpt-4o-mini", input_tokens=100, output_tokens=50, processing_time=1.2)

# 통계 조회
stats = monitor.get_user_stats("user123")
print(f"오늘 사용량: {stats['tokens_used']} 토큰")
print(f"예상 비용: ${stats['cost']:.4f}")
```

#### `APIKeyRotator`
```python
# 키 로테이션 설정
rotator = APIKeyRotator(primary_key="sk-main", backup_key="sk-backup")

# 에러 발생 시 자동 로테이션
try:
    # API 호출
    pass
except Exception as e:
    if rotator.handle_api_error(e):
        print("백업 키로 전환됨")
```

### 🔍 성능 비교 (스트리밍 vs 일반)

| 모드 | 첫 응답 시간 | 전체 완료 시간 | 사용자 체감 |
|------|-------------|---------------|------------|
| **일반** | 3.2초 | 3.2초 | 😐 대기 시간 김 |
| **스트리밍** | 0.5초 | 3.1초 | 😊 즉시 응답 시작 |

→ **스트리밍 모드가 84% 더 빠른 사용자 경험 제공**

---

## 📊 실행 결과 예시

### 웹 인터페이스
```
🤖 AI 챗봇 멘토링 - 1차시: 기본 챗봇

💬 채팅
사용자: 안녕하세요! 파이썬 챗봇에 대해 알려주세요.

🤖 AI: 안녕하세요! 파이썬으로 챗봇을 만드는 것은 정말 흥미로운 주제입니다...
⏱️ 1.23초 | 🎯 156 토큰

📊 사용량 통계
- 요청 수: 5회
- 사용 토큰: 890개  
- 예상 비용: $0.0234
- 남은 한도: 49,110개
```

### CLI 모드
```bash
$ python lesson1_basic_chatbot.py cli

🤖 AI 챗봇 멘토링 1차시 CLI 데모
종료하려면 'quit' 또는 'exit'를 입력하세요.

사용자: OpenAI API의 주요 특징은?

AI (스트리밍): OpenAI API의 주요 특징들을 소개해드리겠습니다:

1. **다양한 모델 지원**: GPT-4, GPT-3.5-turbo 등...
2. **스트리밍 응답**: 실시간으로 텍스트를 생성...
3. **Function Calling**: 외부 도구와 연동 가능...

📊 처리시간: 2.15초, 추정 토큰: 187
```

---

## 🔧 환경변수 설정 가이드

### 필수 환경변수
```bash
# OpenAI API 키 (필수)
OPENAI_API_KEY=sk-your-openai-api-key-here

# 모델 설정 (선택사항)
OPENAI_MODEL=gpt-4o-mini  # 기본값
OPENAI_MAX_TOKENS=2048    # 기본값
OPENAI_TEMPERATURE=0.7    # 기본값
```

### 선택적 환경변수
```bash
# 백업 API 키 (로테이션용)
OPENAI_API_KEY_BACKUP=sk-your-backup-key-here

# 사용량 제한
MAX_TOKENS_PER_USER_DAY=50000

# Redis 설정 (고급 기능용)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your-redis-password

# 로깅 수준
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

---

## 🧪 테스트 실행

### 단위 테스트
```bash
# 전체 테스트 실행
pytest tests/

# 특정 모듈 테스트
pytest tests/test_lesson1.py -v

# 커버리지 포함
pytest tests/ --cov=. --cov-report=html
```

### 수동 테스트
```bash
# 설정 검증
python config.py

# API 연결 테스트  
python -c "from lesson1_basic_chatbot import BasicChatbot, TokenMonitor, APIKeyRotator; print('Import successful')"
```

---

## 🐛 트러블슈팅

### 자주 발생하는 문제

#### 1. API 키 에러
```
❌ OpenAI API 키가 설정되지 않았습니다!
```
**해결방법**: `.env` 파일에서 `OPENAI_API_KEY` 확인

#### 2. Redis 연결 실패
```
❌ Redis 연결 실패, 메모리 모드로 전환
```
**해결방법**: Redis 서버 실행 또는 로컬 메모리 모드로 사용

#### 3. 토큰 한도 초과
```
❌ 일일 토큰 한도 초과 (50000 토큰)
```
**해결방법**: `MAX_TOKENS_PER_USER_DAY` 환경변수 조정

#### 4. 모듈 임포트 에러
```
❌ ModuleNotFoundError: No module named 'openai'
```
**해결방법**: `pip install -r requirements.txt` 재실행

### 로그 파일 확인
```bash
# 최신 로그 확인
tail -f logs/app.log

# 에러만 필터링
grep "ERROR" logs/app.log

# API 호출 로그만 확인
grep "API" logs/app.log
```

---

## 📈 성능 최적화 팁

### 1. 스트리밍 모드 활용
- 일반 모드 대비 **84% 빠른 첫 응답**
- 사용자 체감 성능 대폭 향상

### 2. 토큰 사용량 최적화
```python
# 시스템 프롬프트 최적화
system_prompt = "You are a helpful assistant."  # 간결하게

# max_tokens 제한
max_tokens = 500  # 필요한 만큼만

# Temperature 조정
temperature = 0.7  # 일관성과 창의성 균형
```

### 3. 캐싱 전략
```python
# 자주 묻는 질문 캐싱 (2차시에서 구현)
# Redis를 활용한 응답 캐싱
```

---

## 🔮 다음 차시 미리보기

### 2차시: 프롬프트 엔지니어링
- **Jinja2** 기반 동적 프롬프트 템플릿
- **페르소나** 및 브랜드 톤앤매너 적용
- **A/B 테스트**를 통한 프롬프트 최적화
- 답변 **품질 측정** 및 개선

### 3차시: RAG 시스템 구현
- **PDF 업로드** 및 문서 파싱
- **벡터 임베딩** 생성 및 검색
- **ChromaDB**를 활용한 지식 베이스
- **출처 표시** 기능

---

## 🤝 기여하기

이 프로젝트는 오픈소스입니다. 기여를 환영합니다!

### 기여 방법
1. **Fork** 저장소
2. **Feature 브랜치** 생성 (`git checkout -b feature/amazing-feature`)
3. **Commit** 변경사항 (`git commit -m 'Add amazing feature'`)
4. **Push** to 브랜치 (`git push origin feature/amazing-feature`)
5. **Pull Request** 생성

### 버그 신고
- [Issues](https://github.com/your-repo/ai_chatbot_workshop/issues)에서 버그를 신고해주세요
- 재현 가능한 예시와 함께 신고해주시면 도움이 됩니다

---

## 📜 라이선스

이 프로젝트는 [MIT 라이선스](LICENSE) 하에 배포됩니다.

---

## 👨‍💻 작성자

**AI Chatbot Workshop Team**
- 📧 Email: workshop@example.com
- 🌐 Website: [workshop-site.com](https://workshop-site.com)
- 💬 Discord: [워크샵 커뮤니티](https://discord.gg/workshop)

---

## ⭐ 마무리

이 워크샵을 통해 **실무에서 바로 사용 가능한** AI 챗봇 시스템을 구축하는 능력을 기를 수 있습니다. 

**핵심은 단순히 API를 호출하는 것이 아니라**, 안정적이고 확장 가능한 시스템을 설계하는 것입니다.

### 🎯 워크샵 완주 시 달성할 수 있는 것들:
- ✅ **프로덕션 레벨** AI 챗봇 시스템 구축
- ✅ **대화형 AI** 서비스 설계 및 운영
- ✅ **RAG 기반** 지식 검색 시스템
- ✅ **성능 최적화** 및 비용 관리
- ✅ **클라우드 배포** 및 운영

**지금 시작해보세요!** 🚀

```bash
git clone https://github.com/your-repo/ai_chatbot_workshop.git
cd ai_chatbot_workshop
pip install -r requirements.txt
streamlit run lesson1_basic_chatbot.py
```

---

*마지막 업데이트: 2024년 8월 30일*
