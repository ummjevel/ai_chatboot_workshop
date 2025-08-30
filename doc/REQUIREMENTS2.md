## 📅 7시간 멘토링 커리큘럼 (AI 챗봇)

### ⏱️ 1시간차 – AI 챗봇 기본 이해

- **목표**: LLM 기반 챗봇이 기존 rule-based 챗봇과 어떻게 다른지 이해
- **내용**
    - 챗봇 역사: Rule-based → Retrieval → Generative LLM
    - LLM 챗봇 구조: 입력 → 프롬프트 → LLM → 응답
    - 스트리밍 응답 데모 (OpenAI API / Hugging Face 모델)
- **실습**: OpenAI API로 "Hello chatbot" 호출 → 스트리밍 출력 확인

---

### ⏱️ 2시간차 – LLM과 프롬프트 엔지니어링

- **목표**: 프롬프트의 힘과 한계 이해
- **내용**
    - System/User/Assistant 역할
    - Few-shot, Chain-of-thought, Instruction 튜닝
    - 프롬프트 관리 모듈화 (LangChain PromptTemplate, LlamaIndex Prompt)
- **실습**: 동일 질문에 프롬프트를 바꿔가며 답변 차이 보기

---

### ⏱️ 3시간차 – 문서 기반 챗봇 (RAG ① 기본)

- **목표**: PDF/DB/웹 데이터와 연결되는 챗봇 이해
- **내용**
    - RAG 개념 (Retriever + LLM + Generation)
    - Embedding, VectorDB(Faiss/Chroma) 소개
    - LangChain / LlamaIndex 개요
- **실습**: PDF 업로드 → 간단한 QA 챗봇 만들기 (LlamaIndex 버전)

---

### ⏱️ 4시간차 – 문서 기반 챗봇 (RAG ② 확장)

- **목표**: 실제 서비스처럼 검색 품질 개선 및 비교
- **내용**
    - Chunking 전략 (문단/슬라이딩 윈도우)
    - Top-k retrieval 튜닝
    - LangChain RetrievalQA vs ConversationalRetrievalChain
- **실습**: 같은 문서에 대해 LangChain과 LlamaIndex 비교

---

### ⏱️ 5시간차 – 에이전트와 툴콜링

- **목표**: 챗봇이 검색/DB/API를 능동적으로 호출하게 만들기
- **내용**
    - Tool 정의 (검색 API, 계산기, DB 질의)
    - Agent 구조 (ReAct, MRKL, LangChain AgentExecutor)
    - Tool Calling vs Agent 차이
- **실습**: 챗봇이 "서울 날씨 알려줘" → 외부 API 툴 호출

---

### ⏱️ 6시간차 – 아키텍처 & 운영 고려

- **목표**: 실제 서비스로 배포할 때 고려해야 할 점 이해
- **내용**
    - 모듈화: Prompt templates, Retriever, LLM wrapper, Output parser
    - 에러 핸들링: rate limit, timeout, fallback
    - 비용 최적화: context trimming, 캐싱 전략
    - 스트리밍 응답 UX
- **실습**: 토큰 길이 줄이기, 캐싱으로 같은 질문 빠르게 응답

---

### ⏱️ 7시간차 – 실무 트러블슈팅 & 종합 프로젝트

- **목표**: 실무 문제 해결 능력 강화
- **내용**
    - 토큰 폭주 문제 → 청킹 전략
    - 답변 일관성 부족 → system prompt 관리
    - 지식 누락 → RAG + 임베딩 품질 개선
    - 멀티모달 확장 (이미지 질문, 음성 TTS 챗봇)
- **실습 프로젝트**:
    - 자신만의 PDF 기반 Q&A 챗봇 완성
    - 추가로 Tool(검색 API) 연결 → “지식 + 실시간 정보” 챗봇
