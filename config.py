#!/usr/bin/env python3
"""
AI 챗봇 멘토링 - 설정 관리 모듈
Author: AI Chatbot Workshop
Date: 2024-08-30
Description: 환경변수 기반 설정 관리 및 검증
"""

import os
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

@dataclass
class LLMConfig:
    """LLM API 설정"""
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    max_tokens: int = 2048
    temperature: float = 0.7
    backup_api_key: Optional[str] = None
    
    # Anthropic 설정 (선택사항)
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-haiku-20240307"

@dataclass
class AppConfig:
    """애플리케이션 기본 설정"""
    env: str = "development"
    log_level: str = "INFO"
    debug: bool = True
    host: str = "localhost"
    port: int = 8000
    reload: bool = True

@dataclass
class DatabaseConfig:
    """데이터베이스 설정"""
    # Redis 설정
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # SQLite 설정
    database_url: str = "sqlite:///./chatbot.db"
    
    # PostgreSQL 설정 (프로덕션용)
    postgres_host: Optional[str] = None
    postgres_port: int = 5432
    postgres_db: Optional[str] = None
    postgres_user: Optional[str] = None
    postgres_password: Optional[str] = None

@dataclass
class ExternalAPIConfig:
    """외부 API 설정"""
    weather_api_key: Optional[str] = None
    weather_api_url: str = "https://api.openweathermap.org/data/2.5"
    serper_api_key: Optional[str] = None
    serpapi_api_key: Optional[str] = None

@dataclass
class RAGConfig:
    """RAG 시스템 설정"""
    vector_db_type: str = "chroma"
    chroma_persist_dir: str = "./data/chroma_db"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1000
    chunk_overlap: int = 200

@dataclass
class PerformanceConfig:
    """성능 및 모니터링 설정"""
    cache_ttl: int = 3600  # 1시간
    max_cache_size: int = 1000
    max_tokens_per_request: int = 4000
    max_tokens_per_user_day: int = 50000
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000

@dataclass
class SecurityConfig:
    """보안 설정"""
    secret_key: str
    access_token_expire_minutes: int = 30
    allowed_hosts: List[str] = None
    cors_origins: List[str] = None
    
    def __post_init__(self):
        if self.allowed_hosts is None:
            self.allowed_hosts = ["localhost", "127.0.0.1"]
        if self.cors_origins is None:
            self.cors_origins = ["http://localhost:3000", "http://localhost:8080"]

@dataclass
class FileConfig:
    """파일 업로드 설정"""
    upload_max_size: int = 10485760  # 10MB
    upload_allowed_extensions: List[str] = None
    upload_dir: str = "./data/uploads"
    
    def __post_init__(self):
        if self.upload_allowed_extensions is None:
            self.upload_allowed_extensions = [".pdf", ".txt", ".docx", ".md"]

class Config:
    """전체 설정 관리 클래스"""
    
    def __init__(self):
        """환경변수에서 설정값 로드 및 검증"""
        self.llm = self._load_llm_config()
        self.app = self._load_app_config()
        self.database = self._load_database_config()
        self.external_api = self._load_external_api_config()
        self.rag = self._load_rag_config()
        self.performance = self._load_performance_config()
        self.security = self._load_security_config()
        self.file = self._load_file_config()
        
        # 로깅 설정
        self._setup_logging()
        
        # 설정 검증
        self._validate_config()
    
    def _load_llm_config(self) -> LLMConfig:
        """LLM 설정 로드"""
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY는 필수 환경변수입니다")
        
        return LLMConfig(
            openai_api_key=openai_api_key,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "2048")),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
            backup_api_key=os.getenv("OPENAI_API_KEY_BACKUP"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
        )
    
    def _load_app_config(self) -> AppConfig:
        """애플리케이션 설정 로드"""
        return AppConfig(
            env=os.getenv("APP_ENV", "development"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            debug=os.getenv("DEBUG", "true").lower() == "true",
            host=os.getenv("HOST", "localhost"),
            port=int(os.getenv("PORT", "8000")),
            reload=os.getenv("RELOAD", "true").lower() == "true"
        )
    
    def _load_database_config(self) -> DatabaseConfig:
        """데이터베이스 설정 로드"""
        return DatabaseConfig(
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            redis_db=int(os.getenv("REDIS_DB", "0")),
            redis_password=os.getenv("REDIS_PASSWORD"),
            database_url=os.getenv("DATABASE_URL", "sqlite:///./chatbot.db"),
            postgres_host=os.getenv("POSTGRES_HOST"),
            postgres_port=int(os.getenv("POSTGRES_PORT", "5432")),
            postgres_db=os.getenv("POSTGRES_DB"),
            postgres_user=os.getenv("POSTGRES_USER"),
            postgres_password=os.getenv("POSTGRES_PASSWORD")
        )
    
    def _load_external_api_config(self) -> ExternalAPIConfig:
        """외부 API 설정 로드"""
        return ExternalAPIConfig(
            weather_api_key=os.getenv("WEATHER_API_KEY"),
            weather_api_url=os.getenv("WEATHER_API_URL", "https://api.openweathermap.org/data/2.5"),
            serper_api_key=os.getenv("SERPER_API_KEY"),
            serpapi_api_key=os.getenv("SERPAPI_API_KEY")
        )
    
    def _load_rag_config(self) -> RAGConfig:
        """RAG 설정 로드"""
        return RAGConfig(
            vector_db_type=os.getenv("VECTOR_DB_TYPE", "chroma"),
            chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200"))
        )
    
    def _load_performance_config(self) -> PerformanceConfig:
        """성능 설정 로드"""
        return PerformanceConfig(
            cache_ttl=int(os.getenv("CACHE_TTL", "3600")),
            max_cache_size=int(os.getenv("MAX_CACHE_SIZE", "1000")),
            max_tokens_per_request=int(os.getenv("MAX_TOKENS_PER_REQUEST", "4000")),
            max_tokens_per_user_day=int(os.getenv("MAX_TOKENS_PER_USER_DAY", "50000")),
            rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")),
            rate_limit_per_hour=int(os.getenv("RATE_LIMIT_PER_HOUR", "1000"))
        )
    
    def _load_security_config(self) -> SecurityConfig:
        """보안 설정 로드"""
        secret_key = os.getenv("SECRET_KEY")
        if not secret_key:
            # 개발 환경에서는 기본값 사용
            if self.app.env == "development":
                secret_key = "dev-secret-key-change-in-production"
            else:
                raise ValueError("SECRET_KEY는 프로덕션에서 필수 환경변수입니다")
        
        allowed_hosts = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
        cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")
        
        return SecurityConfig(
            secret_key=secret_key,
            access_token_expire_minutes=int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")),
            allowed_hosts=[host.strip() for host in allowed_hosts],
            cors_origins=[origin.strip() for origin in cors_origins]
        )
    
    def _load_file_config(self) -> FileConfig:
        """파일 업로드 설정 로드"""
        extensions = os.getenv("UPLOAD_ALLOWED_EXTENSIONS", ".pdf,.txt,.docx,.md").split(",")
        
        return FileConfig(
            upload_max_size=int(os.getenv("UPLOAD_MAX_SIZE", "10485760")),
            upload_allowed_extensions=[ext.strip() for ext in extensions],
            upload_dir=os.getenv("UPLOAD_DIR", "./data/uploads")
        )
    
    def _setup_logging(self):
        """로깅 설정"""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        logging.basicConfig(
            level=getattr(logging, self.app.log_level.upper()),
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("logs/app.log", encoding="utf-8")
            ]
        )
        
        # 외부 라이브러리 로그 레벨 조정
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    def _validate_config(self):
        """설정 검증"""
        logger = logging.getLogger(__name__)
        
        # API 키 검증
        if not self.llm.openai_api_key.startswith("sk-"):
            logger.warning("OpenAI API 키 형식이 올바르지 않을 수 있습니다")
        
        # 디렉토리 생성
        os.makedirs("logs", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        os.makedirs(self.file.upload_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.rag.chroma_persist_dir), exist_ok=True)
        
        logger.info(f"설정 로드 완료 - 환경: {self.app.env}")
    
    def get_openai_client_config(self) -> Dict[str, Any]:
        """OpenAI 클라이언트 설정 반환"""
        return {
            "api_key": self.llm.openai_api_key,
            "timeout": 30.0,
            "max_retries": 3
        }
    
    def get_redis_url(self) -> str:
        """Redis 연결 URL 생성"""
        auth = f":{self.database.redis_password}@" if self.database.redis_password else ""
        return f"redis://{auth}{self.database.redis_host}:{self.database.redis_port}/{self.database.redis_db}"
    
    def is_development(self) -> bool:
        """개발 환경 여부 확인"""
        return self.app.env == "development"
    
    def is_production(self) -> bool:
        """프로덕션 환경 여부 확인"""
        return self.app.env == "production"

# 전역 설정 인스턴스
config = Config()

# 편의 함수들
def get_config() -> Config:
    """설정 인스턴스 반환"""
    return config

def reload_config():
    """설정 재로드 (환경변수 변경 시 사용)"""
    global config
    load_dotenv(override=True)
    config = Config()

if __name__ == "__main__":
    # 설정 테스트
    import json
    
    print("=== AI 챗봇 설정 정보 ===")
    print(f"환경: {config.app.env}")
    print(f"로그 레벨: {config.app.log_level}")
    print(f"OpenAI 모델: {config.llm.openai_model}")
    print(f"API 키 (마스킹): {config.llm.openai_api_key[:10]}...")
    print(f"Redis URL: {config.get_redis_url()}")
    print(f"업로드 디렉토리: {config.file.upload_dir}")
    print(f"허용 확장자: {config.file.upload_allowed_extensions}")
    print("설정 검증 완료!")