#!/usr/bin/env python3
"""
7차시: 배포 및 통합
Author: AI Chatbot Workshop
Date: 2024-08-30
Description: FastAPI REST API, Docker 배포, CI/CD 파이프라인, 프로덕션 환경 구성
"""

import os
import sys
import json
import time
import uuid
import logging
import asyncio
import hashlib
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import subprocess

# FastAPI 및 웹 서버
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# 모니터링 및 로깅
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import structlog
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# 외부 라이브러리
import redis
import psutil
from openai import OpenAI
import docker

# 로컬 모듈 (이전 차시에서 구현한 클래스들)
sys.path.append('..')
from config import get_config
from lesson6.lesson6_performance_monitoring import (
    CacheManager, AsyncChatbot, PerformanceMonitor, 
    TokenOptimizer, PerformanceAnalyzer
)

# 설정
config = get_config()
logger = structlog.get_logger(__name__)

# Prometheus 메트릭
API_REQUESTS_TOTAL = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
API_REQUEST_DURATION = Histogram('api_request_duration_seconds', 'API request duration')
ACTIVE_CONNECTIONS = Gauge('api_active_connections', 'Active API connections')
DEPLOYMENT_INFO = Gauge('deployment_info', 'Deployment information', ['version', 'environment'])

# 보안
security = HTTPBearer(auto_error=False)

@dataclass
class ChatRequest:
    """채팅 요청 모델"""
    message: str = Field(..., description="사용자 메시지")
    conversation_id: Optional[str] = Field(None, description="대화 ID")
    model: str = Field("gpt-3.5-turbo", description="사용할 모델")
    temperature: float = Field(0.7, description="응답 창의성")
    max_tokens: Optional[int] = Field(None, description="최대 토큰 수")
    stream: bool = Field(False, description="스트리밍 응답 여부")
    use_cache: bool = Field(True, description="캐시 사용 여부")

@dataclass 
class ChatResponse:
    """채팅 응답 모델"""
    response: str
    conversation_id: str
    model: str
    usage: Dict[str, int]
    response_time: float
    cached: bool
    timestamp: str

@dataclass
class HealthResponse:
    """헬스 체크 응답"""
    status: str
    timestamp: str
    version: str
    uptime: float
    services: Dict[str, str]

class ProductionChatbotAPI:
    """프로덕션용 챗봇 API 서버"""
    
    def __init__(self):
        """API 서버 초기화"""
        self.app = FastAPI(
            title="AI Chatbot API",
            description="Production-ready AI Chatbot API",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # 시작 시간
        self.start_time = time.time()
        
        # 핵심 컴포넌트 초기화
        self._init_components()
        
        # 미들웨어 설정
        self._setup_middleware()
        
        # 라우터 설정
        self._setup_routes()
        
        # 트레이싱 설정
        self._setup_tracing()
        
        # 백그라운드 작업
        self._setup_background_tasks()
    
    def _init_components(self):
        """핵심 컴포넌트 초기화"""
        try:
            # 캐시 매니저
            self.cache_manager = CacheManager(
                redis_host=config.get('redis_host', 'localhost'),
                redis_port=config.get('redis_port', 6379)
            )
            
            # 비동기 챗봇
            self.chatbot = AsyncChatbot(self.cache_manager)
            
            # 성능 모니터
            self.performance_monitor = PerformanceMonitor()
            self.performance_monitor.start_monitoring(interval=30)
            
            # 토큰 최적화
            self.token_optimizer = TokenOptimizer()
            
            # 성능 분석기
            self.analyzer = PerformanceAnalyzer(self.performance_monitor)
            
            logger.info("모든 컴포넌트 초기화 완료")
            
        except Exception as e:
            logger.error(f"컴포넌트 초기화 실패: {e}")
            raise
    
    def _setup_middleware(self):
        """미들웨어 설정"""
        # CORS 설정
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=config.get('allowed_origins', ['http://localhost:3000']),
            allow_credentials=True,
            allow_methods=['GET', 'POST'],
            allow_headers=['*'],
        )
        
        # 신뢰할 수 있는 호스트
        trusted_hosts = config.get('trusted_hosts', ['localhost', '127.0.0.1'])
        self.app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=trusted_hosts
        )
        
        # 커스텀 미들웨어
        @self.app.middleware("http")
        async def monitoring_middleware(request: Request, call_next):
            """요청 모니터링 미들웨어"""
            start_time = time.time()
            
            # 활성 연결 수 증가
            ACTIVE_CONNECTIONS.inc()
            
            try:
                response = await call_next(request)
                
                # 메트릭 기록
                duration = time.time() - start_time
                API_REQUEST_DURATION.observe(duration)
                API_REQUESTS_TOTAL.labels(
                    method=request.method,
                    endpoint=request.url.path,
                    status=response.status_code
                ).inc()
                
                # 응답 헤더에 추가 정보
                response.headers["X-Response-Time"] = str(duration)
                response.headers["X-Request-ID"] = str(uuid.uuid4())
                
                return response
                
            except Exception as e:
                # 에러 메트릭
                API_REQUESTS_TOTAL.labels(
                    method=request.method,
                    endpoint=request.url.path,
                    status=500
                ).inc()
                
                logger.error(f"요청 처리 중 오류: {e}")
                raise
            
            finally:
                # 활성 연결 수 감소
                ACTIVE_CONNECTIONS.dec()
    
    def _setup_routes(self):
        """API 라우터 설정"""
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """헬스 체크"""
            uptime = time.time() - self.start_time
            
            # 서비스 상태 확인
            services = {
                "redis": "healthy" if self.cache_manager.redis_available else "unhealthy",
                "openai": "healthy",  # API 키 검증 로직 추가 가능
                "database": "healthy"  # 데이터베이스 연결 확인
            }
            
            status = "healthy" if all(s == "healthy" for s in services.values()) else "degraded"
            
            return HealthResponse(
                status=status,
                timestamp=datetime.now().isoformat(),
                version="1.0.0",
                uptime=uptime,
                services=services
            )
        
        @self.app.get("/metrics")
        async def metrics():
            """Prometheus 메트릭 엔드포인트"""
            return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
        
        @self.app.post("/chat", response_model=ChatResponse)
        async def chat_completion(
            request: ChatRequest,
            background_tasks: BackgroundTasks,
            auth: Optional[HTTPAuthorizationCredentials] = Depends(security)
        ):
            """채팅 완성 API"""
            start_time = time.time()
            
            try:
                # 인증 확인 (옵션)
                if config.get('require_auth', False):
                    if not auth or not self._verify_token(auth.credentials):
                        raise HTTPException(status_code=401, detail="인증 실패")
                
                # 대화 ID 생성
                conversation_id = request.conversation_id or str(uuid.uuid4())
                
                # 메시지 구성
                messages = [{"role": "user", "content": request.message}]
                
                # 토큰 최적화
                optimized_messages = self.token_optimizer.optimize_messages(
                    messages, model=request.model
                )
                
                # 채팅 완성 요청
                result = await self.chatbot.chat_completion(
                    messages=optimized_messages,
                    model=request.model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    use_cache=request.use_cache
                )
                
                # 응답 구성
                response = ChatResponse(
                    response=result['response'],
                    conversation_id=conversation_id,
                    model=result['model'],
                    usage=result['usage'],
                    response_time=time.time() - start_time,
                    cached=False,  # 실제 캐시 여부 확인
                    timestamp=datetime.now().isoformat()
                )
                
                # 백그라운드 작업: 메트릭 기록
                background_tasks.add_task(
                    self._record_chat_metrics,
                    result, time.time() - start_time
                )
                
                return response
                
            except Exception as e:
                logger.error(f"채팅 완성 오류: {e}")
                raise HTTPException(status_code=500, detail=f"내부 서버 오류: {str(e)}")
        
        @self.app.post("/chat/stream")
        async def chat_stream(
            request: ChatRequest,
            auth: Optional[HTTPAuthorizationCredentials] = Depends(security)
        ):
            """스트리밍 채팅 API"""
            try:
                # 인증 확인
                if config.get('require_auth', False):
                    if not auth or not self._verify_token(auth.credentials):
                        raise HTTPException(status_code=401, detail="인증 실패")
                
                # 스트리밍 응답 생성
                async def generate():
                    # 실제 스트리밍 구현은 OpenAI API의 stream 파라미터 활용
                    # 여기서는 시뮬레이션
                    response_parts = [
                        "안녕하세요! ",
                        "무엇을 도와드릴까요? ",
                        "궁금한 것이 있으시면 ",
                        "언제든지 말씀해 주세요."
                    ]
                    
                    for part in response_parts:
                        yield f"data: {json.dumps({'text': part, 'done': False})}\\n\\n"
                        await asyncio.sleep(0.1)  # 스트리밍 시뮬레이션
                    
                    yield f"data: {json.dumps({'text': '', 'done': True})}\\n\\n"
                
                return StreamingResponse(
                    generate(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache"}
                )
                
            except Exception as e:
                logger.error(f"스트리밍 오류: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/analytics/performance")
        async def get_performance_analytics(hours: int = 24):
            """성능 분석 데이터"""
            try:
                response_analysis = self.analyzer.analyze_response_times(hours)
                system_analysis = self.analyzer.analyze_system_performance(hours)
                
                return {
                    "response_metrics": response_analysis,
                    "system_metrics": system_analysis,
                    "cache_metrics": asdict(self.cache_manager.get_metrics())
                }
                
            except Exception as e:
                logger.error(f"분석 데이터 조회 오류: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/analytics/report")
        async def get_performance_report():
            """성능 리포트"""
            try:
                report = self.analyzer.generate_performance_report()
                return {"report": report}
                
            except Exception as e:
                logger.error(f"리포트 생성 오류: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/admin/cache/clear")
        async def clear_cache(
            auth: HTTPAuthorizationCredentials = Depends(security)
        ):
            """캐시 클리어 (관리자용)"""
            try:
                # 관리자 권한 확인
                if not self._verify_admin_token(auth.credentials):
                    raise HTTPException(status_code=403, detail="관리자 권한 필요")
                
                # 캐시 클리어
                self.cache_manager.local_cache.clear()
                self.cache_manager.local_cache_order.clear()
                
                if self.cache_manager.redis_available:
                    self.cache_manager.redis_client.flushall()
                
                return {"message": "캐시가 클리어되었습니다"}
                
            except Exception as e:
                logger.error(f"캐시 클리어 오류: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def _setup_tracing(self):
        """분산 트레이싱 설정"""
        try:
            # Jaeger 설정
            jaeger_exporter = JaegerExporter(
                agent_host_name=config.get('jaeger_host', 'localhost'),
                agent_port=config.get('jaeger_port', 6831)
            )
            
            # TracerProvider 설정
            trace.set_tracer_provider(TracerProvider())
            tracer = trace.get_tracer(__name__)
            
            # Span processor
            span_processor = BatchSpanProcessor(jaeger_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
            
            # FastAPI instrumentation
            FastAPIInstrumentor.instrument_app(self.app, tracer_provider=trace.get_tracer_provider())
            
            logger.info("분산 트레이싱 설정 완료")
            
        except Exception as e:
            logger.warning(f"트레이싱 설정 실패: {e}")
    
    def _setup_background_tasks(self):
        """백그라운드 작업 설정"""
        @self.app.on_event("startup")
        async def startup_event():
            """앱 시작 시 실행"""
            logger.info("API 서버 시작")
            
            # 배포 정보 메트릭
            DEPLOYMENT_INFO.labels(
                version="1.0.0",
                environment=config.get('environment', 'development')
            ).set(1)
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """앱 종료 시 실행"""
            logger.info("API 서버 종료")
            
            # 모니터링 중지
            self.performance_monitor.stop_monitoring()
    
    def _verify_token(self, token: str) -> bool:
        """토큰 검증"""
        # 실제 구현에서는 JWT 토큰 검증 로직
        return token == config.get('api_token', 'default_token')
    
    def _verify_admin_token(self, token: str) -> bool:
        """관리자 토큰 검증"""
        return token == config.get('admin_token', 'admin_token')
    
    async def _record_chat_metrics(self, result: Dict, response_time: float):
        """채팅 메트릭 기록 (백그라운드 작업)"""
        try:
            # 성능 모니터에 기록
            self.performance_monitor.record_response_time(
                response_time=response_time,
                tokens_used=result['usage']['total_tokens'],
                model=result['model'],
                cached=False
            )
            
            # 토큰 사용량 추적
            self.token_optimizer.track_usage(result['usage'], result['model'])
            
        except Exception as e:
            logger.error(f"메트릭 기록 오류: {e}")
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """API 서버 실행"""
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_config={
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": {
                    "default": {
                        "()": "uvicorn.logging.DefaultFormatter",
                        "fmt": "%(levelprefix)s %(asctime)s %(message)s",
                        "datefmt": "%Y-%m-%d %H:%M:%S",
                    },
                },
                "handlers": {
                    "default": {
                        "formatter": "default",
                        "class": "logging.StreamHandler",
                        "stream": "ext://sys.stdout",
                    },
                },
                "root": {
                    "level": "INFO",
                    "handlers": ["default"],
                },
            },
            **kwargs
        )

class DockerManager:
    """Docker 배포 관리자"""
    
    def __init__(self):
        """Docker 매니저 초기화"""
        self.client = docker.from_env()
        self.logger = structlog.get_logger(__name__)
    
    def build_image(self, dockerfile_path: str = ".", image_name: str = "chatbot-api", tag: str = "latest"):
        """Docker 이미지 빌드"""
        try:
            self.logger.info(f"Docker 이미지 빌드 시작: {image_name}:{tag}")
            
            image, logs = self.client.images.build(
                path=dockerfile_path,
                tag=f"{image_name}:{tag}",
                rm=True,
                forcerm=True
            )
            
            # 빌드 로그 출력
            for log in logs:
                if 'stream' in log:
                    print(log['stream'].strip())
            
            self.logger.info(f"Docker 이미지 빌드 완료: {image.id}")
            return image
            
        except Exception as e:
            self.logger.error(f"Docker 이미지 빌드 실패: {e}")
            raise
    
    def run_container(self, image_name: str, container_name: str = "chatbot-api",
                      ports: Dict[int, int] = None, environment: Dict[str, str] = None,
                      volumes: Dict[str, Dict[str, str]] = None):
        """Docker 컨테이너 실행"""
        try:
            # 기존 컨테이너 제거
            try:
                old_container = self.client.containers.get(container_name)
                old_container.stop()
                old_container.remove()
                self.logger.info(f"기존 컨테이너 제거: {container_name}")
            except docker.errors.NotFound:
                pass
            
            # 새 컨테이너 실행
            container = self.client.containers.run(
                image=image_name,
                name=container_name,
                ports=ports or {8000: 8000},
                environment=environment or {},
                volumes=volumes or {},
                detach=True,
                restart_policy={"Name": "unless-stopped"}
            )
            
            self.logger.info(f"컨테이너 실행 완료: {container.id}")
            return container
            
        except Exception as e:
            self.logger.error(f"컨테이너 실행 실패: {e}")
            raise
    
    def generate_dockerfile(self, output_path: str = "Dockerfile"):
        """Dockerfile 생성"""
        dockerfile_content = '''# 멀티스테이지 빌드
FROM python:3.9-slim as builder

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# 프로덕션 스테이지
FROM python:3.9-slim

# 시스템 사용자 생성
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 복사
COPY --from=builder /root/.local /home/appuser/.local

# 애플리케이션 디렉토리 생성
WORKDIR /app

# 소스 코드 복사
COPY --chown=appuser:appuser . .

# 권한 설정
RUN chmod +x /app/scripts/*.sh || true

# 포트 노출
EXPOSE 8000

# 헬스 체크
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# 사용자 변경
USER appuser

# 환경 변수
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/app

# 애플리케이션 실행
CMD ["python", "-m", "uvicorn", "lesson7.lesson7_deployment_integration:create_app", "--host", "0.0.0.0", "--port", "8000"]
'''
        
        with open(output_path, 'w') as f:
            f.write(dockerfile_content)
        
        self.logger.info(f"Dockerfile 생성: {output_path}")
    
    def generate_docker_compose(self, output_path: str = "docker-compose.yml"):
        """docker-compose.yml 생성"""
        compose_content = '''version: '3.8'

services:
  chatbot-api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: chatbot-api
    ports:
      - "8000:8000"
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - ENVIRONMENT=production
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    depends_on:
      - redis
      - prometheus
    restart: unless-stopped
    networks:
      - chatbot-network

  redis:
    image: redis:7-alpine
    container_name: chatbot-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    networks:
      - chatbot-network

  prometheus:
    image: prom/prometheus:latest
    container_name: chatbot-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped
    networks:
      - chatbot-network

  grafana:
    image: grafana/grafana:latest
    container_name: chatbot-grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    restart: unless-stopped
    networks:
      - chatbot-network

  jaeger:
    image: jaegertracing/all-in-one:latest
    container_name: chatbot-jaeger
    ports:
      - "16686:16686"
      - "6831:6831/udp"
    environment:
      - COLLECTOR_OTLP_ENABLED=true
    restart: unless-stopped
    networks:
      - chatbot-network

volumes:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  chatbot-network:
    driver: bridge
'''
        
        with open(output_path, 'w') as f:
            f.write(compose_content)
        
        self.logger.info(f"docker-compose.yml 생성: {output_path}")

class CICDManager:
    """CI/CD 파이프라인 관리자"""
    
    def __init__(self, project_path: str = "."):
        """CI/CD 매니저 초기화"""
        self.project_path = Path(project_path)
        self.logger = structlog.get_logger(__name__)
    
    def generate_github_actions(self):
        """GitHub Actions 워크플로우 생성"""
        workflow_dir = self.project_path / ".github" / "workflows"
        workflow_dir.mkdir(parents=True, exist_ok=True)
        
        # CI 워크플로우
        ci_workflow = '''name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Lint with flake8
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

    - name: Test with pytest
      run: |
        pytest tests/ -v --cov=./src --cov-report=xml

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Run security checks
      run: |
        pip install bandit safety
        bandit -r ./src
        safety check --requirements requirements.txt

  docker:
    runs-on: ubuntu-latest
    needs: [test]
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Login to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
    
    - name: Build and push
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ secrets.DOCKER_USERNAME }}/chatbot-api:latest
        cache-from: type=gha
        cache-to: type=gha,mode=max
'''
        
        with open(workflow_dir / "ci.yml", 'w') as f:
            f.write(ci_workflow)
        
        # CD 워크플로우
        cd_workflow = '''name: CD

on:
  push:
    branches: [ main ]
  release:
    types: [ published ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to production
      uses: appleboy/ssh-action@v0.1.7
      with:
        host: ${{ secrets.HOST }}
        username: ${{ secrets.USERNAME }}
        key: ${{ secrets.SSH_KEY }}
        script: |
          cd /opt/chatbot-api
          git pull origin main
          docker-compose down
          docker-compose build --no-cache
          docker-compose up -d
          
    - name: Health check
      run: |
        sleep 30
        curl -f ${{ secrets.API_URL }}/health || exit 1
        
    - name: Notify Slack
      uses: 8398a7/action-slack@v3
      with:
        status: ${{ job.status }}
        channel: '#deployments'
        webhook_url: ${{ secrets.SLACK_WEBHOOK }}
      if: always()
'''
        
        with open(workflow_dir / "cd.yml", 'w') as f:
            f.write(cd_workflow)
        
        self.logger.info("GitHub Actions 워크플로우 생성 완료")
    
    def generate_requirements_files(self):
        """requirements 파일들 생성"""
        # 기본 requirements.txt
        requirements = '''fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
openai==1.3.7
redis==5.0.1
prometheus-client==0.19.0
structlog==23.2.0
psutil==5.9.6
docker==6.1.3
opentelemetry-api==1.21.0
opentelemetry-sdk==1.21.0
opentelemetry-exporter-jaeger==1.21.0
opentelemetry-instrumentation-fastapi==0.42b0
'''
        
        with open(self.project_path / "requirements.txt", 'w') as f:
            f.write(requirements)
        
        # 개발용 requirements-dev.txt
        requirements_dev = '''pytest==7.4.3
pytest-cov==4.1.0
pytest-asyncio==0.21.1
black==23.11.0
flake8==6.1.0
bandit==1.7.5
safety==2.3.5
pre-commit==3.5.0
'''
        
        with open(self.project_path / "requirements-dev.txt", 'w') as f:
            f.write(requirements_dev)
        
        self.logger.info("Requirements 파일 생성 완료")
    
    def generate_pre_commit_config(self):
        """pre-commit 설정 생성"""
        pre_commit_config = '''repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
        args: [--max-line-length=127]

  - repo: https://github.com/pycqa/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: ['-c', 'pyproject.toml']
        additional_dependencies: ['bandit[toml]']
'''
        
        with open(self.project_path / ".pre-commit-config.yaml", 'w') as f:
            f.write(pre_commit_config)
        
        self.logger.info("pre-commit 설정 생성 완료")

class MonitoringSetup:
    """모니터링 시스템 설정"""
    
    def __init__(self, output_dir: str = "monitoring"):
        """모니터링 설정 초기화"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = structlog.get_logger(__name__)
    
    def generate_prometheus_config(self):
        """Prometheus 설정 파일 생성"""
        prometheus_config = '''global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

rule_files:
  - "alerts.yml"

scrape_configs:
  - job_name: 'chatbot-api'
    static_configs:
      - targets: ['chatbot-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
'''
        
        with open(self.output_dir / "prometheus.yml", 'w') as f:
            f.write(prometheus_config)
        
        # 알림 규칙
        alerts_config = '''groups:
  - name: chatbot-alerts
    rules:
      - alert: HighResponseTime
        expr: api_request_duration_seconds{quantile="0.95"} > 5
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High API response time"
          description: "95th percentile response time is {{ $value }}s"

      - alert: HighErrorRate
        expr: rate(api_requests_total{status=~"5.."}[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High API error rate"
          description: "Error rate is {{ $value | humanizePercentage }}"

      - alert: LowCacheHitRate
        expr: chatbot_cache_hits_total / (chatbot_cache_hits_total + chatbot_cache_misses_total) < 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Low cache hit rate"
          description: "Cache hit rate is {{ $value | humanizePercentage }}"

      - alert: HighCPUUsage
        expr: 100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage"
          description: "CPU usage is {{ $value | humanizePercentage }}"
'''
        
        with open(self.output_dir / "alerts.yml", 'w') as f:
            f.write(alerts_config)
        
        self.logger.info("Prometheus 설정 생성 완료")
    
    def generate_grafana_dashboards(self):
        """Grafana 대시보드 생성"""
        # Grafana provisioning 디렉토리
        grafana_dir = self.output_dir / "grafana" / "provisioning"
        datasources_dir = grafana_dir / "datasources"
        dashboards_dir = grafana_dir / "dashboards"
        
        datasources_dir.mkdir(parents=True, exist_ok=True)
        dashboards_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터소스 설정
        datasource_config = '''apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    jsonData:
      timeInterval: 15s
'''
        
        with open(datasources_dir / "prometheus.yml", 'w') as f:
            f.write(datasource_config)
        
        # 대시보드 프로바이더
        dashboard_provider = '''apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
'''
        
        with open(dashboards_dir / "dashboard-provider.yml", 'w') as f:
            f.write(dashboard_provider)
        
        # 간단한 대시보드 JSON (실제로는 Grafana UI에서 생성 후 export)
        dashboard_json = {
            "dashboard": {
                "title": "AI Chatbot API Dashboard",
                "tags": ["chatbot", "api"],
                "panels": [
                    {
                        "title": "Request Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(api_requests_total[5m])",
                                "legendFormat": "{{method}} {{endpoint}}"
                            }
                        ]
                    },
                    {
                        "title": "Response Time",
                        "type": "graph", 
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.95, api_request_duration_seconds_bucket)",
                                "legendFormat": "95th percentile"
                            }
                        ]
                    }
                ]
            }
        }
        
        with open(dashboards_dir / "api-dashboard.json", 'w') as f:
            json.dump(dashboard_json, f, indent=2)
        
        self.logger.info("Grafana 설정 생성 완료")

def create_app() -> FastAPI:
    """애플리케이션 팩토리"""
    api = ProductionChatbotAPI()
    return api.app

def main():
    """메인 함수 - 통합 배포 시스템 구축"""
    logger.info("=== 7차시: 배포 및 통합 시스템 구축 ===")
    
    try:
        # 1. Docker 설정
        logger.info("1. Docker 배포 환경 구성")
        docker_manager = DockerManager()
        docker_manager.generate_dockerfile()
        docker_manager.generate_docker_compose()
        
        # 2. CI/CD 설정
        logger.info("2. CI/CD 파이프라인 구성")
        cicd_manager = CICDManager()
        cicd_manager.generate_github_actions()
        cicd_manager.generate_requirements_files()
        cicd_manager.generate_pre_commit_config()
        
        # 3. 모니터링 설정
        logger.info("3. 모니터링 시스템 구성")
        monitoring_setup = MonitoringSetup()
        monitoring_setup.generate_prometheus_config()
        monitoring_setup.generate_grafana_dashboards()
        
        # 4. API 서버 테스트
        logger.info("4. API 서버 초기화 테스트")
        api = ProductionChatbotAPI()
        logger.info("API 서버 초기화 완료")
        
        # 5. 배포 스크립트 생성
        logger.info("5. 배포 스크립트 생성")
        deploy_script = '''#!/bin/bash
set -e

echo "=== 챗봇 API 배포 시작 ==="

# 환경 변수 확인
if [ -z "$OPENAI_API_KEY" ]; then
    echo "OPENAI_API_KEY 환경 변수가 설정되지 않았습니다."
    exit 1
fi

# Docker 이미지 빌드
echo "Docker 이미지 빌드 중..."
docker-compose build --no-cache

# 기존 컨테이너 중지
echo "기존 서비스 중지 중..."
docker-compose down

# 새 컨테이너 시작
echo "새 서비스 시작 중..."
docker-compose up -d

# 헬스 체크
echo "헬스 체크 중..."
sleep 30
curl -f http://localhost:8000/health || exit 1

echo "=== 배포 완료 ==="
'''
        
        with open("deploy.sh", 'w') as f:
            f.write(deploy_script)
        
        # 실행 권한 부여
        os.chmod("deploy.sh", 0o755)
        
        logger.info("배포 및 통합 시스템 구축 완료!")
        logger.info("다음 명령어로 API 서버를 실행할 수 있습니다:")
        logger.info("python -m uvicorn lesson7.lesson7_deployment_integration:create_app --host 0.0.0.0 --port 8000")
        logger.info("또는 Docker Compose 사용: docker-compose up -d")
        
    except Exception as e:
        logger.error(f"시스템 구축 실패: {e}")
        raise

if __name__ == "__main__":
    # 구조화된 로깅 설정
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # API 서버 실행 또는 배포 환경 구축
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        # 배포 환경 구축
        main()
    else:
        # API 서버 실행
        api = ProductionChatbotAPI()
        api.run(host="0.0.0.0", port=8000, reload=False)