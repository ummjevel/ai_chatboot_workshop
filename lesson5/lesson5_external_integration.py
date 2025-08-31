#!/usr/bin/env python3
"""
5차시: 외부 연동 & 실시간 데이터 처리
Author: AI Chatbot Workshop
Date: 2024-08-30
Description: OpenAI Function Calling, 외부 API 연동, 데이터베이스 연동, Tool 시스템 구현
"""

import os
import sys
import json
import time
import uuid
import logging
import asyncio
import sqlite3
import requests
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from functools import wraps
import traceback
from urllib.parse import quote

# 외부 라이브러리
import streamlit as st
from openai import OpenAI
import psycopg2
import redis
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 로컬 모듈
sys.path.append('..')
from config import get_config

# 설정 및 로깅
config = get_config()
logger = logging.getLogger(__name__)

# Tool 실행 결과 데이터 클래스
@dataclass
class ToolResult:
    """Tool 실행 결과"""
    success: bool
    data: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ToolCall:
    """Tool 호출 정보"""
    tool_name: str
    parameters: Dict[str, Any]
    call_id: str
    timestamp: datetime
    result: Optional[ToolResult] = None

# Circuit Breaker 패턴 구현
class CircuitBreakerState:
    """Circuit Breaker 상태"""
    CLOSED = "closed"       # 정상 상태
    OPEN = "open"           # 차단 상태 (에러 다발)
    HALF_OPEN = "half_open" # 반개방 상태 (복구 테스트)

class CircuitBreaker:
    """Circuit Breaker 패턴으로 외부 서비스 호출 보호"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, expected_exception: Exception = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        # 상태 관리
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        
        logger.info(f"Circuit Breaker 초기화 - 실패 임계값: {failure_threshold}, 복구 시간: {recovery_timeout}초")
    
    def call(self, func: Callable, *args, **kwargs):
        """함수 호출 with Circuit Breaker"""
        if self.state == CircuitBreakerState.OPEN:
            # 복구 시간 확인
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info("Circuit Breaker: HALF_OPEN 상태로 전환")
            else:
                raise Exception(f"Circuit Breaker OPEN - 서비스 일시적으로 사용 불가 ({self.failure_count}회 연속 실패)")
        
        try:
            # 함수 실행
            result = func(*args, **kwargs)
            self._on_success()
            return result
        
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """복구 시도 여부 확인"""
        return (
            self.last_failure_time and 
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        """성공 시 상태 관리"""
        self.failure_count = 0
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            logger.info("Circuit Breaker: CLOSED 상태로 복구")
    
    def _on_failure(self):
        """실패 시 상태 관리"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit Breaker: OPEN 상태로 전환 ({self.failure_count}회 연속 실패)")

# 재시도 데코레이터
def retry_on_failure(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """재시도 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"{func.__name__} 실패 (시도 {attempt + 1}/{max_attempts}): {str(e)}")
                    
                    if attempt < max_attempts - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff
                    
            logger.error(f"{func.__name__} 최종 실패: {str(last_exception)}")
            raise last_exception
        return wrapper
    return decorator

# Tool 베이스 클래스
class BaseTool(ABC):
    """Tool 베이스 클래스"""
    
    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.circuit_breaker = CircuitBreaker()
        
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Tool 실행 (추상 메서드)"""
        pass
    
    def get_openai_function_spec(self) -> Dict[str, Any]:
        """OpenAI Function Calling용 스펙 반환"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }
    
    def _create_result(self, success: bool, data: Any = None, error: str = None, execution_time: float = 0.0) -> ToolResult:
        """ToolResult 생성 헬퍼"""
        return ToolResult(
            success=success,
            data=data,
            error=error,
            execution_time=execution_time,
            metadata={
                'tool_name': self.name,
                'timestamp': datetime.now().isoformat()
            }
        )

# 날씨 조회 Tool
class WeatherTool(BaseTool):
    """날씨 정보 조회 도구"""
    
    def __init__(self, api_key: str = None):
        super().__init__(
            name="get_weather",
            description="특정 도시의 현재 날씨 정보를 조회합니다",
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "날씨를 조회할 도시명 (예: Seoul, Tokyo, New York)"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["metric", "imperial", "kelvin"],
                        "description": "온도 단위 (metric=섭씨, imperial=화씨, kelvin=켈빈)",
                        "default": "metric"
                    }
                },
                "required": ["city"]
            }
        )
        
        # API 키 설정 (실제 환경에서는 환경변수나 설정파일에서 가져오기)
        self.api_key = api_key or "your_openweathermap_api_key_here"
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        
        # HTTP 세션 설정 (재시도 및 타임아웃)
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        logger.info("날씨 조회 도구 초기화 완료")
    
    @retry_on_failure(max_attempts=3, delay=1.0)
    def execute(self, city: str, units: str = "metric") -> ToolResult:
        """날씨 정보 조회 실행"""
        start_time = time.time()
        logger.info(f"날씨 조회 시작: {city} ({units})")
        
        try:
            # API 호출 (Circuit Breaker 적용)
            def api_call():
                params = {
                    'q': city,
                    'appid': self.api_key,
                    'units': units,
                    'lang': 'kr'
                }
                
                response = self.session.get(self.base_url, params=params, timeout=10)
                response.raise_for_status()
                return response.json()
            
            # 실제 API 키가 없는 경우 모의 데이터 반환
            if self.api_key == "your_openweathermap_api_key_here":
                logger.warning("실제 API 키가 설정되지 않음, 모의 데이터 반환")
                weather_data = self._get_mock_weather_data(city, units)
            else:
                weather_data = self.circuit_breaker.call(api_call)
            
            # 데이터 파싱
            result_data = self._parse_weather_data(weather_data, units)
            execution_time = time.time() - start_time
            
            logger.info(f"날씨 조회 완료: {city} - {result_data['temperature']}{result_data['unit']}")
            
            return self._create_result(
                success=True,
                data=result_data,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"날씨 조회 실패: {str(e)}"
            logger.error(error_msg)
            
            return self._create_result(
                success=False,
                error=error_msg,
                execution_time=execution_time
            )
    
    def _get_mock_weather_data(self, city: str, units: str) -> Dict[str, Any]:
        """모의 날씨 데이터 생성"""
        import random
        
        mock_data = {
            "weather": [{"main": "Clear", "description": "맑음"}],
            "main": {
                "temp": random.randint(15, 30) if units == "metric" else random.randint(59, 86),
                "feels_like": random.randint(15, 30) if units == "metric" else random.randint(59, 86),
                "humidity": random.randint(40, 80),
                "pressure": random.randint(1000, 1020)
            },
            "wind": {"speed": random.randint(1, 10)},
            "name": city
        }
        return mock_data
    
    def _parse_weather_data(self, data: Dict[str, Any], units: str) -> Dict[str, Any]:
        """날씨 데이터 파싱"""
        unit_symbol = {"metric": "°C", "imperial": "°F", "kelvin": "K"}[units]
        
        return {
            "city": data.get("name", "Unknown"),
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "description": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "wind_speed": data["wind"]["speed"],
            "unit": unit_symbol,
            "raw_data": data
        }

# 검색 Tool
class SearchTool(BaseTool):
    """웹 검색 도구"""
    
    def __init__(self, api_key: str = None, search_engine: str = "google"):
        super().__init__(
            name="web_search",
            description="인터넷에서 정보를 검색합니다",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색할 키워드 또는 질문"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "반환할 검색 결과 수 (기본값: 5)",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        )
        
        self.api_key = api_key or "your_search_api_key_here"
        self.search_engine = search_engine
        
        logger.info(f"검색 도구 초기화 완료 - 엔진: {search_engine}")
    
    def execute(self, query: str, num_results: int = 5) -> ToolResult:
        """웹 검색 실행"""
        start_time = time.time()
        logger.info(f"웹 검색 시작: '{query}' (결과 수: {num_results})")
        
        try:
            # 실제 API 키가 없는 경우 모의 데이터 반환
            if self.api_key == "your_search_api_key_here":
                logger.warning("실제 검색 API 키가 설정되지 않음, 모의 데이터 반환")
                search_results = self._get_mock_search_results(query, num_results)
            else:
                search_results = self._perform_real_search(query, num_results)
            
            execution_time = time.time() - start_time
            
            logger.info(f"웹 검색 완료: {len(search_results)}개 결과 반환")
            
            return self._create_result(
                success=True,
                data={
                    "query": query,
                    "results": search_results,
                    "result_count": len(search_results)
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"웹 검색 실패: {str(e)}"
            logger.error(error_msg)
            
            return self._create_result(
                success=False,
                error=error_msg,
                execution_time=execution_time
            )
    
    def _get_mock_search_results(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """모의 검색 결과 생성"""
        mock_results = [
            {
                "title": f"{query}에 대한 검색 결과 {i+1}",
                "url": f"https://example.com/result_{i+1}",
                "snippet": f"이것은 '{query}'에 대한 {i+1}번째 검색 결과의 요약입니다. 유용한 정보를 포함하고 있습니다."
            }
            for i in range(num_results)
        ]
        return mock_results
    
    @retry_on_failure(max_attempts=2)
    def _perform_real_search(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """실제 검색 API 호출 (구현 예시)"""
        # 실제 구현에서는 Google Search API, Bing API 등 사용
        # 여기서는 DuckDuckGo instant answer API 사용 예시
        
        url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_html': '1',
            'skip_disambig': '1'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        results = []
        if data.get('Abstract'):
            results.append({
                'title': data.get('Heading', query),
                'url': data.get('AbstractURL', ''),
                'snippet': data.get('Abstract', '')
            })
        
        # Related topics에서 추가 결과 가져오기
        for topic in data.get('RelatedTopics', [])[:num_results-1]:
            if isinstance(topic, dict) and 'Text' in topic:
                results.append({
                    'title': topic.get('Text', '')[:100],
                    'url': topic.get('FirstURL', ''),
                    'snippet': topic.get('Text', '')
                })
        
        return results[:num_results]

# 데이터베이스 Tool
class DatabaseTool(BaseTool):
    """데이터베이스 조회 도구"""
    
    def __init__(self, db_type: str = "sqlite", connection_params: Dict[str, Any] = None):
        super().__init__(
            name="query_database",
            description="데이터베이스에서 정보를 조회합니다",
            parameters={
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "enum": ["user_info", "order_history", "product_info", "analytics"],
                        "description": "조회할 데이터 유형"
                    },
                    "params": {
                        "type": "object",
                        "description": "조회 파라미터 (user_id, product_id 등)",
                        "additionalProperties": True
                    }
                },
                "required": ["query_type"]
            }
        )
        
        self.db_type = db_type
        self.connection_params = connection_params or {}
        self.connection = None
        
        self._init_database()
        logger.info(f"데이터베이스 도구 초기화 완료 - 타입: {db_type}")
    
    def _init_database(self):
        """데이터베이스 연결 초기화"""
        try:
            if self.db_type == "sqlite":
                # SQLite 연결 및 테스트 테이블 생성
                db_path = self.connection_params.get("db_path", ":memory:")
                self.connection = sqlite3.connect(db_path, check_same_thread=False)
                self.connection.row_factory = sqlite3.Row
                self._create_test_tables()
                
            elif self.db_type == "postgresql":
                # PostgreSQL 연결 (psycopg2 사용)
                self.connection = psycopg2.connect(**self.connection_params)
                
            logger.info("데이터베이스 연결 성공")
            
        except Exception as e:
            logger.error(f"데이터베이스 연결 실패: {e}")
            # 실패 시 메모리 SQLite로 폴백
            self.connection = sqlite3.connect(":memory:", check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
            self._create_test_tables()
    
    def _create_test_tables(self):
        """테스트용 테이블 및 데이터 생성"""
        cursor = self.connection.cursor()
        
        # 사용자 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                email TEXT,
                created_at TEXT,
                last_login TEXT
            )
        ''')
        
        # 상품 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                product_id INTEGER PRIMARY KEY,
                name TEXT,
                category TEXT,
                price REAL,
                stock INTEGER
            )
        ''')
        
        # 주문 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                order_id INTEGER PRIMARY KEY,
                user_id INTEGER,
                product_id INTEGER,
                quantity INTEGER,
                order_date TEXT,
                status TEXT,
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                FOREIGN KEY (product_id) REFERENCES products(product_id)
            )
        ''')
        
        # 테스트 데이터 삽입
        test_users = [
            (1, 'john_doe', 'john@example.com', '2024-01-15', '2024-08-30'),
            (2, 'jane_smith', 'jane@example.com', '2024-02-20', '2024-08-29'),
            (3, 'bob_wilson', 'bob@example.com', '2024-03-10', '2024-08-28')
        ]
        
        test_products = [
            (1, 'Python 프로그래밍 책', '도서', 29.99, 50),
            (2, '무선 마우스', '전자제품', 45.00, 30),
            (3, '커피 머그컵', '생활용품', 12.50, 100)
        ]
        
        test_orders = [
            (1, 1, 1, 2, '2024-08-25', 'completed'),
            (2, 1, 3, 1, '2024-08-26', 'completed'),
            (3, 2, 2, 1, '2024-08-27', 'pending'),
            (4, 3, 1, 1, '2024-08-28', 'completed')
        ]
        
        cursor.executemany('INSERT OR REPLACE INTO users VALUES (?, ?, ?, ?, ?)', test_users)
        cursor.executemany('INSERT OR REPLACE INTO products VALUES (?, ?, ?, ?, ?)', test_products)
        cursor.executemany('INSERT OR REPLACE INTO orders VALUES (?, ?, ?, ?, ?, ?)', test_orders)
        
        self.connection.commit()
        logger.info("테스트 데이터 생성 완료")
    
    def execute(self, query_type: str, params: Dict[str, Any] = None) -> ToolResult:
        """데이터베이스 쿼리 실행"""
        start_time = time.time()
        params = params or {}
        
        logger.info(f"DB 쿼리 실행: {query_type}, 파라미터: {params}")
        
        try:
            cursor = self.connection.cursor()
            
            # 쿼리 타입별 처리
            if query_type == "user_info":
                result = self._query_user_info(cursor, params)
            elif query_type == "order_history":
                result = self._query_order_history(cursor, params)
            elif query_type == "product_info":
                result = self._query_product_info(cursor, params)
            elif query_type == "analytics":
                result = self._query_analytics(cursor, params)
            else:
                raise ValueError(f"지원하지 않는 쿼리 타입: {query_type}")
            
            execution_time = time.time() - start_time
            
            logger.info(f"DB 쿼리 완료: {len(result) if isinstance(result, list) else 1}건 조회")
            
            return self._create_result(
                success=True,
                data={
                    "query_type": query_type,
                    "result": result,
                    "record_count": len(result) if isinstance(result, list) else 1
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"DB 쿼리 실패: {str(e)}"
            logger.error(error_msg)
            
            return self._create_result(
                success=False,
                error=error_msg,
                execution_time=execution_time
            )
    
    def _query_user_info(self, cursor, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """사용자 정보 조회"""
        if 'user_id' in params:
            cursor.execute('SELECT * FROM users WHERE user_id = ?', (params['user_id'],))
        elif 'username' in params:
            cursor.execute('SELECT * FROM users WHERE username = ?', (params['username'],))
        else:
            cursor.execute('SELECT * FROM users LIMIT 10')
        
        return [dict(row) for row in cursor.fetchall()]
    
    def _query_order_history(self, cursor, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """주문 내역 조회"""
        if 'user_id' in params:
            query = '''
                SELECT o.*, u.username, p.name as product_name, p.price
                FROM orders o
                JOIN users u ON o.user_id = u.user_id
                JOIN products p ON o.product_id = p.product_id
                WHERE o.user_id = ?
                ORDER BY o.order_date DESC
            '''
            cursor.execute(query, (params['user_id'],))
        else:
            query = '''
                SELECT o.*, u.username, p.name as product_name, p.price
                FROM orders o
                JOIN users u ON o.user_id = u.user_id
                JOIN products p ON o.product_id = p.product_id
                ORDER BY o.order_date DESC
                LIMIT 20
            '''
            cursor.execute(query)
        
        return [dict(row) for row in cursor.fetchall()]
    
    def _query_product_info(self, cursor, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """상품 정보 조회"""
        if 'product_id' in params:
            cursor.execute('SELECT * FROM products WHERE product_id = ?', (params['product_id'],))
        elif 'category' in params:
            cursor.execute('SELECT * FROM products WHERE category = ?', (params['category'],))
        else:
            cursor.execute('SELECT * FROM products LIMIT 10')
        
        return [dict(row) for row in cursor.fetchall()]
    
    def _query_analytics(self, cursor, params: Dict[str, Any]) -> Dict[str, Any]:
        """분석 데이터 조회"""
        analytics = {}
        
        # 총 사용자 수
        cursor.execute('SELECT COUNT(*) as total_users FROM users')
        analytics['total_users'] = cursor.fetchone()[0]
        
        # 총 주문 수
        cursor.execute('SELECT COUNT(*) as total_orders FROM orders')
        analytics['total_orders'] = cursor.fetchone()[0]
        
        # 총 매출
        cursor.execute('''
            SELECT SUM(p.price * o.quantity) as total_revenue
            FROM orders o
            JOIN products p ON o.product_id = p.product_id
            WHERE o.status = 'completed'
        ''')
        analytics['total_revenue'] = cursor.fetchone()[0] or 0.0
        
        # 인기 상품
        cursor.execute('''
            SELECT p.name, SUM(o.quantity) as total_sold
            FROM orders o
            JOIN products p ON o.product_id = p.product_id
            WHERE o.status = 'completed'
            GROUP BY p.product_id
            ORDER BY total_sold DESC
            LIMIT 5
        ''')
        analytics['popular_products'] = [dict(row) for row in cursor.fetchall()]
        
        return analytics

# Tool Manager
class ToolManager:
    """Tool 관리 및 실행 시스템"""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.call_history: List[ToolCall] = []
        
        # 기본 도구들 등록
        self._register_default_tools()
        
        logger.info(f"Tool Manager 초기화 완료 - 등록된 도구: {len(self.tools)}개")
    
    def _register_default_tools(self):
        """기본 도구들 등록"""
        try:
            # 날씨 도구
            weather_tool = WeatherTool()
            self.register_tool(weather_tool)
            
            # 검색 도구
            search_tool = SearchTool()
            self.register_tool(search_tool)
            
            # 데이터베이스 도구
            db_tool = DatabaseTool()
            self.register_tool(db_tool)
            
            logger.info("기본 도구 등록 완료")
            
        except Exception as e:
            logger.error(f"기본 도구 등록 실패: {e}")
    
    def register_tool(self, tool: BaseTool):
        """도구 등록"""
        self.tools[tool.name] = tool
        logger.info(f"도구 등록: {tool.name}")
    
    def get_openai_functions(self) -> List[Dict[str, Any]]:
        """OpenAI Function Calling용 함수 스펙 목록 반환"""
        return [tool.get_openai_function_spec() for tool in self.tools.values()]
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
        """도구 실행"""
        call_id = str(uuid.uuid4())
        call_timestamp = datetime.now()
        
        logger.info(f"도구 실행: {tool_name} (호출 ID: {call_id[:8]}...)")
        
        # ToolCall 객체 생성
        tool_call = ToolCall(
            tool_name=tool_name,
            parameters=parameters,
            call_id=call_id,
            timestamp=call_timestamp
        )
        
        try:
            if tool_name not in self.tools:
                raise ValueError(f"등록되지 않은 도구: {tool_name}")
            
            tool = self.tools[tool_name]
            result = tool.execute(**parameters)
            tool_call.result = result
            
            # 호출 히스토리에 저장
            self.call_history.append(tool_call)
            
            # 히스토리 크기 제한 (최근 100개만 유지)
            if len(self.call_history) > 100:
                self.call_history = self.call_history[-100:]
            
            logger.info(f"도구 실행 완료: {tool_name} ({'성공' if result.success else '실패'})")
            
            return result
            
        except Exception as e:
            error_result = ToolResult(
                success=False,
                data=None,
                error=str(e),
                execution_time=0.0,
                metadata={'tool_name': tool_name, 'call_id': call_id}
            )
            
            tool_call.result = error_result
            self.call_history.append(tool_call)
            
            logger.error(f"도구 실행 실패: {tool_name} - {str(e)}")
            
            return error_result
    
    def get_call_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """도구 호출 히스토리 조회"""
        recent_calls = self.call_history[-limit:] if limit > 0 else self.call_history
        
        return [
            {
                'call_id': call.call_id,
                'tool_name': call.tool_name,
                'parameters': call.parameters,
                'timestamp': call.timestamp.isoformat(),
                'success': call.result.success if call.result else False,
                'execution_time': call.result.execution_time if call.result else 0.0,
                'error': call.result.error if call.result and not call.result.success else None
            }
            for call in recent_calls
        ]

# Function Calling 챗봇
class FunctionCallingChatbot:
    """OpenAI Function Calling 기능을 활용한 챗봇"""
    
    def __init__(self):
        self.client = OpenAI(api_key=config.llm.openai_api_key)
        self.tool_manager = ToolManager()
        self.conversation_history: List[Dict[str, str]] = []
        
        logger.info("Function Calling 챗봇 초기화 완료")
    
    def chat(self, user_message: str) -> Dict[str, Any]:
        """사용자 메시지 처리 및 응답 생성"""
        start_time = time.time()
        logger.info(f"Function Calling 챗봇 처리 시작: {user_message[:50]}...")
        
        try:
            # 대화 히스토리에 사용자 메시지 추가
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
            
            # OpenAI API 호출 (Function Calling 포함)
            messages = self._build_messages()
            functions = self.tool_manager.get_openai_functions()
            
            response = self.client.chat.completions.create(
                model=config.llm.openai_model,
                messages=messages,
                functions=functions if functions else None,
                function_call="auto" if functions else None,
                temperature=config.llm.temperature
            )
            
            # 응답 처리
            result = self._process_response(response, user_message, start_time)
            
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            
            logger.info(f"Function Calling 처리 완료 - 시간: {processing_time:.2f}초")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Function Calling 처리 실패: {str(e)}"
            logger.error(error_msg)
            
            return {
                'success': False,
                'response': "죄송합니다. 처리 중 오류가 발생했습니다.",
                'error': error_msg,
                'processing_time': processing_time
            }
    
    def _build_messages(self) -> List[Dict[str, str]]:
        """대화 메시지 구성"""
        system_message = {
            "role": "system",
            "content": """당신은 다양한 도구를 활용할 수 있는 AI 어시스턴트입니다.
            
사용 가능한 도구:
- get_weather: 날씨 정보 조회
- web_search: 인터넷 검색
- query_database: 데이터베이스 조회

사용자의 요청에 따라 적절한 도구를 사용하여 정확하고 유용한 정보를 제공하세요.
도구 사용이 필요하지 않은 일반적인 질문에는 직접 답변하세요."""
        }
        
        return [system_message] + self.conversation_history[-10:]  # 최근 10개 메시지만 유지
    
    def _process_response(self, response, user_message: str, start_time: float) -> Dict[str, Any]:
        """OpenAI 응답 처리"""
        message = response.choices[0].message
        
        # Function Call이 있는지 확인
        if hasattr(message, 'function_call') and message.function_call:
            return self._handle_function_call(message, user_message)
        else:
            # 일반 응답 처리
            ai_response = message.content
            self.conversation_history.append({
                "role": "assistant",
                "content": ai_response
            })
            
            return {
                'success': True,
                'response': ai_response,
                'function_calls': [],
                'tokens_used': response.usage.total_tokens
            }
    
    def _handle_function_call(self, message, user_message: str) -> Dict[str, Any]:
        """Function Call 처리"""
        function_call = message.function_call
        function_name = function_call.name
        
        try:
            # 함수 파라미터 파싱
            function_args = json.loads(function_call.arguments)
            
            logger.info(f"Function Call 요청: {function_name}({function_args})")
            
            # 도구 실행
            tool_result = self.tool_manager.execute_tool(function_name, function_args)
            
            # Function Call 결과를 기반으로 추가 응답 생성
            function_result_message = {
                "role": "function",
                "name": function_name,
                "content": json.dumps(tool_result.data if tool_result.success else {"error": tool_result.error})
            }
            
            # 대화 히스토리에 추가
            self.conversation_history.append({
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": function_name,
                    "arguments": function_call.arguments
                }
            })
            self.conversation_history.append(function_result_message)
            
            # 최종 응답 생성
            final_response = self.client.chat.completions.create(
                model=config.llm.openai_model,
                messages=self._build_messages(),
                temperature=config.llm.temperature
            )
            
            final_message = final_response.choices[0].message.content
            self.conversation_history.append({
                "role": "assistant",
                "content": final_message
            })
            
            return {
                'success': True,
                'response': final_message,
                'function_calls': [{
                    'function_name': function_name,
                    'arguments': function_args,
                    'result': tool_result.data if tool_result.success else None,
                    'error': tool_result.error if not tool_result.success else None,
                    'execution_time': tool_result.execution_time
                }],
                'tokens_used': final_response.usage.total_tokens
            }
            
        except Exception as e:
            logger.error(f"Function Call 처리 실패: {e}")
            
            return {
                'success': False,
                'response': f"도구 실행 중 오류가 발생했습니다: {str(e)}",
                'function_calls': [{
                    'function_name': function_name,
                    'error': str(e)
                }]
            }
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """사용 가능한 도구 목록 반환"""
        return [
            {
                'name': tool.name,
                'description': tool.description,
                'parameters': tool.parameters
            }
            for tool in self.tool_manager.tools.values()
        ]
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """대화 히스토리 반환"""
        return self.conversation_history.copy()
    
    def get_tool_call_history(self) -> List[Dict[str, Any]]:
        """도구 호출 히스토리 반환"""
        return self.tool_manager.get_call_history()

# Streamlit 웹 인터페이스
def streamlit_app():
    """Streamlit 기반 웹 인터페이스"""
    st.set_page_config(
        page_title="Function Calling 챗봇",
        page_icon="🔧",
        layout="wide"
    )
    
    st.title("🔧 AI 챗봇 멘토링 - 5차시: 외부 연동 & Tool Calling")
    st.write("OpenAI Function Calling을 활용한 도구 연동 챗봇을 체험해보세요!")
    
    # 세션 상태 초기화
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = FunctionCallingChatbot()
    
    # 사이드바 - 도구 정보 및 설정
    with st.sidebar:
        st.header("🛠️ 사용 가능한 도구")
        
        tools = st.session_state.chatbot.get_available_tools()
        for tool in tools:
            with st.expander(f"📋 {tool['name']}"):
                st.write(f"**설명**: {tool['description']}")
                st.write("**파라미터**:")
                st.json(tool['parameters'])
        
        st.divider()
        
        # 도구 호출 히스토리
        st.header("📊 도구 호출 히스토리")
        if st.button("히스토리 새로고침"):
            st.rerun()
        
        call_history = st.session_state.chatbot.get_tool_call_history()
        
        if call_history:
            st.write(f"최근 {len(call_history)}개 호출")
            for call in call_history[-5:]:  # 최근 5개만 표시
                status_emoji = "✅" if call['success'] else "❌"
                st.write(f"{status_emoji} {call['tool_name']} ({call['execution_time']:.2f}s)")
                st.caption(f"시간: {call['timestamp'][11:19]}")
        else:
            st.write("아직 도구 호출 기록이 없습니다.")
        
        st.divider()
        
        # 시스템 상태
        st.header("🔧 시스템 상태")
        st.success("✅ OpenAI API 연결")
        st.info(f"📊 등록된 도구: {len(tools)}개")
        st.info(f"🔗 모델: {config.llm.openai_model}")
    
    # 메인 영역 - 채팅
    st.header("💬 Function Calling 챗봇")
    
    # 대화 히스토리 표시
    conversation_history = st.session_state.chatbot.get_conversation_history()
    
    for message in conversation_history:
        if message['role'] == 'user':
            with st.chat_message("user"):
                st.write(message['content'])
        elif message['role'] == 'assistant' and message.get('content'):
            with st.chat_message("assistant"):
                st.write(message['content'])
        elif message['role'] == 'function':
            with st.chat_message("assistant"):
                st.info(f"🔧 도구 '{message['name']}' 실행 결과를 처리했습니다.")
    
    # 새 메시지 입력
    if user_input := st.chat_input("메시지를 입력하세요... (예: 서울 날씨 알려줘, Python 검색해줘)"):
        # 사용자 메시지 표시
        with st.chat_message("user"):
            st.write(user_input)
        
        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("응답 생성 중..."):
                result = st.session_state.chatbot.chat(user_input)
            
            if result['success']:
                st.write(result['response'])
                
                # Function Call 정보 표시
                if result.get('function_calls'):
                    with st.expander("🔧 실행된 도구 정보", expanded=False):
                        for call in result['function_calls']:
                            st.write(f"**도구**: {call['function_name']}")
                            st.write(f"**파라미터**: {call['arguments']}")
                            st.write(f"**실행 시간**: {call['execution_time']:.2f}초")
                            
                            if call.get('error'):
                                st.error(f"오류: {call['error']}")
                            else:
                                st.success("성공적으로 실행됨")
                
                # 메타데이터 표시
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"🎯 토큰 사용: {result.get('tokens_used', 'N/A')}")
                with col2:
                    st.caption(f"⏱️ 처리 시간: {result['processing_time']:.2f}초")
            else:
                st.error(f"오류가 발생했습니다: {result.get('error', 'Unknown error')}")
        
        # 페이지 새로고침으로 대화 업데이트
        st.rerun()
    
    # 사용 예시
    st.divider()
    
    with st.expander("💡 사용 예시", expanded=False):
        st.write("""
        **날씨 조회**:
        - "서울 날씨 알려줘"
        - "도쿄의 현재 날씨는?"
        - "뉴욕 날씨 화씨로 알려줘"
        
        **웹 검색**:
        - "Python 최신 뉴스 검색해줘"
        - "인공지능에 대해 검색해서 알려줘"
        - "OpenAI GPT-4 정보 찾아줘"
        
        **데이터베이스 조회**:
        - "사용자 정보 조회해줘"
        - "주문 내역을 보여줘"
        - "상품 정보를 알려줘"
        - "매출 분석 데이터 보여줘"
        
        **일반 대화**:
        - "안녕하세요"
        - "AI에 대해 설명해줘"
        - "프로그래밍 언어 추천해줘"
        """)
    
    # 디버깅 정보
    with st.expander("🔍 디버깅 정보", expanded=False):
        st.write("**대화 히스토리 (JSON)**:")
        st.json(conversation_history[-5:] if conversation_history else [])
        
        st.write("**최근 도구 호출**:")
        recent_calls = st.session_state.chatbot.get_tool_call_history()
        st.json(recent_calls[-3:] if recent_calls else [])

# CLI 데모
def run_cli_demo():
    """CLI 데모 실행"""
    print("=== Function Calling 챗봇 CLI 데모 ===")
    print("사용 가능한 도구:")
    print("  - get_weather: 날씨 조회")
    print("  - web_search: 웹 검색") 
    print("  - query_database: DB 조회")
    print("\n종료하려면 'quit' 또는 'exit'를 입력하세요.\n")
    
    chatbot = FunctionCallingChatbot()
    
    while True:
        try:
            user_input = input("\n사용자: ")
            if user_input.lower() in ['quit', 'exit']:
                print("챗봇을 종료합니다.")
                break
            
            if not user_input.strip():
                continue
            
            print("\n🤖 처리 중...")
            result = chatbot.chat(user_input)
            
            if result['success']:
                print(f"\nAI: {result['response']}")
                
                # Function Call 정보 출력
                if result.get('function_calls'):
                    print(f"\n🔧 실행된 도구:")
                    for call in result['function_calls']:
                        status = "✅" if not call.get('error') else "❌"
                        print(f"  {status} {call['function_name']} ({call['execution_time']:.2f}초)")
                        if call.get('error'):
                            print(f"     오류: {call['error']}")
                
                print(f"\n📊 토큰: {result.get('tokens_used', 'N/A')}, "
                      f"시간: {result['processing_time']:.2f}초")
            else:
                print(f"\n❌ 오류: {result.get('error')}")
        
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
        # Streamlit 모드
        streamlit_app()