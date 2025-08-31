#!/usr/bin/env python3
"""
6차시: 성능 최적화 & 모니터링
Author: AI Chatbot Workshop
Date: 2024-08-30
Description: 응답 캐싱, 비동기 처리, 토큰 최적화, 모니터링 시스템 구현
"""

import os
import sys
import json
import time
import uuid
import logging
import asyncio
import hashlib
import threading
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps, lru_cache
import pickle
import sqlite3

# 외부 라이브러리
import streamlit as st
from openai import OpenAI
import redis
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import asyncio
import aiohttp

# 로컬 모듈
sys.path.append('..')
from config import get_config

# 설정 및 로깅
config = get_config()
logger = logging.getLogger(__name__)

# Prometheus 메트릭 정의
REQUESTS_TOTAL = Counter('chatbot_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('chatbot_request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('chatbot_active_connections', 'Active connections')
CACHE_HITS = Counter('chatbot_cache_hits_total', 'Cache hits')
CACHE_MISSES = Counter('chatbot_cache_misses_total', 'Cache misses')
TOKEN_USAGE = Counter('chatbot_tokens_used_total', 'Tokens used', ['model'])
ERROR_TOTAL = Counter('chatbot_errors_total', 'Total errors', ['error_type'])

@dataclass
class PerformanceMetrics:
    """성능 메트릭 데이터"""
    timestamp: datetime
    request_id: str
    method: str
    endpoint: str
    duration: float
    tokens_used: int
    cache_hit: bool
    error: Optional[str] = None
    user_id: Optional[str] = None
    model: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'request_id': self.request_id,
            'method': self.method,
            'endpoint': self.endpoint,
            'duration': self.duration,
            'tokens_used': self.tokens_used,
            'cache_hit': self.cache_hit,
            'error': self.error,
            'user_id': self.user_id,
            'model': self.model
        }

@dataclass
class SystemMetrics:
    """시스템 리소스 메트릭"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    active_threads: int
    open_files: int

class CacheManager:
    """캐싱 시스템 관리자"""
    
    def __init__(self, redis_url: str = None, ttl: int = 3600, max_local_size: int = 1000):
        """
        캐시 매니저 초기화
        
        Args:
            redis_url: Redis 연결 URL
            ttl: 캐시 만료 시간(초)
            max_local_size: 로컬 캐시 최대 크기
        """
        self.redis_client = None
        self.ttl = ttl
        self.max_local_size = max_local_size
        self.local_cache = {}
        self.local_cache_order = deque()
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'redis_hits': 0,
            'local_hits': 0
        }
        
        self._init_redis(redis_url)
        logger.info(f"캐시 매니저 초기화 완료 - TTL: {ttl}초, 로컬 최대 크기: {max_local_size}")
    
    def _init_redis(self, redis_url: str):
        """Redis 연결 초기화"""
        try:
            if redis_url:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info("Redis 캐시 연결 성공")
            else:
                logger.info("Redis URL이 제공되지 않음, 로컬 캐시만 사용")
        except Exception as e:
            logger.warning(f"Redis 연결 실패, 로컬 캐시만 사용: {e}")
            self.redis_client = None
    
    def _generate_cache_key(self, prefix: str, data: Dict[str, Any]) -> str:
        """캐시 키 생성"""
        # 입력 데이터를 정규화하여 해시 생성
        normalized_data = json.dumps(data, sort_keys=True, ensure_ascii=False)
        hash_key = hashlib.sha256(normalized_data.encode()).hexdigest()[:16]
        return f"{prefix}:{hash_key}"
    
    def get(self, prefix: str, data: Dict[str, Any]) -> Optional[Any]:
        """캐시에서 데이터 조회"""
        cache_key = self._generate_cache_key(prefix, data)
        
        # 1. 로컬 캐시 확인
        if cache_key in self.local_cache:
            cache_entry = self.local_cache[cache_key]
            if datetime.now() < cache_entry['expires_at']:
                self.cache_stats['hits'] += 1
                self.cache_stats['local_hits'] += 1
                CACHE_HITS.inc()
                logger.debug(f"로컬 캐시 히트: {cache_key}")
                return cache_entry['data']
            else:
                # 만료된 로컬 캐시 삭제
                del self.local_cache[cache_key]
                self.local_cache_order.remove(cache_key)
        
        # 2. Redis 캐시 확인
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    data_obj = json.loads(cached_data)
                    # 로컬 캐시에도 저장
                    self._store_local(cache_key, data_obj)
                    self.cache_stats['hits'] += 1
                    self.cache_stats['redis_hits'] += 1
                    CACHE_HITS.inc()
                    logger.debug(f"Redis 캐시 히트: {cache_key}")
                    return data_obj
            except Exception as e:
                logger.error(f"Redis 조회 실패: {e}")
        
        # 캐시 미스
        self.cache_stats['misses'] += 1
        CACHE_MISSES.inc()
        logger.debug(f"캐시 미스: {cache_key}")
        return None
    
    def set(self, prefix: str, data: Dict[str, Any], value: Any):
        """캐시에 데이터 저장"""
        cache_key = self._generate_cache_key(prefix, data)
        
        # 로컬 캐시에 저장
        self._store_local(cache_key, value)
        
        # Redis에 저장
        if self.redis_client:
            try:
                serialized_value = json.dumps(value, ensure_ascii=False, default=str)
                self.redis_client.setex(cache_key, self.ttl, serialized_value)
                logger.debug(f"Redis에 저장: {cache_key}")
            except Exception as e:
                logger.error(f"Redis 저장 실패: {e}")
    
    def _store_local(self, cache_key: str, value: Any):
        """로컬 캐시에 저장"""
        # 크기 제한 확인
        if len(self.local_cache) >= self.max_local_size:
            # LRU 정책으로 오래된 항목 제거
            oldest_key = self.local_cache_order.popleft()
            if oldest_key in self.local_cache:
                del self.local_cache[oldest_key]
        
        expires_at = datetime.now() + timedelta(seconds=self.ttl)
        self.local_cache[cache_key] = {
            'data': value,
            'expires_at': expires_at
        }
        
        if cache_key in self.local_cache_order:
            self.local_cache_order.remove(cache_key)
        self.local_cache_order.append(cache_key)
        
        logger.debug(f"로컬 캐시에 저장: {cache_key}")
    
    def clear(self):
        """캐시 전체 삭제"""
        self.local_cache.clear()
        self.local_cache_order.clear()
        
        if self.redis_client:
            try:
                # 특정 패턴의 키만 삭제 (실제 구현에서는 더 정교하게)
                keys = self.redis_client.keys("*")
                if keys:
                    self.redis_client.delete(*keys)
                logger.info("Redis 캐시 삭제 완료")
            except Exception as e:
                logger.error(f"Redis 캐시 삭제 실패: {e}")
        
        logger.info("캐시 전체 삭제 완료")
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / total_requests) if total_requests > 0 else 0
        
        return {
            'total_requests': total_requests,
            'cache_hits': self.cache_stats['hits'],
            'cache_misses': self.cache_stats['misses'],
            'hit_rate': hit_rate,
            'redis_hits': self.cache_stats['redis_hits'],
            'local_hits': self.cache_stats['local_hits'],
            'local_cache_size': len(self.local_cache),
            'redis_connected': self.redis_client is not None
        }

class TokenOptimizer:
    """토큰 사용량 최적화"""
    
    def __init__(self):
        self.optimization_stats = {
            'original_tokens': 0,
            'optimized_tokens': 0,
            'savings_percentage': 0.0
        }
        logger.info("토큰 최적화기 초기화 완료")
    
    def optimize_prompt(self, prompt: str, max_tokens: int = 4000) -> str:
        """프롬프트 토큰 최적화"""
        original_length = len(prompt)
        
        if original_length <= max_tokens:
            return prompt
        
        # 최적화 전략들 적용
        optimized = prompt
        
        # 1. 중복 공백 제거
        import re
        optimized = re.sub(r'\s+', ' ', optimized)
        
        # 2. 불필요한 구문 제거
        unnecessary_phrases = [
            'please note that',
            'it is important to',
            'you should understand that',
            'as mentioned before',
            'in other words',
            'to summarize'
        ]
        
        for phrase in unnecessary_phrases:
            optimized = optimized.replace(phrase, '')
        
        # 3. 문장 압축
        if len(optimized) > max_tokens:
            sentences = optimized.split('. ')
            # 중요도가 높은 문장 우선 유지 (간단한 휴리스틱)
            important_sentences = []
            current_length = 0
            
            for sentence in sentences:
                if current_length + len(sentence) <= max_tokens:
                    important_sentences.append(sentence)
                    current_length += len(sentence)
                else:
                    break
            
            optimized = '. '.join(important_sentences)
        
        # 통계 업데이트
        self.optimization_stats['original_tokens'] += original_length
        self.optimization_stats['optimized_tokens'] += len(optimized)
        
        if self.optimization_stats['original_tokens'] > 0:
            savings = 1 - (self.optimization_stats['optimized_tokens'] / self.optimization_stats['original_tokens'])
            self.optimization_stats['savings_percentage'] = savings * 100
        
        if len(optimized) < original_length:
            logger.info(f"프롬프트 최적화: {original_length} -> {len(optimized)} 문자 ({((original_length - len(optimized)) / original_length * 100):.1f}% 절약)")
        
        return optimized
    
    def optimize_messages(self, messages: List[Dict[str, str]], max_total_tokens: int = 3000) -> List[Dict[str, str]]:
        """메시지 목록 최적화"""
        if not messages:
            return messages
        
        # 전체 토큰 수 추정
        total_tokens = sum(len(msg.get('content', '')) for msg in messages)
        
        if total_tokens <= max_total_tokens:
            return messages
        
        # 시스템 메시지는 보존
        system_messages = [msg for msg in messages if msg.get('role') == 'system']
        other_messages = [msg for msg in messages if msg.get('role') != 'system']
        
        # 최근 메시지 우선 보존
        optimized_messages = system_messages[:]
        current_tokens = sum(len(msg.get('content', '')) for msg in system_messages)
        
        for msg in reversed(other_messages):
            msg_tokens = len(msg.get('content', ''))
            if current_tokens + msg_tokens <= max_total_tokens:
                optimized_messages.insert(-len(system_messages) if system_messages else 0, msg)
                current_tokens += msg_tokens
            else:
                break
        
        logger.info(f"메시지 최적화: {len(messages)} -> {len(optimized_messages)} 개")
        return optimized_messages
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """최적화 통계 반환"""
        return self.optimization_stats.copy()

class PerformanceMonitor:
    """성능 모니터링 시스템"""
    
    def __init__(self, db_path: str = "performance_metrics.db"):
        self.db_path = db_path
        self.metrics_buffer = deque(maxlen=1000)  # 최근 1000개 메트릭 유지
        self.system_metrics = []
        self.monitoring_active = False
        self.monitor_thread = None
        
        self._init_database()
        logger.info(f"성능 모니터 초기화 완료 - DB: {db_path}")
    
    def _init_database(self):
        """성능 메트릭 DB 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    request_id TEXT,
                    method TEXT,
                    endpoint TEXT,
                    duration REAL,
                    tokens_used INTEGER,
                    cache_hit BOOLEAN,
                    error TEXT,
                    user_id TEXT,
                    model TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    cpu_percent REAL,
                    memory_percent REAL,
                    memory_used_mb REAL,
                    disk_usage_percent REAL,
                    active_threads INTEGER,
                    open_files INTEGER
                )
            ''')
            
            # 인덱스 생성
            conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON performance_metrics(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_endpoint ON performance_metrics(endpoint)')
    
    def record_metric(self, metric: PerformanceMetrics):
        """성능 메트릭 기록"""
        self.metrics_buffer.append(metric)
        
        # Prometheus 메트릭 업데이트
        REQUESTS_TOTAL.labels(method=metric.method, endpoint=metric.endpoint).inc()
        REQUEST_DURATION.observe(metric.duration)
        TOKEN_USAGE.labels(model=metric.model or 'unknown').inc(metric.tokens_used)
        
        if metric.cache_hit:
            CACHE_HITS.inc()
        else:
            CACHE_MISSES.inc()
        
        if metric.error:
            ERROR_TOTAL.labels(error_type=type(metric.error).__name__).inc()
        
        # 주기적으로 DB에 저장
        if len(self.metrics_buffer) >= 100:
            self._flush_metrics_to_db()
    
    def _flush_metrics_to_db(self):
        """메트릭 버퍼를 DB에 저장"""
        if not self.metrics_buffer:
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                metrics_data = [
                    (
                        metric.timestamp.isoformat(),
                        metric.request_id,
                        metric.method,
                        metric.endpoint,
                        metric.duration,
                        metric.tokens_used,
                        metric.cache_hit,
                        metric.error,
                        metric.user_id,
                        metric.model
                    )
                    for metric in self.metrics_buffer
                ]
                
                conn.executemany('''
                    INSERT INTO performance_metrics 
                    (timestamp, request_id, method, endpoint, duration, tokens_used, 
                     cache_hit, error, user_id, model)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', metrics_data)
                
                self.metrics_buffer.clear()
                logger.debug(f"성능 메트릭 {len(metrics_data)}개를 DB에 저장")
                
        except Exception as e:
            logger.error(f"메트릭 DB 저장 실패: {e}")
    
    def start_system_monitoring(self, interval: int = 60):
        """시스템 리소스 모니터링 시작"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._system_monitor_loop, 
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"시스템 모니터링 시작 - 간격: {interval}초")
    
    def stop_system_monitoring(self):
        """시스템 모니터링 중지"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("시스템 모니터링 중지")
    
    def _system_monitor_loop(self, interval: int):
        """시스템 모니터링 루프"""
        while self.monitoring_active:
            try:
                # 시스템 메트릭 수집
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # 프로세스 정보
                process = psutil.Process()
                num_threads = process.num_threads()
                num_fds = len(process.open_files())
                
                system_metric = SystemMetrics(
                    timestamp=datetime.now(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_used_mb=memory.used / (1024 * 1024),
                    disk_usage_percent=disk.percent,
                    active_threads=num_threads,
                    open_files=num_fds
                )
                
                self.system_metrics.append(system_metric)
                
                # 오래된 메트릭 제거 (최근 24시간만 유지)
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.system_metrics = [
                    m for m in self.system_metrics 
                    if m.timestamp > cutoff_time
                ]
                
                # DB에 저장
                self._save_system_metric(system_metric)
                
                # Prometheus 메트릭 업데이트
                ACTIVE_CONNECTIONS.set(num_threads)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"시스템 모니터링 오류: {e}")
                time.sleep(interval)
    
    def _save_system_metric(self, metric: SystemMetrics):
        """시스템 메트릭 DB 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO system_metrics 
                    (timestamp, cpu_percent, memory_percent, memory_used_mb, 
                     disk_usage_percent, active_threads, open_files)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metric.timestamp.isoformat(),
                    metric.cpu_percent,
                    metric.memory_percent,
                    metric.memory_used_mb,
                    metric.disk_usage_percent,
                    metric.active_threads,
                    metric.open_files
                ))
        except Exception as e:
            logger.error(f"시스템 메트릭 DB 저장 실패: {e}")
    
    def get_performance_stats(self, hours: int = 1) -> Dict[str, Any]:
        """성능 통계 조회"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # 기본 통계
                cursor = conn.execute('''
                    SELECT 
                        COUNT(*) as total_requests,
                        AVG(duration) as avg_duration,
                        MAX(duration) as max_duration,
                        MIN(duration) as min_duration,
                        SUM(tokens_used) as total_tokens,
                        AVG(tokens_used) as avg_tokens,
                        SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) as cache_hits,
                        SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as error_count
                    FROM performance_metrics 
                    WHERE timestamp > ?
                ''', (cutoff_time.isoformat(),))
                
                stats = dict(cursor.fetchone())
                
                # 엔드포인트별 통계
                cursor = conn.execute('''
                    SELECT endpoint, COUNT(*) as count, AVG(duration) as avg_duration
                    FROM performance_metrics 
                    WHERE timestamp > ?
                    GROUP BY endpoint
                    ORDER BY count DESC
                ''', (cutoff_time.isoformat(),))
                
                endpoint_stats = [dict(row) for row in cursor.fetchall()]
                
                # 에러 통계
                cursor = conn.execute('''
                    SELECT error, COUNT(*) as count
                    FROM performance_metrics 
                    WHERE timestamp > ? AND error IS NOT NULL
                    GROUP BY error
                    ORDER BY count DESC
                ''', (cutoff_time.isoformat(),))
                
                error_stats = [dict(row) for row in cursor.fetchall()]
                
                return {
                    'period_hours': hours,
                    'basic_stats': stats,
                    'endpoint_stats': endpoint_stats,
                    'error_stats': error_stats,
                    'cache_hit_rate': (stats['cache_hits'] / stats['total_requests']) if stats['total_requests'] > 0 else 0
                }
                
        except Exception as e:
            logger.error(f"성능 통계 조회 실패: {e}")
            return {}
    
    def get_system_stats(self, hours: int = 1) -> Dict[str, Any]:
        """시스템 통계 조회"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                cursor = conn.execute('''
                    SELECT 
                        AVG(cpu_percent) as avg_cpu,
                        MAX(cpu_percent) as max_cpu,
                        AVG(memory_percent) as avg_memory,
                        MAX(memory_percent) as max_memory,
                        AVG(memory_used_mb) as avg_memory_mb,
                        MAX(active_threads) as max_threads,
                        MAX(open_files) as max_open_files
                    FROM system_metrics 
                    WHERE timestamp > ?
                ''', (cutoff_time.isoformat(),))
                
                return dict(cursor.fetchone())
                
        except Exception as e:
            logger.error(f"시스템 통계 조회 실패: {e}")
            return {}

def performance_decorator(endpoint: str, cache_prefix: str = None):
    """성능 모니터링 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            request_id = str(uuid.uuid4())
            start_time = time.time()
            cache_hit = False
            error = None
            tokens_used = 0
            
            # 캐싱 확인
            if cache_prefix and hasattr(wrapper, '_cache_manager'):
                cached_result = wrapper._cache_manager.get(cache_prefix, kwargs)
                if cached_result is not None:
                    cache_hit = True
                    duration = time.time() - start_time
                    
                    if hasattr(wrapper, '_monitor'):
                        metric = PerformanceMetrics(
                            timestamp=datetime.now(),
                            request_id=request_id,
                            method='cached',
                            endpoint=endpoint,
                            duration=duration,
                            tokens_used=0,
                            cache_hit=True
                        )
                        wrapper._monitor.record_metric(metric)
                    
                    return cached_result
            
            try:
                # 함수 실행
                result = func(*args, **kwargs)
                
                # 토큰 사용량 추출 (결과에서)
                if isinstance(result, dict) and 'tokens_used' in result:
                    tokens_used = result['tokens_used']
                
                # 결과 캐싱
                if cache_prefix and hasattr(wrapper, '_cache_manager'):
                    wrapper._cache_manager.set(cache_prefix, kwargs, result)
                
                return result
                
            except Exception as e:
                error = str(e)
                raise
            
            finally:
                # 성능 메트릭 기록
                duration = time.time() - start_time
                
                if hasattr(wrapper, '_monitor'):
                    metric = PerformanceMetrics(
                        timestamp=datetime.now(),
                        request_id=request_id,
                        method=func.__name__,
                        endpoint=endpoint,
                        duration=duration,
                        tokens_used=tokens_used,
                        cache_hit=cache_hit,
                        error=error
                    )
                    wrapper._monitor.record_metric(metric)
        
        return wrapper
    return decorator

class AsyncChatbot:
    """비동기 처리를 지원하는 최적화된 챗봇"""
    
    def __init__(self):
        self.client = OpenAI(api_key=config.llm.openai_api_key)
        self.cache_manager = CacheManager(redis_url=config.get_redis_url())
        self.token_optimizer = TokenOptimizer()
        self.performance_monitor = PerformanceMonitor()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 성능 모니터링 시작
        self.performance_monitor.start_system_monitoring()
        
        logger.info("비동기 최적화 챗봇 초기화 완료")
    
    @performance_decorator(endpoint="/chat", cache_prefix="chat_response")
    def chat(self, user_message: str, user_id: str = None, use_cache: bool = True) -> Dict[str, Any]:
        """최적화된 채팅 응답"""
        start_time = time.time()
        
        # 캐시 키 생성을 위한 데이터
        cache_data = {
            'message': user_message,
            'model': config.llm.openai_model,
            'temperature': config.llm.temperature
        }
        
        # 캐시 확인
        if use_cache:
            cached_response = self.cache_manager.get("chat_response", cache_data)
            if cached_response:
                logger.info(f"캐시된 응답 반환: {user_message[:50]}...")
                return cached_response
        
        try:
            # 프롬프트 최적화
            messages = [
                {"role": "system", "content": "당신은 도움이 되는 AI 어시스턴트입니다."},
                {"role": "user", "content": user_message}
            ]
            
            optimized_messages = self.token_optimizer.optimize_messages(messages)
            
            # OpenAI API 호출
            response = self.client.chat.completions.create(
                model=config.llm.openai_model,
                messages=optimized_messages,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens
            )
            
            processing_time = time.time() - start_time
            tokens_used = response.usage.total_tokens
            ai_response = response.choices[0].message.content
            
            result = {
                'success': True,
                'response': ai_response,
                'tokens_used': tokens_used,
                'processing_time': processing_time,
                'cached': False,
                'user_id': user_id
            }
            
            # 결과 캐싱
            if use_cache:
                self.cache_manager.set("chat_response", cache_data, result)
            
            logger.info(f"새 응답 생성: {user_message[:50]}... ({tokens_used} 토큰, {processing_time:.2f}초)")
            
            return result
            
        except Exception as e:
            error_msg = f"채팅 처리 실패: {str(e)}"
            logger.error(error_msg)
            
            return {
                'success': False,
                'error': error_msg,
                'processing_time': time.time() - start_time,
                'tokens_used': 0,
                'user_id': user_id
            }
    
    async def chat_async(self, user_message: str, user_id: str = None) -> Dict[str, Any]:
        """비동기 채팅 처리"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.chat, user_message, user_id)
    
    async def batch_chat(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """배치 메시지 병렬 처리"""
        tasks = [
            self.chat_async(msg['content'], msg.get('user_id'))
            for msg in messages
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 예외 처리
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'success': False,
                    'error': str(result),
                    'message_index': i
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """종합 성능 통계"""
        cache_stats = self.cache_manager.get_stats()
        optimization_stats = self.token_optimizer.get_optimization_stats()
        performance_stats = self.performance_monitor.get_performance_stats()
        system_stats = self.performance_monitor.get_system_stats()
        
        return {
            'cache': cache_stats,
            'optimization': optimization_stats,
            'performance': performance_stats,
            'system': system_stats,
            'timestamp': datetime.now().isoformat()
        }
    
    def cleanup(self):
        """리소스 정리"""
        self.performance_monitor.stop_system_monitoring()
        self.performance_monitor._flush_metrics_to_db()
        self.executor.shutdown(wait=True)
        logger.info("챗봇 리소스 정리 완료")

class DashboardGenerator:
    """성능 대시보드 생성기"""
    
    def __init__(self, chatbot: AsyncChatbot):
        self.chatbot = chatbot
    
    def generate_performance_charts(self) -> Dict[str, Any]:
        """성능 차트 생성"""
        try:
            # 성능 데이터 조회
            with sqlite3.connect(self.chatbot.performance_monitor.db_path) as conn:
                # 최근 24시간 응답 시간 트렌드
                df_response = pd.read_sql_query('''
                    SELECT 
                        datetime(timestamp) as time,
                        AVG(duration) as avg_duration,
                        COUNT(*) as request_count
                    FROM performance_metrics 
                    WHERE timestamp > datetime('now', '-24 hours')
                    GROUP BY datetime(timestamp, 'localtime', 'start of hour')
                    ORDER BY time
                ''', conn)
                
                # 엔드포인트별 성능
                df_endpoint = pd.read_sql_query('''
                    SELECT 
                        endpoint,
                        COUNT(*) as request_count,
                        AVG(duration) as avg_duration,
                        SUM(tokens_used) as total_tokens
                    FROM performance_metrics 
                    WHERE timestamp > datetime('now', '-24 hours')
                    GROUP BY endpoint
                    ORDER BY request_count DESC
                ''', conn)
                
                # 시스템 리소스 트렌드
                df_system = pd.read_sql_query('''
                    SELECT 
                        datetime(timestamp) as time,
                        AVG(cpu_percent) as cpu_percent,
                        AVG(memory_percent) as memory_percent
                    FROM system_metrics 
                    WHERE timestamp > datetime('now', '-24 hours')
                    GROUP BY datetime(timestamp, 'localtime', 'start of hour')
                    ORDER BY time
                ''', conn)
            
            # 차트 데이터 반환
            return {
                'response_time_trend': df_response.to_dict('records'),
                'endpoint_performance': df_endpoint.to_dict('records'),
                'system_resource_trend': df_system.to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"성능 차트 생성 실패: {e}")
            return {}
    
    def create_matplotlib_charts(self, save_path: str = "performance_dashboard.png"):
        """Matplotlib을 사용한 차트 생성"""
        try:
            chart_data = self.generate_performance_charts()
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('챗봇 성능 대시보드', fontsize=16)
            
            # 1. 응답 시간 트렌드
            if chart_data.get('response_time_trend'):
                df = pd.DataFrame(chart_data['response_time_trend'])
                if not df.empty:
                    ax1.plot(df['time'], df['avg_duration'], marker='o')
                    ax1.set_title('평균 응답 시간 트렌드')
                    ax1.set_xlabel('시간')
                    ax1.set_ylabel('응답 시간 (초)')
                    ax1.tick_params(axis='x', rotation=45)
            
            # 2. 엔드포인트별 요청 수
            if chart_data.get('endpoint_performance'):
                df = pd.DataFrame(chart_data['endpoint_performance'])
                if not df.empty:
                    ax2.bar(df['endpoint'], df['request_count'])
                    ax2.set_title('엔드포인트별 요청 수')
                    ax2.set_xlabel('엔드포인트')
                    ax2.set_ylabel('요청 수')
            
            # 3. 시스템 리소스 사용률
            if chart_data.get('system_resource_trend'):
                df = pd.DataFrame(chart_data['system_resource_trend'])
                if not df.empty:
                    ax3.plot(df['time'], df['cpu_percent'], label='CPU %', marker='o')
                    ax3.plot(df['time'], df['memory_percent'], label='Memory %', marker='s')
                    ax3.set_title('시스템 리소스 사용률')
                    ax3.set_xlabel('시간')
                    ax3.set_ylabel('사용률 (%)')
                    ax3.legend()
                    ax3.tick_params(axis='x', rotation=45)
            
            # 4. 캐시 통계
            cache_stats = self.chatbot.cache_manager.get_stats()
            if cache_stats['total_requests'] > 0:
                labels = ['Cache Hits', 'Cache Misses']
                sizes = [cache_stats['cache_hits'], cache_stats['cache_misses']]
                colors = ['lightgreen', 'lightcoral']
                ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
                ax4.set_title('캐시 히트율')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"성능 차트 저장: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"차트 생성 실패: {e}")
            return None

# Streamlit 웹 인터페이스
def streamlit_app():
    """Streamlit 성능 모니터링 대시보드"""
    st.set_page_config(
        page_title="성능 모니터링 대시보드",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 AI 챗봇 멘토링 - 6차시: 성능 최적화 & 모니터링")
    st.write("실시간 성능 모니터링과 최적화 기능을 체험해보세요!")
    
    # 세션 상태 초기화
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = AsyncChatbot()
        st.session_state.dashboard = DashboardGenerator(st.session_state.chatbot)
    
    # 사이드바 - 설정 및 통계
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 캐시 관리
        st.subheader("🗄️ 캐시 관리")
        cache_stats = st.session_state.chatbot.cache_manager.get_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("캐시 히트율", f"{cache_stats['hit_rate']:.1%}")
            st.metric("캐시 히트", cache_stats['cache_hits'])
        with col2:
            st.metric("총 요청", cache_stats['total_requests'])
            st.metric("캐시 미스", cache_stats['cache_misses'])
        
        if st.button("캐시 전체 삭제"):
            st.session_state.chatbot.cache_manager.clear()
            st.success("캐시가 삭제되었습니다!")
            st.rerun()
        
        st.divider()
        
        # 시스템 상태
        st.subheader("🖥️ 시스템 상태")
        system_stats = st.session_state.chatbot.performance_monitor.get_system_stats()
        
        if system_stats:
            st.metric("평균 CPU", f"{system_stats.get('avg_cpu', 0):.1f}%")
            st.metric("평균 메모리", f"{system_stats.get('avg_memory', 0):.1f}%")
            st.metric("최대 스레드", f"{system_stats.get('max_threads', 0)}")
        
        st.divider()
        
        # 토큰 최적화 통계
        st.subheader("🎯 토큰 최적화")
        opt_stats = st.session_state.chatbot.token_optimizer.get_optimization_stats()
        
        if opt_stats['original_tokens'] > 0:
            st.metric("절약률", f"{opt_stats['savings_percentage']:.1f}%")
            st.metric("원본 토큰", f"{opt_stats['original_tokens']:,}")
            st.metric("최적화 토큰", f"{opt_stats['optimized_tokens']:,}")
        
        # 실시간 모니터링 설정
        st.divider()
        st.subheader("📡 모니터링")
        
        if st.button("Prometheus 메트릭 서버 시작"):
            try:
                start_http_server(8000)
                st.success("Prometheus 서버가 포트 8000에서 시작되었습니다!")
                st.info("http://localhost:8000/metrics 에서 확인 가능")
            except Exception as e:
                st.error(f"서버 시작 실패: {e}")
    
    # 메인 영역
    tab1, tab2, tab3, tab4 = st.tabs(["💬 채팅", "📊 성능 차트", "📈 실시간 통계", "🧪 배치 테스트"])
    
    with tab1:
        st.header("💬 최적화된 채팅")
        
        # 채팅 옵션
        col1, col2 = st.columns([3, 1])
        with col1:
            use_cache = st.checkbox("캐싱 사용", value=True)
        with col2:
            user_id = st.text_input("사용자 ID", value="test_user")
        
        # 채팅 인터페이스
        if user_input := st.chat_input("메시지를 입력하세요..."):
            # 사용자 메시지 표시
            with st.chat_message("user"):
                st.write(user_input)
            
            # AI 응답 생성
            with st.chat_message("assistant"):
                with st.spinner("응답 생성 중..."):
                    result = st.session_state.chatbot.chat(
                        user_message=user_input,
                        user_id=user_id,
                        use_cache=use_cache
                    )
                
                if result['success']:
                    st.write(result['response'])
                    
                    # 성능 정보 표시
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("처리 시간", f"{result['processing_time']:.3f}초")
                    with col2:
                        st.metric("토큰 사용", result['tokens_used'])
                    with col3:
                        cache_status = "캐시됨" if result.get('cached') else "신규"
                        st.metric("캐시 상태", cache_status)
                    with col4:
                        st.metric("사용자 ID", result.get('user_id', 'N/A'))
                else:
                    st.error(f"오류: {result.get('error')}")
    
    with tab2:
        st.header("📊 성능 분석 차트")
        
        if st.button("차트 새로고침"):
            with st.spinner("차트 생성 중..."):
                chart_path = st.session_state.dashboard.create_matplotlib_charts()
                if chart_path and os.path.exists(chart_path):
                    st.image(chart_path)
                    st.success("차트가 성공적으로 생성되었습니다!")
                else:
                    st.warning("차트 생성에 실패했습니다. 충분한 데이터가 있는지 확인하세요.")
        
        # 성능 통계 테이블
        st.subheader("📋 성능 통계 요약")
        perf_stats = st.session_state.chatbot.performance_monitor.get_performance_stats(hours=24)
        
        if perf_stats and perf_stats.get('basic_stats'):
            basic_stats = perf_stats['basic_stats']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("총 요청", basic_stats.get('total_requests', 0))
            with col2:
                st.metric("평균 응답시간", f"{basic_stats.get('avg_duration', 0):.3f}초")
            with col3:
                st.metric("총 토큰", f"{basic_stats.get('total_tokens', 0):,}")
            with col4:
                st.metric("에러 수", basic_stats.get('error_count', 0))
            
            # 엔드포인트별 통계
            if perf_stats.get('endpoint_stats'):
                st.subheader("🔗 엔드포인트별 성능")
                df = pd.DataFrame(perf_stats['endpoint_stats'])
                st.dataframe(df, use_container_width=True)
    
    with tab3:
        st.header("📈 실시간 통계")
        
        # 자동 새로고침
        auto_refresh = st.checkbox("자동 새로고침 (30초)", value=False)
        if auto_refresh:
            time.sleep(30)
            st.rerun()
        
        if st.button("지금 새로고침"):
            st.rerun()
        
        # 종합 성능 통계
        all_stats = st.session_state.chatbot.get_performance_stats()
        
        # 캐시 성능
        st.subheader("🗄️ 캐시 성능")
        cache_stats = all_stats.get('cache', {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("캐시 히트율", f"{cache_stats.get('hit_rate', 0):.1%}")
        with col2:
            st.metric("Redis 연결", "✅" if cache_stats.get('redis_connected') else "❌")
        with col3:
            st.metric("로컬 캐시 크기", cache_stats.get('local_cache_size', 0))
        
        # 토큰 최적화
        st.subheader("🎯 토큰 최적화")
        opt_stats = all_stats.get('optimization', {})
        
        if opt_stats.get('original_tokens', 0) > 0:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("절약된 토큰", f"{opt_stats['original_tokens'] - opt_stats['optimized_tokens']:,}")
            with col2:
                st.metric("절약률", f"{opt_stats['savings_percentage']:.1f}%")
        
        # 시스템 리소스
        st.subheader("🖥️ 시스템 리소스")
        system_stats = all_stats.get('system', {})
        
        if system_stats:
            col1, col2, col3 = st.columns(3)
            with col1:
                cpu_percent = system_stats.get('avg_cpu', 0)
                st.metric("평균 CPU", f"{cpu_percent:.1f}%", 
                         delta=f"{cpu_percent - 50:.1f}%" if cpu_percent > 50 else None)
            with col2:
                memory_percent = system_stats.get('avg_memory', 0)
                st.metric("평균 메모리", f"{memory_percent:.1f}%",
                         delta=f"{memory_percent - 60:.1f}%" if memory_percent > 60 else None)
            with col3:
                st.metric("최대 스레드", system_stats.get('max_threads', 0))
    
    with tab4:
        st.header("🧪 배치 처리 테스트")
        
        st.write("여러 메시지를 동시에 처리하여 병렬 처리 성능을 테스트합니다.")
        
        # 테스트 메시지 입력
        default_messages = """안녕하세요! 
Python에 대해 알려주세요.
AI는 어떻게 작동하나요?
오늘 날씨는 어떤가요?
프로그래밍 언어 추천해주세요."""
        
        test_messages = st.text_area(
            "테스트 메시지들 (줄바꿈으로 구분):",
            value=default_messages,
            height=150
        )
        
        batch_size = st.slider("배치 크기", min_value=1, max_value=10, value=5)
        
        if st.button("배치 테스트 실행"):
            messages = [msg.strip() for msg in test_messages.split('\n') if msg.strip()]
            messages = messages[:batch_size]  # 배치 크기로 제한
            
            if messages:
                with st.spinner(f"{len(messages)}개 메시지 배치 처리 중..."):
                    start_time = time.time()
                    
                    # 배치 처리를 위한 메시지 포맷
                    batch_messages = [
                        {'content': msg, 'user_id': f'batch_user_{i}'}
                        for i, msg in enumerate(messages)
                    ]
                    
                    # 비동기 배치 처리 실행
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        results = loop.run_until_complete(
                            st.session_state.chatbot.batch_chat(batch_messages)
                        )
                        
                        total_time = time.time() - start_time
                        successful_results = [r for r in results if r.get('success')]
                        
                        # 결과 요약
                        st.success(f"배치 처리 완료!")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("총 처리 시간", f"{total_time:.2f}초")
                        with col2:
                            st.metric("성공률", f"{len(successful_results)}/{len(results)}")
                        with col3:
                            avg_time = sum(r.get('processing_time', 0) for r in successful_results) / len(successful_results) if successful_results else 0
                            st.metric("평균 응답시간", f"{avg_time:.3f}초")
                        with col4:
                            total_tokens = sum(r.get('tokens_used', 0) for r in successful_results)
                            st.metric("총 토큰", f"{total_tokens:,}")
                        
                        # 상세 결과
                        st.subheader("📋 상세 결과")
                        for i, (msg, result) in enumerate(zip(messages, results)):
                            with st.expander(f"메시지 {i+1}: {msg[:50]}..."):
                                if result.get('success'):
                                    st.write(f"**응답**: {result['response'][:200]}...")
                                    st.write(f"**처리시간**: {result['processing_time']:.3f}초")
                                    st.write(f"**토큰**: {result['tokens_used']}")
                                else:
                                    st.error(f"오류: {result.get('error')}")
                    
                    finally:
                        loop.close()
            else:
                st.warning("테스트할 메시지를 입력해주세요.")

def run_performance_test():
    """CLI 성능 테스트"""
    print("=== 성능 최적화 챗봇 테스트 ===")
    
    chatbot = AsyncChatbot()
    
    # 성능 테스트 시나리오
    test_messages = [
        "안녕하세요!",
        "Python에 대해 알려주세요.",
        "AI는 어떻게 작동하나요?",
        "프로그래밍을 배우려면 어떻게 해야 하나요?",
        "데이터 사이언스 공부법을 알려주세요."
    ]
    
    print(f"\n🧪 {len(test_messages)}개 메시지로 성능 테스트 시작")
    
    # 캐시 없이 첫 번째 실행
    print("\n1️⃣ 캐시 없이 실행:")
    chatbot.cache_manager.clear()
    
    start_time = time.time()
    results_no_cache = []
    
    for i, message in enumerate(test_messages, 1):
        print(f"  {i}. 처리 중: {message[:30]}...")
        result = chatbot.chat(message, use_cache=False)
        results_no_cache.append(result)
        print(f"     응답시간: {result['processing_time']:.3f}초, 토큰: {result['tokens_used']}")
    
    total_time_no_cache = time.time() - start_time
    
    # 캐시 사용 두 번째 실행
    print("\n2️⃣ 캐시 사용하여 동일 메시지 재실행:")
    
    start_time = time.time()
    results_with_cache = []
    
    for i, message in enumerate(test_messages, 1):
        print(f"  {i}. 처리 중: {message[:30]}...")
        result = chatbot.chat(message, use_cache=True)
        results_with_cache.append(result)
        cache_status = "캐시됨" if result.get('cached') else "신규"
        print(f"     응답시간: {result['processing_time']:.3f}초, 상태: {cache_status}")
    
    total_time_with_cache = time.time() - start_time
    
    # 비동기 배치 처리 테스트
    print("\n3️⃣ 비동기 배치 처리 테스트:")
    
    batch_messages = [{'content': msg, 'user_id': f'test_user_{i}'} for i, msg in enumerate(test_messages)]
    
    start_time = time.time()
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        batch_results = loop.run_until_complete(chatbot.batch_chat(batch_messages))
        batch_time = time.time() - start_time
        
        successful_batch = [r for r in batch_results if r.get('success')]
        
        print(f"  배치 처리 완료: {len(successful_batch)}/{len(batch_results)} 성공")
        print(f"  총 처리시간: {batch_time:.3f}초")
        print(f"  평균 응답시간: {sum(r.get('processing_time', 0) for r in successful_batch) / len(successful_batch):.3f}초")
    
    finally:
        loop.close()
    
    # 결과 요약
    print("\n" + "="*60)
    print("📊 성능 테스트 결과 요약")
    print("="*60)
    
    avg_time_no_cache = sum(r['processing_time'] for r in results_no_cache) / len(results_no_cache)
    avg_time_with_cache = sum(r['processing_time'] for r in results_with_cache) / len(results_with_cache)
    total_tokens_no_cache = sum(r['tokens_used'] for r in results_no_cache)
    
    print(f"🚫 캐시 없음:")
    print(f"  - 총 시간: {total_time_no_cache:.3f}초")
    print(f"  - 평균 응답시간: {avg_time_no_cache:.3f}초")
    print(f"  - 총 토큰: {total_tokens_no_cache:,}")
    
    print(f"\n✅ 캐시 사용:")
    print(f"  - 총 시간: {total_time_with_cache:.3f}초")
    print(f"  - 평균 응답시간: {avg_time_with_cache:.3f}초")
    print(f"  - 시간 절약: {((total_time_no_cache - total_time_with_cache) / total_time_no_cache * 100):.1f}%")
    
    print(f"\n⚡ 비동기 배치:")
    print(f"  - 총 시간: {batch_time:.3f}초")
    print(f"  - 순차 대비 향상: {((total_time_no_cache - batch_time) / total_time_no_cache * 100):.1f}%")
    
    # 캐시 및 최적화 통계
    cache_stats = chatbot.cache_manager.get_stats()
    opt_stats = chatbot.token_optimizer.get_optimization_stats()
    
    print(f"\n📈 캐시 성능:")
    print(f"  - 히트율: {cache_stats['hit_rate']:.1%}")
    print(f"  - 총 요청: {cache_stats['total_requests']}")
    print(f"  - 캐시 히트: {cache_stats['cache_hits']}")
    
    if opt_stats['original_tokens'] > 0:
        print(f"\n🎯 토큰 최적화:")
        print(f"  - 절약률: {opt_stats['savings_percentage']:.1f}%")
        print(f"  - 원본 토큰: {opt_stats['original_tokens']:,}")
        print(f"  - 최적화 토큰: {opt_stats['optimized_tokens']:,}")
    
    # 정리
    chatbot.cleanup()
    print("\n✅ 성능 테스트 완료!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 성능 테스트 모드
        run_performance_test()
    elif len(sys.argv) > 1 and sys.argv[1] == "prometheus":
        # Prometheus 모니터링 서버만 시작
        print("Prometheus 메트릭 서버 시작 중...")
        start_http_server(8000)
        print("메트릭 서버가 포트 8000에서 실행 중입니다.")
        print("http://localhost:8000/metrics 에서 확인하세요.")
        print("종료하려면 Ctrl+C를 누르세요.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("서버를 종료합니다.")
    else:
        # Streamlit 모드
        streamlit_app()