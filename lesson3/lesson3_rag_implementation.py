#!/usr/bin/env python3
"""
AI 챗봇 멘토링 - 3차시: RAG 시스템 구현
Author: AI Chatbot Workshop
Date: 2024-08-30
Description: 문서 기반 검색 증강 생성(RAG) 챗봇 구현
            PDF/TXT 문서 처리, 벡터 임베딩, 유사도 검색, 답변 생성
"""

import os
import sys
import logging
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio
from datetime import datetime
import hashlib
import uuid

# 외부 라이브러리 imports
import streamlit as st
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import openai
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 문서 처리 imports
import pypdf
from docx import Document
import markdown

# 벡터 DB imports
import chromadb
from chromadb.config import Settings
import faiss

# 설정 파일 import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config

# 전역 설정 및 로깅
config = get_config()
console = Console()
app = typer.Typer(help="RAG 챗봇 구현 - 문서 기반 질의응답 시스템")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/lesson3_rag.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """문서 청크 데이터 클래스"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'id': self.id,
            'content': self.content,
            'metadata': self.metadata,
            'embedding': self.embedding
        }

@dataclass
class SearchResult:
    """검색 결과 데이터 클래스"""
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'chunk_id': self.chunk_id,
            'content': self.content,
            'score': self.score,
            'metadata': self.metadata
        }

@dataclass
class RAGResponse:
    """RAG 응답 데이터 클래스"""
    query: str
    answer: str
    sources: List[SearchResult]
    processing_time: float
    tokens_used: int
    metadata: Dict[str, Any]

class DocumentProcessor:
    """문서 처리 파이프라인"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        문서 처리기 초기화
        
        Args:
            chunk_size: 청크 크기 (토큰 기준)
            chunk_overlap: 청크 간 겹치는 부분 크기
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_extensions = ['.pdf', '.txt', '.docx', '.md']
        
        logger.info(f"문서 처리기 초기화 완료 - chunk_size: {chunk_size}, overlap: {chunk_overlap}")
    
    def process_file(self, file_path: str) -> List[DocumentChunk]:
        """
        파일을 처리하여 청크로 분할
        
        Args:
            file_path: 처리할 파일 경로
            
        Returns:
            List[DocumentChunk]: 분할된 문서 청크들
            
        Raises:
            ValueError: 지원하지 않는 파일 형식
            FileNotFoundError: 파일을 찾을 수 없음
        """
        start_time = time.time()
        logger.info(f"파일 처리 시작: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in self.supported_extensions:
            raise ValueError(f"지원하지 않는 파일 형식: {file_ext}")
        
        # 파일 읽기
        try:
            if file_ext == '.pdf':
                text_content = self._extract_pdf(file_path)
            elif file_ext == '.txt':
                text_content = self._extract_txt(file_path)
            elif file_ext == '.docx':
                text_content = self._extract_docx(file_path)
            elif file_ext == '.md':
                text_content = self._extract_markdown(file_path)
            else:
                raise ValueError(f"처리할 수 없는 파일 형식: {file_ext}")
            
            logger.info(f"텍스트 추출 완료 - 길이: {len(text_content)}자")
            
            # 스마트 청킹
            chunks = self._smart_chunking(text_content, file_path)
            
            processing_time = time.time() - start_time
            logger.info(f"파일 처리 완료 - 청크 수: {len(chunks)}, 처리 시간: {processing_time:.2f}초")
            
            return chunks
            
        except Exception as e:
            logger.error(f"파일 처리 중 오류 발생: {str(e)}")
            raise
    
    def _extract_pdf(self, file_path: str) -> str:
        """PDF 파일에서 텍스트 추출"""
        logger.debug(f"PDF 텍스트 추출 시작: {file_path}")
        
        text_content = []
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():  # 빈 페이지 제외
                        text_content.append(f"[Page {page_num + 1}]\n{page_text}")
                        logger.debug(f"페이지 {page_num + 1} 텍스트 추출 완료")
                
                full_text = '\n\n'.join(text_content)
                logger.info(f"PDF 텍스트 추출 완료 - 페이지 수: {len(pdf_reader.pages)}")
                
                return full_text
                
        except Exception as e:
            logger.error(f"PDF 텍스트 추출 실패: {str(e)}")
            raise
    
    def _extract_txt(self, file_path: str) -> str:
        """TXT 파일 읽기"""
        logger.debug(f"TXT 파일 읽기: {file_path}")
        
        try:
            # 인코딩 자동 감지 시도
            encodings = ['utf-8', 'cp949', 'euc-kr', 'utf-16']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        content = file.read()
                    logger.info(f"TXT 파일 읽기 완료 - 인코딩: {encoding}")
                    return content
                except UnicodeDecodeError:
                    continue
            
            raise ValueError("지원하는 인코딩으로 파일을 읽을 수 없습니다")
            
        except Exception as e:
            logger.error(f"TXT 파일 읽기 실패: {str(e)}")
            raise
    
    def _extract_docx(self, file_path: str) -> str:
        """DOCX 파일에서 텍스트 추출"""
        logger.debug(f"DOCX 텍스트 추출 시작: {file_path}")
        
        try:
            doc = Document(file_path)
            paragraphs = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
            
            full_text = '\n'.join(paragraphs)
            logger.info(f"DOCX 텍스트 추출 완료 - 문단 수: {len(paragraphs)}")
            
            return full_text
            
        except Exception as e:
            logger.error(f"DOCX 텍스트 추출 실패: {str(e)}")
            raise
    
    def _extract_markdown(self, file_path: str) -> str:
        """Markdown 파일 처리"""
        logger.debug(f"Markdown 파일 처리: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                md_content = file.read()
            
            # HTML로 변환 후 텍스트만 추출 (선택사항)
            # 여기서는 원본 마크다운을 그대로 사용
            logger.info("Markdown 파일 읽기 완료")
            return md_content
            
        except Exception as e:
            logger.error(f"Markdown 파일 처리 실패: {str(e)}")
            raise
    
    def _smart_chunking(self, text: str, file_path: str) -> List[DocumentChunk]:
        """
        스마트 청킹 알고리즘
        문단, 문장 경계를 고려한 지능적 분할
        
        Args:
            text: 분할할 텍스트
            file_path: 원본 파일 경로 (메타데이터용)
            
        Returns:
            List[DocumentChunk]: 분할된 청크들
        """
        logger.debug("스마트 청킹 시작")
        
        # 1단계: 문단 기준 분할 시도
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            # 현재 청크에 추가했을 때 크기 확인
            potential_chunk = current_chunk + '\n\n' + paragraph if current_chunk else paragraph
            
            if len(potential_chunk) <= self.chunk_size:
                # 크기가 허용 범위 내면 추가
                current_chunk = potential_chunk
            else:
                # 현재 청크를 저장하고 새 청크 시작
                if current_chunk:
                    chunk = self._create_chunk(
                        current_chunk, 
                        chunk_index, 
                        file_path
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # 단일 문단이 너무 클 경우 문장 기준으로 분할
                if len(paragraph) > self.chunk_size:
                    sub_chunks = self._split_by_sentences(paragraph, chunk_index, file_path)
                    chunks.extend(sub_chunks)
                    chunk_index += len(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = paragraph
        
        # 마지막 청크 저장
        if current_chunk:
            chunk = self._create_chunk(current_chunk, chunk_index, file_path)
            chunks.append(chunk)
        
        logger.info(f"스마트 청킹 완료 - 생성된 청크 수: {len(chunks)}")
        
        # 청킹 결과 로깅
        for i, chunk in enumerate(chunks):
            logger.debug(f"청크 {i}: 길이 {len(chunk.content)}자, ID: {chunk.id}")
        
        return chunks
    
    def _split_by_sentences(self, text: str, start_index: int, file_path: str) -> List[DocumentChunk]:
        """문장 기준으로 텍스트 분할"""
        logger.debug("문장 기준 분할 시작")
        
        # 간단한 문장 분할 (실제 구현에서는 더 정교한 알고리즘 사용 가능)
        sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
        chunks = []
        current_chunk = ""
        chunk_index = start_index
        
        for sentence in sentences:
            potential_chunk = current_chunk + ' ' + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                if current_chunk:
                    chunk = self._create_chunk(current_chunk, chunk_index, file_path)
                    chunks.append(chunk)
                    chunk_index += 1
                current_chunk = sentence
        
        if current_chunk:
            chunk = self._create_chunk(current_chunk, chunk_index, file_path)
            chunks.append(chunk)
        
        logger.debug(f"문장 기준 분할 완료 - 생성된 청크 수: {len(chunks)}")
        return chunks
    
    def _create_chunk(self, content: str, index: int, file_path: str) -> DocumentChunk:
        """청크 객체 생성"""
        chunk_id = self._generate_chunk_id(content, index)
        
        metadata = {
            'source_file': os.path.basename(file_path),
            'file_path': file_path,
            'chunk_index': index,
            'length': len(content),
            'created_at': datetime.now().isoformat(),
            'word_count': len(content.split()),
            'line_count': content.count('\n') + 1
        }
        
        return DocumentChunk(
            id=chunk_id,
            content=content,
            metadata=metadata
        )
    
    def _generate_chunk_id(self, content: str, index: int) -> str:
        """청크 ID 생성 (내용 해시 + 인덱스)"""
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
        return f"chunk_{index:04d}_{content_hash}"

class EmbeddingManager:
    """임베딩 생성 및 관리"""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        """
        임베딩 매니저 초기화
        
        Args:
            model: 사용할 임베딩 모델명
        """
        self.model = model
        self.client = openai.OpenAI(api_key=config.llm.openai_api_key)
        self.embedding_cache = {}  # 임베딩 캐싱
        
        logger.info(f"임베딩 매니저 초기화 완료 - 모델: {model}")
    
    def generate_embeddings(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        청크들의 임베딩 생성
        
        Args:
            chunks: 임베딩을 생성할 청크들
            
        Returns:
            List[DocumentChunk]: 임베딩이 추가된 청크들
        """
        start_time = time.time()
        logger.info(f"임베딩 생성 시작 - 청크 수: {len(chunks)}")
        
        # 배치 처리를 위한 텍스트 리스트 준비
        texts = [chunk.content for chunk in chunks]
        
        try:
            # OpenAI 임베딩 API 호출 (배치 처리)
            logger.debug(f"OpenAI 임베딩 API 호출 - 모델: {self.model}")
            
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            
            # 응답에서 임베딩 추출하여 청크에 할당
            for i, chunk in enumerate(chunks):
                embedding = response.data[i].embedding
                chunk.embedding = embedding
                
                # 캐싱
                self.embedding_cache[chunk.id] = embedding
                
                logger.debug(f"청크 {chunk.id} 임베딩 생성 완료 - 차원: {len(embedding)}")
            
            processing_time = time.time() - start_time
            tokens_used = response.usage.total_tokens
            
            logger.info(f"임베딩 생성 완료 - 처리 시간: {processing_time:.2f}초, 토큰 사용: {tokens_used}")
            
            return chunks
            
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {str(e)}")
            raise
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        쿼리 임베딩 생성
        
        Args:
            query: 검색 쿼리
            
        Returns:
            List[float]: 쿼리 임베딩 벡터
        """
        logger.debug(f"쿼리 임베딩 생성: {query[:50]}...")
        
        # 캐시 확인
        query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
        if query_hash in self.embedding_cache:
            logger.debug("캐시된 쿼리 임베딩 사용")
            return self.embedding_cache[query_hash]
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=[query]
            )
            
            embedding = response.data[0].embedding
            
            # 캐싱
            self.embedding_cache[query_hash] = embedding
            
            logger.debug(f"쿼리 임베딩 생성 완료 - 차원: {len(embedding)}")
            return embedding
            
        except Exception as e:
            logger.error(f"쿼리 임베딩 생성 실패: {str(e)}")
            raise

class VectorDatabase:
    """벡터 데이터베이스 인터페이스"""
    
    def __init__(self, db_type: str = "chroma", persist_dir: str = "./data/vector_db"):
        """
        벡터 DB 초기화
        
        Args:
            db_type: 사용할 벡터 DB 유형 ('chroma' 또는 'faiss')
            persist_dir: 데이터 저장 디렉토리
        """
        self.db_type = db_type
        self.persist_dir = persist_dir
        
        # 저장 디렉토리 생성
        os.makedirs(persist_dir, exist_ok=True)
        
        if db_type == "chroma":
            self._init_chroma()
        elif db_type == "faiss":
            self._init_faiss()
        else:
            raise ValueError(f"지원하지 않는 벡터 DB 유형: {db_type}")
        
        logger.info(f"벡터 DB 초기화 완료 - 유형: {db_type}, 저장 경로: {persist_dir}")
    
    def _init_chroma(self):
        """ChromaDB 초기화"""
        logger.debug("ChromaDB 초기화")
        
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        
        # 컬렉션 생성 또는 가져오기
        self.collection_name = "rag_documents"
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"기존 컬렉션 로드: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "RAG 문서 컬렉션"}
            )
            logger.info(f"새 컬렉션 생성: {self.collection_name}")
    
    def _init_faiss(self):
        """FAISS 초기화"""
        logger.debug("FAISS 초기화")
        
        # FAISS 인덱스는 데이터 추가 시점에 생성
        self.index = None
        self.chunk_mapping = {}  # 인덱스 ID -> 청크 매핑
        self.index_file = os.path.join(self.persist_dir, "faiss.index")
        self.mapping_file = os.path.join(self.persist_dir, "chunk_mapping.json")
        
        # 기존 인덱스 로드 시도
        if os.path.exists(self.index_file) and os.path.exists(self.mapping_file):
            try:
                self.index = faiss.read_index(self.index_file)
                with open(self.mapping_file, 'r', encoding='utf-8') as f:
                    self.chunk_mapping = json.load(f)
                logger.info("기존 FAISS 인덱스 로드 완료")
            except Exception as e:
                logger.warning(f"FAISS 인덱스 로드 실패, 새로 생성: {str(e)}")
    
    def add_chunks(self, chunks: List[DocumentChunk]):
        """
        청크들을 벡터 DB에 추가
        
        Args:
            chunks: 추가할 청크들 (임베딩 포함)
        """
        logger.info(f"벡터 DB에 청크 추가 - 청크 수: {len(chunks)}")
        
        if self.db_type == "chroma":
            self._add_to_chroma(chunks)
        elif self.db_type == "faiss":
            self._add_to_faiss(chunks)
        
        logger.info("벡터 DB 추가 완료")
    
    def _add_to_chroma(self, chunks: List[DocumentChunk]):
        """ChromaDB에 청크 추가"""
        logger.debug("ChromaDB에 청크 추가")
        
        # 데이터 준비
        ids = [chunk.id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # 배치 추가
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            logger.debug(f"ChromaDB 배치 추가 완료 - {len(chunks)}개 청크")
            
        except Exception as e:
            logger.error(f"ChromaDB 추가 실패: {str(e)}")
            raise
    
    def _add_to_faiss(self, chunks: List[DocumentChunk]):
        """FAISS에 청크 추가"""
        logger.debug("FAISS에 청크 추가")
        
        if not chunks[0].embedding:
            raise ValueError("FAISS 추가를 위해서는 임베딩이 필요합니다")
        
        embeddings = np.array([chunk.embedding for chunk in chunks]).astype('float32')
        dimension = embeddings.shape[1]
        
        # 인덱스 생성 (처음 추가 시)
        if self.index is None:
            self.index = faiss.IndexFlatIP(dimension)  # Inner Product (코사인 유사도용)
            logger.debug(f"새 FAISS 인덱스 생성 - 차원: {dimension}")
        
        # 청크 매핑 업데이트
        start_idx = len(self.chunk_mapping)
        for i, chunk in enumerate(chunks):
            self.chunk_mapping[str(start_idx + i)] = {
                'id': chunk.id,
                'content': chunk.content,
                'metadata': chunk.metadata
            }
        
        # 임베딩 추가
        self.index.add(embeddings)
        
        # 인덱스 저장
        faiss.write_index(self.index, self.index_file)
        with open(self.mapping_file, 'w', encoding='utf-8') as f:
            json.dump(self.chunk_mapping, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"FAISS 추가 완료 - 총 벡터 수: {self.index.ntotal}")
    
    def similarity_search(self, query_embedding: List[float], k: int = 5) -> List[SearchResult]:
        """
        유사도 검색 수행
        
        Args:
            query_embedding: 쿼리 임베딩 벡터
            k: 반환할 결과 수
            
        Returns:
            List[SearchResult]: 검색 결과들
        """
        logger.debug(f"유사도 검색 시작 - k: {k}")
        
        if self.db_type == "chroma":
            return self._search_chroma(query_embedding, k)
        elif self.db_type == "faiss":
            return self._search_faiss(query_embedding, k)
    
    def _search_chroma(self, query_embedding: List[float], k: int) -> List[SearchResult]:
        """ChromaDB 검색"""
        logger.debug("ChromaDB 검색 수행")
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )
            
            search_results = []
            for i in range(len(results['ids'][0])):
                result = SearchResult(
                    chunk_id=results['ids'][0][i],
                    content=results['documents'][0][i],
                    score=1 - results['distances'][0][i],  # 거리를 유사도로 변환
                    metadata=results['metadatas'][0][i]
                )
                search_results.append(result)
            
            logger.debug(f"ChromaDB 검색 완료 - 결과 수: {len(search_results)}")
            return search_results
            
        except Exception as e:
            logger.error(f"ChromaDB 검색 실패: {str(e)}")
            raise
    
    def _search_faiss(self, query_embedding: List[float], k: int) -> List[SearchResult]:
        """FAISS 검색"""
        logger.debug("FAISS 검색 수행")
        
        if self.index is None or self.index.ntotal == 0:
            logger.warning("FAISS 인덱스가 비어있습니다")
            return []
        
        try:
            query_vector = np.array([query_embedding]).astype('float32')
            
            # 검색 수행
            scores, indices = self.index.search(query_vector, k)
            
            search_results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                if idx == -1:  # FAISS에서 결과 없음을 나타냄
                    continue
                
                chunk_data = self.chunk_mapping[str(idx)]
                result = SearchResult(
                    chunk_id=chunk_data['id'],
                    content=chunk_data['content'],
                    score=float(scores[0][i]),
                    metadata=chunk_data['metadata']
                )
                search_results.append(result)
            
            logger.debug(f"FAISS 검색 완료 - 결과 수: {len(search_results)}")
            return search_results
            
        except Exception as e:
            logger.error(f"FAISS 검색 실패: {str(e)}")
            raise

class RAGGenerator:
    """RAG 응답 생성기"""
    
    def __init__(self, model: str = "gpt-4o-mini", max_context_length: int = 4000):
        """
        RAG 생성기 초기화
        
        Args:
            model: 사용할 LLM 모델
            max_context_length: 최대 컨텍스트 길이
        """
        self.model = model
        self.max_context_length = max_context_length
        self.client = openai.OpenAI(api_key=config.llm.openai_api_key)
        
        logger.info(f"RAG 생성기 초기화 완료 - 모델: {model}")
    
    def generate_response(
        self, 
        query: str, 
        search_results: List[SearchResult],
        system_prompt: Optional[str] = None
    ) -> RAGResponse:
        """
        검색 결과를 기반으로 응답 생성
        
        Args:
            query: 사용자 질의
            search_results: 검색 결과들
            system_prompt: 시스템 프롬프트 (선택사항)
            
        Returns:
            RAGResponse: 생성된 응답
        """
        start_time = time.time()
        logger.info(f"RAG 응답 생성 시작 - 쿼리: {query[:50]}...")
        
        # 기본 시스템 프롬프트
        if system_prompt is None:
            system_prompt = """당신은 문서 기반 질의응답 시스템입니다. 
제공된 문서 내용을 바탕으로 정확하고 도움이 되는 답변을 제공하세요.
답변할 수 없는 내용이라면 솔직히 모른다고 하세요.
답변 마지막에는 참고한 출처를 명시하세요."""
        
        # 컨텍스트 구성
        context = self._build_context(search_results)
        
        # 프롬프트 구성
        user_prompt = f"""
다음 문서 내용을 참고하여 질문에 답해주세요:

**참고 문서:**
{context}

**질문:** {query}

**답변:**"""
        
        try:
            logger.debug("LLM API 호출 시작")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens
            )
            
            answer = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            processing_time = time.time() - start_time
            
            logger.info(f"RAG 응답 생성 완료 - 처리 시간: {processing_time:.2f}초, 토큰: {tokens_used}")
            
            # 응답 메타데이터
            metadata = {
                'model': self.model,
                'search_results_count': len(search_results),
                'context_length': len(context),
                'prompt_length': len(user_prompt),
                'system_prompt_length': len(system_prompt)
            }
            
            return RAGResponse(
                query=query,
                answer=answer,
                sources=search_results,
                processing_time=processing_time,
                tokens_used=tokens_used,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"RAG 응답 생성 실패: {str(e)}")
            raise
    
    def _build_context(self, search_results: List[SearchResult]) -> str:
        """검색 결과로부터 컨텍스트 구성"""
        logger.debug(f"컨텍스트 구성 - 검색 결과 수: {len(search_results)}")
        
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(search_results):
            # 출처 정보 포함
            source_info = f"[출처 {i+1}: {result.metadata.get('source_file', 'Unknown')}]"
            content_part = f"{source_info}\n{result.content}\n"
            
            # 컨텍스트 길이 확인
            if current_length + len(content_part) > self.max_context_length:
                logger.debug(f"컨텍스트 길이 제한으로 {i}번째에서 중단")
                break
            
            context_parts.append(content_part)
            current_length += len(content_part)
        
        context = "\n---\n".join(context_parts)
        logger.debug(f"컨텍스트 구성 완료 - 길이: {len(context)}자")
        
        return context

class RAGChatbot:
    """통합 RAG 챗봇 시스템"""
    
    def __init__(
        self,
        vector_db_type: str = "chroma",
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini"
    ):
        """
        RAG 챗봇 초기화
        
        Args:
            vector_db_type: 벡터 DB 유형
            embedding_model: 임베딩 모델
            llm_model: LLM 모델
        """
        self.session_id = str(uuid.uuid4())
        
        # 컴포넌트 초기화
        self.doc_processor = DocumentProcessor(
            chunk_size=config.rag.chunk_size,
            chunk_overlap=config.rag.chunk_overlap
        )
        
        self.embedding_manager = EmbeddingManager(model=embedding_model)
        
        self.vector_db = VectorDatabase(
            db_type=vector_db_type,
            persist_dir=config.rag.chroma_persist_dir
        )
        
        self.rag_generator = RAGGenerator(model=llm_model)
        
        # 세션 메타데이터
        self.session_metadata = {
            'session_id': self.session_id,
            'created_at': datetime.now().isoformat(),
            'vector_db_type': vector_db_type,
            'embedding_model': embedding_model,
            'llm_model': llm_model
        }
        
        logger.info(f"RAG 챗봇 초기화 완료 - 세션 ID: {self.session_id}")
    
    def add_document(self, file_path: str) -> Dict[str, Any]:
        """
        문서를 RAG 시스템에 추가
        
        Args:
            file_path: 추가할 문서 파일 경로
            
        Returns:
            Dict[str, Any]: 처리 결과 정보
        """
        start_time = time.time()
        logger.info(f"문서 추가 시작: {file_path}")
        
        try:
            # 1단계: 문서 처리 및 청킹
            logger.info("1단계: 문서 처리 및 청킹")
            chunks = self.doc_processor.process_file(file_path)
            
            # 2단계: 임베딩 생성
            logger.info("2단계: 임베딩 생성")
            chunks_with_embeddings = self.embedding_manager.generate_embeddings(chunks)
            
            # 3단계: 벡터 DB에 저장
            logger.info("3단계: 벡터 DB에 저장")
            self.vector_db.add_chunks(chunks_with_embeddings)
            
            processing_time = time.time() - start_time
            
            result = {
                'success': True,
                'file_path': file_path,
                'chunks_count': len(chunks),
                'processing_time': processing_time,
                'session_id': self.session_id
            }
            
            logger.info(f"문서 추가 완료 - 청크 수: {len(chunks)}, 처리 시간: {processing_time:.2f}초")
            return result
            
        except Exception as e:
            logger.error(f"문서 추가 실패: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'file_path': file_path,
                'session_id': self.session_id
            }
    
    def query(self, question: str, k: int = 5) -> RAGResponse:
        """
        질의 처리 및 응답 생성
        
        Args:
            question: 사용자 질의
            k: 검색할 문서 수
            
        Returns:
            RAGResponse: 생성된 응답
        """
        start_time = time.time()
        logger.info(f"질의 처리 시작: {question[:100]}...")
        
        try:
            # 1단계: 쿼리 임베딩 생성
            logger.debug("1단계: 쿼리 임베딩 생성")
            query_embedding = self.embedding_manager.generate_query_embedding(question)
            
            # 2단계: 유사도 검색
            logger.debug("2단계: 유사도 검색")
            search_results = self.vector_db.similarity_search(query_embedding, k)
            
            # 검색 결과 로깅
            logger.info(f"검색 완료 - 결과 수: {len(search_results)}")
            for i, result in enumerate(search_results):
                logger.debug(f"검색 결과 {i+1}: 점수 {result.score:.4f}, 출처: {result.metadata.get('source_file', 'Unknown')}")
            
            # 3단계: RAG 응답 생성
            logger.debug("3단계: RAG 응답 생성")
            response = self.rag_generator.generate_response(question, search_results)
            
            total_time = time.time() - start_time
            logger.info(f"질의 처리 완료 - 총 처리 시간: {total_time:.2f}초")
            
            return response
            
        except Exception as e:
            logger.error(f"질의 처리 실패: {str(e)}")
            raise

# Streamlit 웹 인터페이스
def streamlit_app():
    """Streamlit 기반 웹 인터페이스"""
    st.set_page_config(
        page_title="RAG 챗봇 - 문서 기반 질의응답",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 RAG 챗봇 - 문서 기반 질의응답")
    st.write("PDF, TXT, DOCX, Markdown 파일을 업로드하고 질문해보세요!")
    
    # 세션 상태 초기화
    if 'rag_chatbot' not in st.session_state:
        st.session_state.rag_chatbot = RAGChatbot(
            vector_db_type=config.rag.vector_db_type,
            embedding_model=config.rag.embedding_model
        )
    
    # 사이드바 - 문서 업로드
    with st.sidebar:
        st.header("📁 문서 업로드")
        
        uploaded_file = st.file_uploader(
            "문서를 선택하세요",
            type=['pdf', 'txt', 'docx', 'md'],
            help="PDF, TXT, DOCX, Markdown 파일을 업로드할 수 있습니다."
        )
        
        if uploaded_file is not None:
            if st.button("문서 추가", type="primary"):
                # 임시 파일로 저장
                temp_dir = "temp_uploads"
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # 문서 처리
                with st.spinner("문서를 처리하고 있습니다..."):
                    result = st.session_state.rag_chatbot.add_document(temp_path)
                
                if result['success']:
                    st.success(f"문서 추가 완료!\n청크 수: {result['chunks_count']}\n처리 시간: {result['processing_time']:.2f}초")
                else:
                    st.error(f"문서 추가 실패: {result['error']}")
                
                # 임시 파일 삭제
                os.remove(temp_path)
        
        st.header("⚙️ 설정")
        
        # 검색 결과 수 설정
        k_results = st.slider("검색 결과 수", min_value=1, max_value=10, value=5)
        
        # 벡터 DB 정보
        st.info(f"벡터 DB: {config.rag.vector_db_type}")
        st.info(f"임베딩 모델: {config.rag.embedding_model}")
    
    # 메인 영역 - 챗봇 인터페이스
    st.header("💬 질의응답")
    
    # 질문 입력
    question = st.text_input(
        "질문을 입력하세요:",
        placeholder="업로드한 문서에 대해 궁금한 것을 물어보세요...",
        help="업로드한 문서 내용을 기반으로 답변을 생성합니다."
    )
    
    # 질문 처리
    if st.button("질문하기", type="primary") and question:
        with st.spinner("답변을 생성하고 있습니다..."):
            try:
                response = st.session_state.rag_chatbot.query(question, k=k_results)
                
                # 답변 표시
                st.subheader("🤖 답변")
                st.write(response.answer)
                
                # 메타데이터 표시
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("처리 시간", f"{response.processing_time:.2f}초")
                with col2:
                    st.metric("토큰 사용", f"{response.tokens_used}")
                with col3:
                    st.metric("참조 문서", f"{len(response.sources)}개")
                
                # 참고 문서 표시
                with st.expander("📋 참고한 문서 내용", expanded=False):
                    for i, source in enumerate(response.sources):
                        st.write(f"**참고 문서 {i+1}** (유사도: {source.score:.4f})")
                        st.write(f"출처: {source.metadata.get('source_file', 'Unknown')}")
                        st.write(source.content[:500] + "..." if len(source.content) > 500 else source.content)
                        st.write("---")
                
            except Exception as e:
                st.error(f"오류 발생: {str(e)}")
    
    # 디버깅 정보
    with st.expander("🔧 디버그 정보", expanded=False):
        st.json(st.session_state.rag_chatbot.session_metadata)

# CLI 인터페이스
@app.command()
def add_document(
    file_path: str = typer.Argument(..., help="추가할 문서 파일 경로"),
    vector_db: str = typer.Option("chroma", help="벡터 DB 유형 (chroma/faiss)"),
    verbose: bool = typer.Option(False, help="상세 로그 출력")
):
    """문서를 RAG 시스템에 추가"""
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    console.print(f"[bold blue]문서 추가 시작:[/bold blue] {file_path}")
    
    try:
        # RAG 챗봇 초기화
        chatbot = RAGChatbot(vector_db_type=vector_db)
        
        # 진행 상황 표시
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            task = progress.add_task("문서 처리 중...", total=None)
            
            # 문서 추가
            result = chatbot.add_document(file_path)
            
            progress.remove_task(task)
        
        if result['success']:
            console.print(f"[bold green]✓ 문서 추가 완료![/bold green]")
            
            # 결과 테이블
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("항목", style="cyan")
            table.add_column("값", style="green")
            
            table.add_row("파일 경로", result['file_path'])
            table.add_row("청크 수", str(result['chunks_count']))
            table.add_row("처리 시간", f"{result['processing_time']:.2f}초")
            table.add_row("세션 ID", result['session_id'])
            
            console.print(table)
        else:
            console.print(f"[bold red]✗ 문서 추가 실패:[/bold red] {result['error']}")
            
    except Exception as e:
        console.print(f"[bold red]오류 발생:[/bold red] {str(e)}")

@app.command()
def query(
    question: str = typer.Argument(..., help="질문 내용"),
    k: int = typer.Option(5, help="검색할 문서 수"),
    vector_db: str = typer.Option("chroma", help="벡터 DB 유형 (chroma/faiss)"),
    verbose: bool = typer.Option(False, help="상세 로그 출력")
):
    """RAG 시스템에 질문하기"""
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    console.print(f"[bold blue]질문:[/bold blue] {question}")
    
    try:
        # RAG 챗봇 초기화
        chatbot = RAGChatbot(vector_db_type=vector_db)
        
        # 진행 상황 표시
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            task = progress.add_task("답변 생성 중...", total=None)
            
            # 질의 처리
            response = chatbot.query(question, k=k)
            
            progress.remove_task(task)
        
        # 답변 출력
        console.print(f"\n[bold green]🤖 답변:[/bold green]")
        console.print(response.answer)
        
        # 메타데이터 테이블
        console.print(f"\n[bold cyan]📊 처리 정보:[/bold cyan]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("항목", style="cyan")
        table.add_column("값", style="green")
        
        table.add_row("처리 시간", f"{response.processing_time:.2f}초")
        table.add_row("토큰 사용", str(response.tokens_used))
        table.add_row("참조 문서 수", str(len(response.sources)))
        
        console.print(table)
        
        # 참조 문서 출력 (verbose 모드)
        if verbose and response.sources:
            console.print(f"\n[bold yellow]📋 참조 문서:[/bold yellow]")
            for i, source in enumerate(response.sources):
                console.print(f"\n[bold]참조 {i+1}[/bold] (유사도: {source.score:.4f})")
                console.print(f"출처: {source.metadata.get('source_file', 'Unknown')}")
                console.print(source.content[:200] + "..." if len(source.content) > 200 else source.content)
                
    except Exception as e:
        console.print(f"[bold red]오류 발생:[/bold red] {str(e)}")

@app.command()
def web():
    """웹 인터페이스 실행"""
    console.print("[bold green]Streamlit 웹 인터페이스를 시작합니다...[/bold green]")
    console.print("브라우저에서 http://localhost:8501 에 접속하세요")
    
    # Streamlit 앱 실행
    os.system("streamlit run " + __file__ + " -- web-interface")

@app.command()
def test():
    """RAG 시스템 테스트"""
    console.print("[bold blue]RAG 시스템 테스트 시작[/bold blue]")
    
    try:
        # 테스트용 RAG 챗봇 생성
        chatbot = RAGChatbot()
        
        # 컴포넌트 테스트
        console.print("✓ RAG 챗봇 초기화 완료")
        
        # 더미 텍스트로 테스트
        test_text = """
        인공지능(AI)은 인간의 지능을 모방하는 기술입니다.
        머신러닝은 AI의 한 분야로, 데이터로부터 패턴을 학습합니다.
        딥러닝은 신경망을 사용하는 머신러닝의 하위 분야입니다.
        """
        
        # 문서 처리 테스트
        chunks = chatbot.doc_processor._smart_chunking(test_text, "test.txt")
        console.print(f"✓ 문서 청킹 테스트 완료 - 청크 수: {len(chunks)}")
        
        # 임베딩 테스트
        chunks_with_embeddings = chatbot.embedding_manager.generate_embeddings(chunks)
        console.print("✓ 임베딩 생성 테스트 완료")
        
        # 벡터 DB 테스트
        chatbot.vector_db.add_chunks(chunks_with_embeddings)
        console.print("✓ 벡터 DB 저장 테스트 완료")
        
        # 검색 테스트
        query_embedding = chatbot.embedding_manager.generate_query_embedding("AI란 무엇인가요?")
        search_results = chatbot.vector_db.similarity_search(query_embedding, k=2)
        console.print(f"✓ 유사도 검색 테스트 완료 - 검색 결과: {len(search_results)}개")
        
        # RAG 응답 생성 테스트
        response = chatbot.rag_generator.generate_response("AI란 무엇인가요?", search_results)
        console.print("✓ RAG 응답 생성 테스트 완료")
        
        console.print(f"\n[bold green]모든 테스트 통과![/bold green]")
        console.print(f"답변: {response.answer[:100]}...")
        
    except Exception as e:
        console.print(f"[bold red]테스트 실패:[/bold red] {str(e)}")

if __name__ == "__main__":
    # Streamlit 웹 인터페이스 모드 확인
    if len(sys.argv) > 1 and sys.argv[1] == "web-interface":
        streamlit_app()
    else:
        # CLI 모드
        app()