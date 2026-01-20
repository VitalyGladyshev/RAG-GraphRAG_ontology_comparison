# Реализация и оценка конвейера RAG на данных CAD домена. Сравнение с GraphRAG

## Импорты


```python
import os
import sys
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, asdict, field
import pickle
import shutil

# Обработка данных
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Векторные базы данных
import chromadb

# LangChain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Модели и ML
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

# Jupyter
from IPython.display import display, HTML
from tqdm.notebook import tqdm

# OpenAI для custom endpoints
import openai
from openai import AsyncOpenAI

import asyncio
import nest_asyncio
nest_asyncio.apply()
print("✅ nest_asyncio успешно применен для работы с асинхронным кодом в Jupyter")

# RAGAS 0.4+
try:
    from ragas.llms import llm_factory
    from ragas.embeddings.base import embedding_factory
    from ragas.metrics.collections import (
        Faithfulness,              # ✅ Соответствие ответа контексту
        AnswerRelevancy,           # ✅ Релевантность ответа запросу (требует embeddings)
        ContextPrecision,          # ✅ Точность контекста (требует reference)
        ContextRecall,             # ✅ Полнота контекста (требует reference)
        ContextRelevance,          # ✅ Релевантность контекста
        ResponseGroundedness,      # ✅ Обоснованность ответа
        AnswerAccuracy,            # ✅ Точность ответа (требует reference)
    )
    RAGAS_AVAILABLE = True
    print("✅ RAGAS 0.4+ успешно импортирована (7 метрик)")
except ImportError as e:
    RAGAS_AVAILABLE = False
    print(f"⚠️ RAGAS не установлена: {e}")
    print("Установите: pip install ragas --upgrade")

import warnings
warnings.filterwarnings('ignore')
```

    ✅ nest_asyncio успешно применен для работы с асинхронным кодом в Jupyter
    ✅ RAGAS 0.4+ успешно импортирована (7 метрик)
    


```python
%matplotlib inline
```

## Классы конфигураций


```python
class JupyterConfig:
    """Конфигурация для работы в Jupyter Notebook"""
    # Настройки отображения
    DISPLAY_MAX_CHARS = 200
    DISPLAY_TABLE_ROWS = 15
    USE_INTERACTIVE_WIDGETS = True
    
    # Настройки выполнения
    AUTO_SAVE_RESULTS = True
    RESULTS_DIR = "notebook_results"
    
    # Цвета для вывода
    SUCCESS_COLOR = "#4CAF50"
    WARNING_COLOR = "#FFA726"
    ERROR_COLOR = "#EF5350"
    INFO_COLOR = "#2196F3"
    
    @classmethod
    def setup_directories(cls):
        """Создает необходимые директории для результатов."""
        Path(cls.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        print(f"📁 Директория для результатов создана: {cls.RESULTS_DIR}")
```


```python
class Config:
    """Конфигурация RAG-пайплайна."""
    # Пути
    DEFAULT_DATA_DIR = "data/chunks"
    DEFAULT_DB_DIR = "vector_db"
    DEFAULT_MODEL_CACHE = "models/cache"
    DEFAULT_LOGS_DIR = "logs"
    RESULTS_DIR = "notebook_results"
    
    # Модель эмбеддингов
    EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
    EMBEDDING_DIMENSION = 1024
    MAX_SEQUENCE_LENGTH = 512
    
    # Чанкинг
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # Векторная база
    COLLECTION_NAME = "document_chunks"
    DISTANCE_METRIC = "cosine"
    
    # Производительность
    BATCH_SIZE = 32
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_FP16 = True if torch.cuda.is_available() else False
    
    # Поиск
    DEFAULT_TOP_K = 5
    SIMILARITY_THRESHOLD = 0.7
    
    # LLM конфигурация
    LLM_API_BASE = "https://api.vsegpt.ru/v1"
    LLM_API_KEY = "sk-or-vv-eea69f0496482e74e9e5a4cccb233dd0128e98c305328e0a56a1e638f5613b99"
    
    # Модели
    GENERATION_MODEL = "google/gemma-3-27b-it"
    EVALUATION_MODEL = "openai/gpt-5-mini"
    
    # Параметры генерации
    GENERATION_TEMPERATURE = 0.7
    GENERATION_MAX_TOKENS = 1024
    
    # Параметры оценки
    EVALUATION_TEMPERATURE = 0.0  # Для оценки нужна детерминированность
    EVALUATION_MAX_TOKENS = 512
    
    # Логирование
    LOG_LEVEL = "INFO"
    
    @classmethod
    def setup_directories(cls):
        """Создает необходимые директории."""
        for directory in [cls.DEFAULT_DATA_DIR, cls.DEFAULT_DB_DIR, 
                         cls.DEFAULT_MODEL_CACHE, cls.DEFAULT_LOGS_DIR,
                         cls.RESULTS_DIR]:
            Path(directory).mkdir(parents=True, exist_ok=True)
```

## Классы данных


```python
@dataclass
class SearchResult:
    """Результат поиска в векторной базе."""
    id: str
    content: str
    similarity_score: float
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
```


```python
@dataclass
class RAGEvaluationResult:
    """Результат оценки одного примера."""
    query: str
    answer: str
    context: List[str]
    ground_truth: Optional[str] = None
    
    # Метрики
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    context_relevance: Optional[float] = None
    response_groundedness: Optional[float] = None
    answer_accuracy: Optional[float] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def get_average_score(self) -> float:
        """Вычисляет средний балл по всем доступным метрикам."""
        scores = [
            self.faithfulness,
            self.answer_relevancy,
            self.context_precision,
            self.context_recall,
            self.context_relevance,
            self.response_groundedness,
            self.answer_accuracy,
        ]
        valid_scores = [s for s in scores if s is not None]
        return float(np.mean(valid_scores)) if valid_scores else 0.0
    
    def to_dict(self) -> Dict:
        """Конвертирует результат в словарь."""
        return asdict(self)
```

## LLM клиент


```python
class UniversalLLMClient:
    """
    Универсальный клиент для работы с LLM через OpenAI-совместимое API.
    Поддерживает custom endpoints (например, vsegpt.ru).
    """
    
    def __init__(self,
                 model: str,
                 api_key: str,
                 api_base: str = "https://api.openai.com/v1",
                 temperature: float = 0.7,
                 max_tokens: int = 1024,
                 timeout: int = 120):
        """
        Args:
            model: Название модели (например, "openai/gpt-4o-mini")
            api_key: API ключ
            api_base: Base URL API
            temperature: Температура генерации
            max_tokens: Максимальное количество токенов
            timeout: Таймаут запроса в секундах
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        
        # Создаем синхронный OpenAI клиент
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout,
            max_retries=3
        )
        
        # Создаем асинхронный OpenAI клиент для RAGAS
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout,
            max_retries=3,
            # Устанавливаем дефолтные параметры для всех запросов
            default_headers={
                "X-Max-Tokens": str(max_tokens)
            }
        )
        
        self.logger.info(f"✅ LLM клиент инициализирован: {model} @ {api_base}")
        self.logger.info(f"⚙️  max_tokens={max_tokens}, temperature={temperature}")
    
    def generate(self,
                prompt: str,
                system_prompt: str = None,
                temperature: float = None,
                max_tokens: int = None) -> str:
        """
        Генерирует текст на основе промпта.
        
        Args:
            prompt: Пользовательский промпт
            system_prompt: Системный промпт
            temperature: Температура (переопределяет дефолтную)
            max_tokens: Максимум токенов (переопределяет дефолтный)
        
        Returns:
            Сгенерированный текст
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens
            )
            
            generated_text = response.choices[0].message.content
            
            self.logger.debug(f"✅ Генерация завершена ({len(generated_text)} символов)")
            return generated_text.strip()
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка генерации: {e}")
            raise
    
    def get_ragas_llm(self):
        """
        Возвращает RAGAS-совместимый LLM объект.
        """
        try:
            # Создаем обертку с нужными параметрами
            ragas_llm = llm_factory(
                model=self.model,
                client=self.async_client,
                # Передаем параметры генерации
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            self.logger.info(f"✅ RAGAS LLM создан с max_tokens={self.max_tokens}")
            return ragas_llm
            
        except Exception as e:
            self.logger.warning(f"⚠️ Ошибка при создании RAGAS LLM: {e}")
            # Fallback к базовому варианту
            return llm_factory(
                model=self.model,
                client=self.async_client
            )
```


```python
class LLMGenerator:
    """
    Генератор ответов для RAG системы.
    """
    
    def __init__(self,
                 llm_client: UniversalLLMClient,
                 system_prompt: str = None):
        """
        Args:
            llm_client: Клиент LLM
            system_prompt: Системный промпт по умолчанию
        """
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)
        
        if system_prompt is None:
            self.system_prompt = """Ты - полезный ассистент, который отвечает на вопросы пользователя на основе предоставленного контекста.

КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:
1. Используй ТОЛЬКО информацию из предоставленного контекста
2. Если в контексте нет ответа на вопрос, честно скажи об этом
3. НЕ добавляй информацию, которой нет в контексте
4. Отвечай четко, по делу и структурированно
5. Если уместно, цитируй факты из контекста
6. Отвечай на том же языке, что и вопрос пользователя"""
        else:
            self.system_prompt = system_prompt
    
    def generate_answer(self,
                       query: str,
                       contexts: List[str],
                       custom_system_prompt: str = None) -> str:
        """
        Генерирует ответ на основе запроса и контекстов.
        
        Args:
            query: Вопрос пользователя
            contexts: Список релевантных контекстов
            custom_system_prompt: Кастомный системный промпт
        
        Returns:
            Сгенерированный ответ
        """
        if not contexts:
            return "Извините, не удалось найти релевантную информацию для ответа на ваш вопрос."
        
        # Объединяем контексты
        combined_context = "\n\n".join([
            f"[Контекст {i+1}]:\n{ctx}" 
            for i, ctx in enumerate(contexts[:5])  # Берем топ-5
        ])
        
        # Формируем промпт
        user_prompt = f"""Контекст для ответа:
{combined_context}

Вопрос пользователя:
{query}

Ответ на основе контекста:"""

        try:
            answer = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=custom_system_prompt or self.system_prompt
            )
            
            self.logger.info(f"✅ Ответ сгенерирован ({len(answer)} символов)")
            return answer
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка генерации ответа: {e}")
            return f"Ошибка при генерации ответа: {str(e)}"
    
    def batch_generate(self,
                      queries: List[str],
                      contexts_list: List[List[str]]) -> List[str]:
        """
        Генерирует ответы для батча запросов.
        
        Args:
            queries: Список вопросов
            contexts_list: Список списков контекстов
        
        Returns:
            Список сгенерированных ответов
        """
        answers = []
        for query, contexts in tqdm(zip(queries, contexts_list), 
                                    total=len(queries),
                                    desc="Генерация ответов"):
            answer = self.generate_answer(query, contexts)
            answers.append(answer)
        
        return answers
```

### Класс метрик RAG (RAGAS)


```python
class UniversalRAGEvaluator:
    """
    🎯 УНИВЕРСАЛЬНЫЙ оценщик RAG систем с RAGAS 0.4+
    
    Поддерживает 7 проверенных метрик:
    ✅ faithfulness - Соответствие ответа контексту
    ✅ answer_relevancy - Релевантность ответа запросу (требует embeddings)
    ✅ context_precision - Точность контекста (требует reference)
    ✅ context_recall - Полнота контекста (требует reference)
    ✅ context_relevance - Релевантность контекста
    ✅ response_groundedness - Обоснованность ответа
    ✅ answer_accuracy - Точность ответа (требует reference)
    """
    
    # ✅ ПРОВЕРЕННЫЕ метрики для RAGAS 0.4+
    AVAILABLE_METRICS = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "context_relevance",
        "response_groundedness",
        "answer_accuracy",
    ]
    
    def __init__(self, 
                 judge_llm_client: UniversalLLMClient,
                 embedding_model: str = "emb-openai/text-embedding-3-large",
                 metrics: List[str] = None,
                 enable_timing: bool = True):
        """
        Args:
            judge_llm_client: LLM клиент для оценки (Judge)
            embedding_model: Модель для embeddings (для answer_relevancy)
            metrics: Список метрик для оценки (по умолчанию базовые)
        """
        self.judge_llm_client = judge_llm_client
        self.embedding_model = embedding_model
        self.evaluation_results: List[RAGEvaluationResult] = []
        self.logger = logging.getLogger(__name__)
        
        if not RAGAS_AVAILABLE:
            raise ImportError("RAGAS не установлена. Установите: pip install ragas --upgrade")
        
        # Базовые метрики (работают без ground truth)
        self.basic_metrics = [
            "faithfulness",
            "context_relevance",
            "response_groundedness",
        ]
        
        # Метрики, требующие embeddings
        self.embedding_metrics = [
            "answer_relevancy",
        ]
        
        # Метрики, требующие ground truth
        self.ground_truth_metrics = [
            "context_precision",
            "context_recall",
            "answer_accuracy",
        ]
        
        # По умолчанию используем только базовые метрики
        self.metrics_to_use = metrics or self.basic_metrics
        self._validate_metrics()
        
        # Создаем RAGAS LLM и embeddings
        self.ragas_llm = self.judge_llm_client.get_ragas_llm()
        
        # Создаем embeddings только если нужны
        self.ragas_embeddings = None
        if any(m in self.metrics_to_use for m in self.embedding_metrics):
            try:
                self.ragas_embeddings = embedding_factory(
                    "openai",
                    model=embedding_model,
                    client=self.judge_llm_client.async_client
                )
                self.logger.info(f"✅ Embeddings инициализированы: {embedding_model}")
            except Exception as e:
                self.logger.warning(f"⚠️ Ошибка инициализации embeddings: {e}")
                # Убираем метрики, требующие embeddings
                self.metrics_to_use = [m for m in self.metrics_to_use if m not in self.embedding_metrics]
        
        # Инициализируем метрики
        self.metric_scorers = {}
        self._initialize_metrics()

        # Отслеживание времени
        self.enable_timing = enable_timing
        self.metric_timings = defaultdict(list)  # {metric_name: [time1, time2, ...]}
        
        self.logger.info(f"✅ Универсальный RAG Evaluator инициализирован (RAGAS 0.4+)")
        self.logger.info(f"📊 Метрики: {', '.join(self.metrics_to_use)}")

        self._nest_asyncio_applied = False
        # Автоматически применяем nest_asyncio для Jupyter
        if self._is_running_in_jupyter():
            self._apply_nest_asyncio()

    def _is_running_in_jupyter(self) -> bool:
        """Проверяет Jupyter/IPython окружение."""
        try:
            from IPython import get_ipython
            ipython = get_ipython()
            return ipython is not None and 'IPKernelApp' in getattr(ipython, 'config', {})
        except (ImportError, AttributeError):
            return False
    
    def _apply_nest_asyncio(self):
        """Применяет nest_asyncio один раз."""
        if not self._nest_asyncio_applied:
            try:
                import nest_asyncio
                nest_asyncio.apply()
                self._nest_asyncio_applied = True
                self.logger.info("✅ nest_asyncio применен для Jupyter")
            except ImportError:
                self.logger.warning("⚠️ Установите nest_asyncio: pip install nest_asyncio")
    
    def _validate_metrics(self):
        """Проверяет валидность выбранных метрик."""
        invalid_metrics = set(self.metrics_to_use) - set(self.AVAILABLE_METRICS)
        if invalid_metrics:
            raise ValueError(f"Неизвестные метрики: {invalid_metrics}")
    
    def _initialize_metrics(self):
        """Инициализирует объекты метрик с LLM/Embeddings."""
        metric_classes = {
            "faithfulness": Faithfulness,
            "answer_relevancy": AnswerRelevancy,
            "context_precision": ContextPrecision,
            "context_recall": ContextRecall,
            "context_relevance": ContextRelevance,
            "response_groundedness": ResponseGroundedness,
            "answer_accuracy": AnswerAccuracy,
        }
        
        for metric_name in self.metrics_to_use:
            if metric_name not in metric_classes:
                continue
            
            try:
                metric_class = metric_classes[metric_name]
                
                # Инициализируем метрику с нужными параметрами
                if metric_name in self.embedding_metrics:
                    if self.ragas_embeddings is None:
                        self.logger.warning(f"⚠️ Пропущена метрика {metric_name} (нет embeddings)")
                        continue
                    scorer = metric_class(llm=self.ragas_llm, embeddings=self.ragas_embeddings)
                else:
                    scorer = metric_class(llm=self.ragas_llm)
                
                self.metric_scorers[metric_name] = scorer
                self.logger.info(f"✅ Метрика {metric_name} инициализирована")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Ошибка инициализации метрики {metric_name}: {e}")
    
    async def _evaluate_single_async(self, 
                                     query: str, 
                                     answer: str, 
                                     contexts: List[str],
                                     ground_truth: str = None) -> Dict[str, float]:
        """
        Асинхронная оценка одного примера.
        
        Args:
            query: Вопрос
            answer: Ответ
            contexts: Контексты
            ground_truth: Правильный ответ (опционально)
        
        Returns:
            Словарь с метриками
        """
        scores = {}
        
        for metric_name, scorer in self.metric_scorers.items():
            start_time = time.time()
            
            try:
                # Подготовка параметров в зависимости от метрики
                if metric_name == "faithfulness":
                    result = await scorer.ascore(
                        user_input=query,
                        response=answer,
                        retrieved_contexts=contexts
                    )
                
                elif metric_name == "answer_relevancy":
                    result = await scorer.ascore(
                        user_input=query,
                        response=answer
                    )
                
                elif metric_name == "context_precision":
                    if not ground_truth:
                        continue
                    result = await scorer.ascore(
                        user_input=query,
                        reference=ground_truth,
                        retrieved_contexts=contexts
                    )
                
                elif metric_name == "context_recall":
                    if not ground_truth:
                        continue
                    result = await scorer.ascore(
                        user_input=query,
                        retrieved_contexts=contexts,
                        reference=ground_truth
                    )
                
                elif metric_name == "context_relevance":
                    result = await scorer.ascore(
                        user_input=query,
                        retrieved_contexts=contexts
                    )
                
                elif metric_name == "response_groundedness":
                    result = await scorer.ascore(
                        response=answer,
                        retrieved_contexts=contexts
                    )
                
                elif metric_name == "answer_accuracy":
                    if not ground_truth:
                        continue
                    result = await scorer.ascore(
                        user_input=query,
                        response=answer,
                        reference=ground_truth
                    )
                
                else:
                    continue
                
                scores[metric_name] = float(result.value)
                
            except Exception as e:
                self.logger.warning(f"⚠️ Ошибка при оценке {metric_name}: {e}")
                scores[metric_name] = None

            finally:
                # Сохраняем время выполнения
                if self.enable_timing:
                    elapsed_time = time.time() - start_time
                    self.metric_timings[metric_name].append(elapsed_time)
        
        return scores
    
    def _evaluate_single(self, query: str, answer: str, contexts: List[str],
                        ground_truth: str = None) -> Dict[str, float]:
        """
        Синхронная обертка для асинхронной оценки.
        """
        try:
            # Есть запущенный loop - используем его
            loop = asyncio.get_running_loop()
            if not self._nest_asyncio_applied:
                self._apply_nest_asyncio()
            return loop.run_until_complete(
                self._evaluate_single_async(query, answer, contexts, ground_truth)
            )
        except RuntimeError:
            # Нет loop - создаем через asyncio.run()
            return asyncio.run(
                self._evaluate_single_async(query, answer, contexts, ground_truth)
            )

    def evaluate(self, 
                queries: List[str], 
                answers: List[str], 
                contexts: List[List[str]],
                ground_truths: List[str] = None,
                show_progress: bool = True) -> Dict[str, float]:
        """
        Выполняет оценку RAG системы.
        
        Args:
            queries: Список вопросов
            answers: Список ответов
            contexts: Список списков контекстов
            ground_truths: Список правильных ответов (опционально)
            show_progress: Показывать прогресс
        
        Returns:
            Словарь с агрегированными метриками
        """
        if len(queries) != len(answers) or len(queries) != len(contexts):
            raise ValueError("Длины queries, answers и contexts должны совпадать")
        
        if ground_truths is None:
            ground_truths = [None] * len(queries)
        elif len(ground_truths) != len(queries):
            raise ValueError("Длина ground_truths должна совпадать с queries")
        
        has_ground_truth = any(gt and gt.strip() for gt in ground_truths if gt)
        
        self.logger.info("🚀 Запуск оценки RAGAS 0.4+...")
        self.logger.info(f"⚙️  Используется модель: {self.judge_llm_client.model}")
        self.logger.info(f"📊 Активных метрик: {len(self.metric_scorers)}")
        self.logger.info(f"🎯 Ground truth: {'Да' if has_ground_truth else 'Нет'}")
        self.logger.info(f"⏳ Оценка {len(queries)} примеров...")
        self.logger.info("⚠️  Это может занять несколько минут (LLM вызовы)")
        
        # Очищаем предыдущие результаты
        self.evaluation_results.clear()
        
        # Оцениваем каждый пример
        iterator = tqdm(zip(queries, answers, contexts, ground_truths), 
                       total=len(queries),
                       desc="Оценка примеров") if show_progress else zip(queries, answers, contexts, ground_truths)
        
        all_scores = {metric: [] for metric in self.metric_scorers.keys()}
        
        for query, answer, context, ground_truth in iterator:
            # Создаем результат
            result = RAGEvaluationResult(
                query=query,
                answer=answer,
                context=context,
                ground_truth=ground_truth
            )
            
            # Оцениваем
            try:
                scores = self._evaluate_single(query, answer, context, ground_truth)
                
                # Сохраняем метрики
                for metric_name, score in scores.items():
                    setattr(result, metric_name, score)
                    if score is not None:
                        all_scores[metric_name].append(score)
                
            except Exception as e:
                self.logger.error(f"❌ Ошибка при оценке примера: {e}")
            
            self.evaluation_results.append(result)
        
        # Вычисляем агрегированные метрики
        aggregated = self._compute_aggregated_metrics(all_scores)
        
        self.logger.info("✅ Оценка RAGAS завершена успешно!")
        return aggregated
    
    def _compute_aggregated_metrics(self, all_scores: Dict[str, List[float]]) -> Dict[str, float]:
        """Вычисляет агрегированные метрики."""
        aggregated = {}
        
        for metric_name, scores in all_scores.items():
            if scores:
                aggregated[f"{metric_name}_mean"] = float(np.mean(scores))
                aggregated[f"{metric_name}_std"] = float(np.std(scores))
                aggregated[f"{metric_name}_min"] = float(np.min(scores))
                aggregated[f"{metric_name}_max"] = float(np.max(scores))
                aggregated[f"{metric_name}_count"] = len(scores)
        
        mean_values = [v for k, v in aggregated.items() if k.endswith('_mean')]
        if mean_values:
            aggregated['overall_mean'] = float(np.mean(mean_values))
        
        return aggregated
    
    def get_detailed_results(self) -> pd.DataFrame:
        """Возвращает детальные результаты в виде DataFrame."""
        data = []
        for result in self.evaluation_results:
            row = {
                'query': result.query[:100] + '...' if len(result.query) > 100 else result.query,
                'answer': result.answer[:100] + '...' if len(result.answer) > 100 else result.answer,
            }
            
            for metric_name in self.AVAILABLE_METRICS:
                value = getattr(result, metric_name, None)
                row[metric_name] = value
            
            row['average_score'] = result.get_average_score()
            data.append(row)
        
        return pd.DataFrame(data)
    
    def save_results(self, filepath: str = None) -> Path:
        """Сохраняет результаты оценки."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = Path("results") / f"ragas_evaluation_{timestamp}.json"
        else:
            filepath = Path(filepath)
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        results_data = [result.to_dict() for result in self.evaluation_results]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"💾 Результаты сохранены: {filepath}")
        return filepath
    
    def display_results(self, max_examples: int = 5):
        """Отображает результаты оценки в интерактивном формате."""
        if not self.evaluation_results:
            print("⚠️ Нет результатов для отображения")
            return
        
        df = self.get_detailed_results()
        
        # Цвета для метрик
        metric_colors = {
            "faithfulness": "#4CAF50",
            "answer_relevancy": "#2196F3",
            "context_precision": "#FFA726",
            "context_recall": "#9C27B0",
            "context_relevance": "#FF5722",
            "response_groundedness": "#795548",
            "answer_accuracy": "#F44336",
        }
        
        html = f"""
        <div style="border: 2px solid #2196F3; border-radius: 12px; padding: 20px; margin: 15px 0; background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);">
            <h2 style="color: #1565C0; margin-top: 0;">📊 Результаты оценки RAGAS 0.4+</h2>
            <p style="color: #666;"><strong>Всего примеров:</strong> {len(self.evaluation_results)} | 
               <strong>Метрик:</strong> {len([m for m in self.AVAILABLE_METRICS if m in df.columns and df[m].notna().any()])}</p>
        """
        
        html += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0;">'
        
        for metric_name in self.AVAILABLE_METRICS:
            if metric_name in df.columns:
                values = df[metric_name].dropna()
                if len(values) > 0:
                    mean_val = values.mean()
                    color = metric_colors.get(metric_name, "#607D8B")
                    
                    html += f"""
                    <div style="text-align: center; padding: 15px; background: white; border-radius: 8px; border: 2px solid {color};">
                        <div style="font-weight: bold; color: {color}; margin-bottom: 8px; text-transform: capitalize;">
                            {metric_name.replace('_', ' ')}
                        </div>
                        <div style="font-size: 32px; font-weight: bold; color: {color};">{mean_val:.3f}</div>
                        <div style="background-color: {color}22; height: 10px; border-radius: 5px; margin-top: 8px;">
                            <div style="background-color: {color}; height: 100%; width: {min(mean_val*100, 100)}%; border-radius: 5px;"></div>
                        </div>
                        <div style="font-size: 12px; color: #666; margin-top: 5px;">
                            Min: {values.min():.3f} | Max: {values.max():.3f}
                        </div>
                    </div>
                    """
        
        html += '</div>'
        
        # Детальные примеры
        html += f"""
        <h3 style="color: #1565C0; margin-top: 25px;">📋 Детальные результаты (топ-{max_examples}):</h3>
        """
        
        for i, result in enumerate(self.evaluation_results[:max_examples]):
            avg_score = result.get_average_score()
            border_color = "#4CAF50" if avg_score > 0.7 else "#FFA726" if avg_score > 0.5 else "#F44336"
            
            html += f"""
            <div style="background: white; padding: 15px; margin: 12px 0; border-radius: 8px; border-left: 5px solid {border_color};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <strong style="color: #1a237e; font-size: 1.1em;">Пример #{i+1}</strong>
                    <span style="background: {border_color}; color: white; padding: 4px 12px; border-radius: 15px; font-weight: bold;">
                        Средний балл: {avg_score:.3f}
                    </span>
                </div>
                
                <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 8px 0;">
                    <strong style="color: #666;">❓ Вопрос:</strong><br/>
                    <span style="color: #1a237e;">{result.query[:200]}{'...' if len(result.query) > 200 else ''}</span>
                </div>
                
                <div style="background: #e8f5e9; padding: 10px; border-radius: 5px; margin: 8px 0;">
                    <strong style="color: #666;">✅ Ответ:</strong><br/>
                    <span style="color: #2e7d32;">{result.answer[:300]}{'...' if len(result.answer) > 300 else ''}</span>
                </div>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; margin-top: 10px;">
            """
            
            for metric_name in self.AVAILABLE_METRICS:
                value = getattr(result, metric_name, None)
                if value is not None:
                    color = metric_colors.get(metric_name, "#607D8B")
                    html += f"""
                    <div style="background: {color}15; padding: 6px 10px; border-radius: 5px; text-align: center; border: 1px solid {color}50;">
                        <div style="font-size: 11px; color: #666; text-transform: uppercase;">{metric_name.replace('_', ' ')}</div>
                        <div style="font-size: 18px; font-weight: bold; color: {color};">{value:.3f}</div>
                    </div>
                    """
            
            html += """
                </div>
            </div>
            """
        
        html += "</div>"
        
        display(HTML(html))
        
        print("\n📈 Статистическая сводка:")
        display(df.describe().style.background_gradient(cmap='viridis'))

    def get_timing_statistics(self) -> pd.DataFrame:
        """
        Возвращает статистику по времени выполнения метрик.
        """
        if not self.metric_timings:
            return pd.DataFrame()
        
        stats = []
        for metric_name, times in self.metric_timings.items():
            if times:
                stats.append({
                    'metric': metric_name,
                    'count': len(times),
                    'total_time_sec': sum(times),
                    'avg_time_sec': np.mean(times),
                    'min_time_sec': np.min(times),
                    'max_time_sec': np.max(times),
                    'std_time_sec': np.std(times),
                })
        
        df = pd.DataFrame(stats)
        if not df.empty:
            df = df.sort_values('avg_time_sec', ascending=False)
        
        return df
    
    def display_timing_analysis(self):
        """
        Отображает анализ времени выполнения метрик.
        """
        df = self.get_timing_statistics()
        
        if df.empty:
            print("⚠️ Нет данных о времени выполнения")
            return
        
        total_time = df['total_time_sec'].sum()
        
        html = f"""
        <div style="border: 2px solid #FF9800; border-radius: 12px; padding: 20px; margin: 15px 0; 
                    background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);">
            <h2 style="color: #E65100; margin-top: 0;">⏱️ Анализ времени выполнения метрик</h2>
            <p style="color: #666;"><strong>Общее время:</strong> {total_time:.1f} сек ({total_time/60:.1f} мин)</p>
            
            <table style="width: 100%; border-collapse: collapse; background: white; margin-top: 15px;">
                <thead>
                    <tr style="background: #FF9800; color: white;">
                        <th style="padding: 10px; text-align: left;">Метрика</th>
                        <th style="padding: 10px; text-align: center;">Количество</th>
                        <th style="padding: 10px; text-align: right;">Среднее (сек)</th>
                        <th style="padding: 10px; text-align: right;">Общее (сек)</th>
                        <th style="padding: 10px; text-align: right;">% времени</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for idx, row in df.iterrows():
            percentage = (row['total_time_sec'] / total_time) * 100
            avg_time = row['avg_time_sec']
            
            # Цвет в зависимости от времени
            if avg_time > 10:
                bg_color = "#FFCDD2"  # Красный (медленно)
            elif avg_time > 5:
                bg_color = "#FFE0B2"  # Оранжевый (средне)
            else:
                bg_color = "#C8E6C9"  # Зеленый (быстро)
            
            html += f"""
                <tr style="background: {bg_color};">
                    <td style="padding: 8px; font-weight: bold;">{row['metric']}</td>
                    <td style="padding: 8px; text-align: center;">{row['count']}</td>
                    <td style="padding: 8px; text-align: right;">{row['avg_time_sec']:.2f}</td>
                    <td style="padding: 8px; text-align: right;">{row['total_time_sec']:.1f}</td>
                    <td style="padding: 8px; text-align: right;">{percentage:.1f}%</td>
                </tr>
            """
        
        html += """
                </tbody>
            </table>
            
            <div style="margin-top: 15px; padding: 10px; background: #FFF9C4; border-radius: 5px;">
                <strong>💡 Рекомендации:</strong>
                <ul style="margin: 5px 0;">
        """
        
        # Находим самые медленные метрики
        slow_metrics = df[df['avg_time_sec'] > df['avg_time_sec'].median()]['metric'].tolist()
        
        if slow_metrics:
            html += f"<li>🐌 <strong>Медленные метрики:</strong> {', '.join(slow_metrics)}</li>"
        
        # Метрики с низкой информативностью
        if 'faithfulness' in df['metric'].values and 'response_groundedness' in df['metric'].values:
            html += """
                <li>⚠️ <strong>faithfulness</strong> и <strong>response_groundedness</strong> 
                показывают одинаковые значения (1.0) - рассмотрите исключение одной из них</li>
            """
        
        html += """
                </ul>
            </div>
        </div>
        """
        
        display(HTML(html))
        
        # Также выводим DataFrame
        print("\n📊 Детальная таблица:")
        display(df.style.background_gradient(cmap='YlOrRd', subset=['avg_time_sec']))
    
    def get_recommended_metrics(self, 
                               max_avg_time: float = 10.0,
                               exclude_redundant: bool = True) -> List[str]:
        """
        Возвращает рекомендуемые метрики на основе анализа производительности.
        
        Args:
            max_avg_time: Максимальное среднее время выполнения (сек)
            exclude_redundant: Исключить избыточные метрики
        
        Returns:
            Список рекомендуемых метрик
        """
        df = self.get_timing_statistics()
        
        if df.empty:
            return self.metrics_to_use
        
        # Фильтруем по времени
        fast_metrics = df[df['avg_time_sec'] <= max_avg_time]['metric'].tolist()
        
        # Исключаем избыточные
        if exclude_redundant:
            # Проверяем faithfulness и response_groundedness
            results_df = self.get_detailed_results()
            
            if 'faithfulness' in results_df.columns and 'response_groundedness' in results_df.columns:
                faith_vals = results_df['faithfulness'].dropna()
                ground_vals = results_df['response_groundedness'].dropna()
                
                # Если значения почти одинаковые, исключаем одну
                if len(faith_vals) > 0 and len(ground_vals) > 0:
                    correlation = faith_vals.corr(ground_vals)
                    
                    if correlation > 0.95:  # Очень сильная корреляция
                        # Оставляем более быструю метрику
                        faith_time = df[df['metric'] == 'faithfulness']['avg_time_sec'].values
                        ground_time = df[df['metric'] == 'response_groundedness']['avg_time_sec'].values
                        
                        if len(faith_time) > 0 and len(ground_time) > 0:
                            if faith_time[0] > ground_time[0]:
                                fast_metrics = [m for m in fast_metrics if m != 'faithfulness']
                                self.logger.info("ℹ️ Исключена faithfulness (избыточна)")
                            else:
                                fast_metrics = [m for m in fast_metrics if m != 'response_groundedness']
                                self.logger.info("ℹ️ Исключена response_groundedness (избыточна)")
        
        self.logger.info(f"✅ Рекомендуемые метрики: {', '.join(fast_metrics)}")
        return fast_metrics
```


```python
class TextProcessor:
    """Обработчик текста для создания чанков."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Инициализация процессора текста.
        
        Args:
            chunk_size: Размер чанка в символах
            chunk_overlap: Перекрытие между чанками
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.logger = logging.getLogger(__name__)
    
    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Разделяет текст на чанки.
        
        Args:
            text: Исходный текст
            meta Базовые метаданные для всех чанков
        
        Returns:
            Список словарей с чанками и метаданными
        """
        if metadata is None:
            metadata = {}
        
        # Разделяем текст
        chunks = self.text_splitter.split_text(text)
        
        # Создаем метаданные для каждого чанка
        result = []
        for i, chunk in enumerate(chunks):
            chunk_meta = metadata.copy()
            chunk_meta.update({
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk),
                "is_truncated": len(chunk) >= self.chunk_size
            })
            result.append({
                "text": chunk,
                "metadata": chunk_meta
            })
        
        self.logger.info(f"Разделено на {len(chunks)} чанков")
        return result
    
    def process_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Обрабатывает один файл.
        
        Args:
            file_path: Путь к файлу
        
        Returns:
            Список чанков из файла
        """
        try:
            # Пробуем разные кодировки
            encodings = ['utf-8', 'cp1251', 'latin-1']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read().strip()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                self.logger.error(f"Не удалось прочитать файл {file_path}")
                return []
            
            if not content:
                self.logger.warning(f"Файл {file_path} пустой")
                return []
            
            # Создаем базовые метаданные
            base_metadata = {
                "source_file": file_path.name,
                "file_path": str(file_path),
                "file_size": len(content),
                "encoding": encoding
            }
            
            # Разделяем на чанки
            chunks = self.split_text(content, base_metadata)
            return chunks
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки файла {file_path}: {e}")
            return []
```


```python
class DataLoader:
    """Загрузчик текстовых файлов."""
    
    def __init__(self, data_dir: str = None):
        """
        Инициализация загрузчика данных.
        
        Args:
            data_dir: Директория с текстовыми файлами
        """
        self.data_dir = Path(data_dir or Config.DEFAULT_DATA_DIR)
        self.logger = logging.getLogger(__name__)
        
        if not self.data_dir.exists():
            self.logger.warning(f"Директория {self.data_dir} не существует")
            self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_files_to_dataframe(self,
                               file_pattern: str = "*.txt",
                               max_files: int = None,
                               process_chunks: bool = True) -> pd.DataFrame:
        """
        Загружает файлы в DataFrame.
        
        Args:
            file_pattern: Шаблон для поиска файлов
            max_files: Максимальное количество файлов для загрузки
            process_chunks: Разделять ли файлы на чанки
        
        Returns:
            DataFrame с текстами и метаданными
        """
        # Находим файлы
        files = list(self.data_dir.glob(file_pattern))
        if max_files:
            files = files[:max_files]
        
        if not files:
            self.logger.warning(f"Файлы по шаблону {file_pattern} не найдены")
            return pd.DataFrame()
        
        self.logger.info(f"Найдено {len(files)} файлов")
        
        # Обрабатываем файлы
        processor = TextProcessor()
        all_chunks = []
        
        for file_path in tqdm(files, desc="Загрузка файлов"):
            if process_chunks:
                chunks = processor.process_file(file_path)
                all_chunks.extend(chunks)
            else:
                # Загружаем файлы без разделения на чанки
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    all_chunks.append({
                        "text": content,
                        "metadata": {
                            "source_file": file_path.name,
                            "file_path": str(file_path)
                        }
                    })
                except Exception as e:
                    self.logger.error(f"Ошибка загрузки {file_path}: {e}")
        
        if not all_chunks:
            self.logger.error("Не удалось загрузить ни одного файла")
            return pd.DataFrame()
        
        # Создаем DataFrame
        data = []
        for i, chunk in enumerate(all_chunks):
            record = {
                "id": self._generate_chunk_id(chunk["metadata"], i),
                "text": chunk["text"],
                **chunk["metadata"]
            }
            data.append(record)
        
        df = pd.DataFrame(data)
        self.logger.info(f"Создан DataFrame с {len(df)} записями")
        
        return df
    
    def _generate_chunk_id(self, metadata: Dict, index: int) -> str:
        """
        Генерирует уникальный ID для чанка.
        
        Args:
            meta Метаданные чанка
            index: Индекс чанка
        
        Returns:
            Уникальный строковый ID
        """
        # Создаем хэш на основе метаданных
        source = metadata.get("source_file", "unknown")
        chunk_idx = metadata.get("chunk_index", index)
        
        # Используем MD5 для создания короткого уникального ID
        hash_input = f"{source}_{chunk_idx}_{datetime.now().timestamp()}"
        hash_digest = hashlib.md5(hash_input.encode()).hexdigest()[:12]
        return f"chunk_{hash_digest}"
    
    def save_dataframe(self, df: pd.DataFrame, output_path: str):
        """
        Сохраняет DataFrame в файл.
        
        Args:
            df: DataFrame для сохранения
            output_path: Путь для сохранения
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Сохраняем в формате Parquet (эффективно для больших данных)
            if output_path.suffix == '.parquet':
                df.to_parquet(output_path, index=False)
            # Или в CSV
            elif output_path.suffix == '.csv':
                df.to_csv(output_path, index=False, encoding='utf-8')
            # Или в Pickle
            elif output_path.suffix == '.pkl':
                df.to_pickle(output_path)
            else:
                # По умолчанию сохраняем в Parquet
                output_path = output_path.with_suffix('.parquet')
                df.to_parquet(output_path, index=False)
            
            self.logger.info(f"DataFrame сохранен в {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Ошибка сохранения DataFrame: {e}")
            raise
```


```python
class MultilingualE5Embedder:
    """Класс для работы с моделью multilingual-e5-large."""
    
    def __init__(self,
                model_name: str = Config.EMBEDDING_MODEL,
                device: str = None,
                cache_dir: str = None,
                use_fp16: bool = None):
        """
        Инициализация модели эмбеддингов.
        
        Args:
            model_name: Название модели на Hugging Face
            device: Устройство для вычислений (cuda/cpu)
            cache_dir: Директория для кэширования модели
            use_fp16: Использовать ли FP16 для GPU
        """
        self.model_name = model_name
        self.device = device or Config.DEVICE
        self.cache_dir = cache_dir or Config.DEFAULT_MODEL_CACHE
        self.use_fp16 = use_fp16 if use_fp16 is not None else Config.USE_FP16
        
        # Создаем директорию для кэша
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Загружает модель и токенизатор."""
        try:
            self.logger.info(f"Загрузка модели {self.model_name} на устройство {self.device}...")
            
            # Используем sentence-transformers для удобства
            model_kwargs = {}
            if self.device == "cuda" and self.use_fp16:
                model_kwargs["torch_dtype"] = torch.float16
            
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=str(self.cache_dir),
                **model_kwargs
            )
            
            # Загружаем токенизатор отдельно для обработки текста
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir)
            )
            
            self.logger.info(f"✅ Модель загружена на {self.device}")
            self.logger.info(f"📏 Размерность эмбеддингов: {self.get_embedding_dimension()}")
            
            # Выводим информацию о памяти GPU
            if self.device == "cuda":
                gpu_memory = torch.cuda.memory_allocated() / 1e9
                self.logger.info(f"🎮 Использовано памяти GPU: {gpu_memory:.2f} GB")
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки модели: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Возвращает размерность эмбеддингов модели."""
        if self.model is None:
            return Config.EMBEDDING_DIMENSION
        # Получаем размерность из модели
        return self.model.get_sentence_embedding_dimension()
    
    def format_text(self, text: str, text_type: str = "passage") -> str:
        """
        Форматирует текст согласно требованиям E5.
        
        Args:
            text: Исходный текст
            text_type: Тип текста (query/passage)
        
        Returns:
            Отформатированный текст с префиксом
        """
        if text_type not in ["query", "passage"]:
            raise ValueError("text_type должен быть 'query' или 'passage'")
        
        # Очищаем текст и добавляем префикс
        cleaned_text = text.strip()
        return f"{text_type}: {cleaned_text}"
    
    def embed_texts(self,
                   texts: List[str],
                   text_type: str = "passage",
                   batch_size: int = 32,
                   show_progress: bool = True) -> np.ndarray:
        """
        Создает эмбеддинги для списка текстов.
        
        Args:
            texts: Список текстов
            text_type: Тип текстов (query/passage)
            batch_size: Размер батча
            show_progress: Показывать ли прогресс-бар
        
        Returns:
            Массив эмбеддингов
        """
        if not texts:
            return np.array([])
        
        # Форматируем тексты
        formatted_texts = [self.format_text(text, text_type) for text in texts]
        
        self.logger.info(f"🔍 Создание эмбеддингов для {len(texts)} текстов...")
        
        # Создаем эмбеддинги
        try:
            embeddings = self.model.encode(
                formatted_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=self.device
            )
            
            self.logger.info(f"✅ Создано {len(embeddings)} эмбеддингов")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка создания эмбеддингов: {e}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Создает эмбеддинг для одного запроса.
        
        Args:
            query: Текст запроса
        
        Returns:
            Эмбеддинг запроса
        """
        return self.embed_texts([query], text_type="query", show_progress=False)[0]
    
    def compute_similarity(self,
                          query_embedding: np.ndarray,
                          passage_embeddings: np.ndarray) -> np.ndarray:
        """
        Вычисляет косинусное сходство между запросом и пассажами.
        
        Args:
            query_embedding: Эмбеддинг запроса
            passage_embeddings: Эмбеддинги пассажей
        
        Returns:
            Массив сходств
        """
        # Нормализуем векторы (если еще не нормализованы)
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        passages_norm = passage_embeddings / np.linalg.norm(passage_embeddings, axis=1, keepdims=True)
        
        # Вычисляем косинусное сходство
        similarities = np.dot(passages_norm, query_norm)
        return similarities
```


```python
class ChromaStorage:
    """
    Класс для работы с векторным хранилищем ChromaDB.
    Поддерживает multilingual-e5-large модель.
    """
    
    def __init__(self,
                db_path: str,
                collection_name: str = Config.COLLECTION_NAME,
                embedding_model: MultilingualE5Embedder = None,
                create_if_missing: bool = True,
                reset_db: bool = False):
        """
        Инициализация хранилища.
        
        Args:
            db_path: Путь к базе данных
            collection_name: Название коллекции
            embedding_model: Модель эмбеддингов
            create_if_missing: Создавать ли коллекцию, если её нет
            reset_db: Пересоздать ли базу данных (для исправления ошибок)
        """
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.create_if_missing = create_if_missing
        
        # Инициализируем модель эмбеддингов
        if embedding_model is None:
            self.embedding_model = MultilingualE5Embedder()
        else:
            self.embedding_model = embedding_model
        
        # Инициализируем клиента ChromaDB
        self.client = None
        self.collection = None
        self.vector_store = None
        self.logger = logging.getLogger(__name__)
        
        # Пересоздаём базу, если требуется
        if reset_db and self.db_path.exists():
            self.logger.warning(f"🗑️ Удаление повреждённой базы данных: {self.db_path}")
            try:
                shutil.rmtree(self.db_path)
                self.logger.info("✅ База данных удалена")
            except Exception as e:
                self.logger.error(f"❌ Ошибка удаления базы: {e}")
        
        # Создаем директорию для базы данных
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self._init_chroma()
    
    def _init_chroma(self):
        """Инициализирует ChromaDB."""
        try:
            # ✅ НОВЫЙ API ChromaDB (версия >= 1.0.0)
            self.client = chromadb.PersistentClient(
                path=str(self.db_path)
            )
            
            self.logger.info(f"✅ ChromaDB клиент инициализирован: {self.db_path}")
            
            # Создаем или получаем коллекцию
            self._init_collection()
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка инициализации ChromaDB: {e}")
            raise
    
    def _init_collection(self):
        """Инициализирует коллекцию."""
        try:
            # Пытаемся получить существующую коллекцию
            try:
                self.collection = self.client.get_collection(self.collection_name)
                self.logger.info(f"📂 Коллекция '{self.collection_name}' загружена")
            except Exception:
                if self.create_if_missing:
                    # ✅ ИСПРАВЛЕНО: Новый способ создания коллекции
                    self.collection = self.client.create_collection(
                        name=self.collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    self.logger.info(f"🆕 Создана коллекция '{self.collection_name}'")
                else:
                    raise ValueError(f"Коллекция '{self.collection_name}' не существует")
            
            # Создаем LangChain векторное хранилище
            self.vector_store = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self._get_langchain_embedding_function()
            )
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка инициализации коллекции: {e}")
            raise
    
    def _get_langchain_embedding_function(self):
        """Создает функцию эмбеддингов для LangChain."""
        # Создаем обертку для нашей модели
        class E5EmbeddingFunction:
            def __init__(self, embedder: MultilingualE5Embedder):
                self.embedder = embedder
            
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                embeddings = self.embedder.embed_texts(texts, text_type="passage")
                return embeddings.tolist()
            
            def embed_query(self, text: str) -> List[float]:
                embedding = self.embedder.embed_query(text)
                return embedding.tolist()
        
        return E5EmbeddingFunction(self.embedding_model)
    
    def add_texts(self,
                 texts: List[str],
                 metadatas: List[Dict] = None,
                 ids: List[str] = None,
                 batch_size: int = 32) -> List[str]:
        """
        Добавляет тексты в векторное хранилище.
        
        Args:
            texts: Список текстов
            metadatas: Список метаданных
            ids: Список ID (генерируются автоматически, если не указаны)
            batch_size: Размер батча
        
        Returns:
            Список ID добавленных записей
        """
        if not texts:
            self.logger.warning("⚠️ Список текстов пуст")
            return []
        
        # Генерируем ID, если не указаны
        if ids is None:
            ids = [f"doc_{hashlib.md5(text.encode()).hexdigest()[:12]}" 
                  for text in texts]
        
        # Подготавливаем метаданные
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Проверяем длину списков
        if len(texts) != len(ids) or len(texts) != len(metadatas):
            raise ValueError("Длины texts, ids и metadatas должны совпадать")
        
        self.logger.info(f"➕ Добавление {len(texts)} документов...")
        
        # Добавляем батчами
        added_ids = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Добавление в ChromaDB"):
            batch_texts = texts[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            
            # ✅ Создаем эмбеддинги
            batch_embeddings = self.embedding_model.embed_texts(
                batch_texts, 
                text_type="passage",
                show_progress=False
            )
            
            # Добавляем в коллекцию
            self.collection.add(
                documents=batch_texts,
                embeddings=batch_embeddings.tolist(),
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            added_ids.extend(batch_ids)
        
        self.logger.info(f"✅ Добавлено {len(added_ids)} документов")
        return added_ids
    
    def search(self,
              query: str,
              top_k: int = Config.DEFAULT_TOP_K,
              score_threshold: float = Config.SIMILARITY_THRESHOLD,
              filter_meta: Dict[str, Any] = None) -> List[SearchResult]:
        """
        Выполняет семантический поиск.
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            score_threshold: Порог сходства
            filter_meta: Фильтр по метаданным
        
        Returns:
            Список результатов поиска
        """
        if not query.strip():
            raise ValueError("Запрос не может быть пустым")
        
        self.logger.info(f"🔍 Поиск: '{query[:50]}...'")
        
        try:
            # Создаем эмбеддинг запроса
            query_embedding = self.embedding_model.embed_query(query)
            
            # Выполняем поиск
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=filter_meta
            )
            
            # Обрабатываем результаты
            search_results = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    # Получаем расстояние и преобразуем в сходство
                    distance = results['distances'][0][i] if 'distances' in results else 0
                    similarity = 1.0 / (1.0 + distance)  # Преобразование distance в similarity
                    
                    if similarity >= score_threshold:
                        result = SearchResult(
                            id=results['ids'][0][i],
                            content=results['documents'][0][i],
                            similarity_score=similarity,
                            metadata=results['metadatas'][0][i] if results['metadatas'] else {}
                        )
                        search_results.append(result)
            
            self.logger.info(f"✅ Найдено {len(search_results)} результатов")
            return search_results
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка поиска: {e}")
            raise
    
    def delete_by_ids(self, ids: List[str]) -> int:
        """
        Удаляет документы по ID.
        
        Args:
            ids: Список ID для удаления
        
        Returns:
            Количество удаленных документов
        """
        if not ids:
            return 0
        
        try:
            self.collection.delete(ids=ids)
            self.logger.info(f"🗑️ Удалено {len(ids)} документов")
            return len(ids)
        except Exception as e:
            self.logger.error(f"❌ Ошибка удаления: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Возвращает информацию о коллекции."""
        try:
            count = self.collection.count()
            # Получаем метаданные коллекции
            collection_metadata = self.collection.metadata or {}
            # Получаем пример документов
            sample = self.collection.peek() if count > 0 else {}
            
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_dimension": self.embedding_model.get_embedding_dimension(),
                "db_path": str(self.db_path),
                "metadata": collection_metadata,
                "sample_size": len(sample.get("documents", [])),
                "model": self.embedding_model.model_name
            }
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения информации: {e}")
            return {}
    
    def clear_collection(self) -> None:
        """Очищает коллекцию."""
        try:
            # Получаем все ID
            all_data = self.collection.get()
            if all_data['ids']:
                self.delete_by_ids(all_data['ids'])
            self.logger.info(f"🧹 Коллекция '{self.collection_name}' очищена")
        except Exception as e:
            self.logger.error(f"❌ Ошибка очистки коллекции: {e}")
```


```python
class RAGPipeline:
    """
    🚀 Полнофункциональный RAG пайплайн с поддержкой:
    - Векторного хранилища ChromaDB (интеграция через ChromaStorage)
    - LLM генерации ответов
    - RAGAS оценки качества
    - Детальной статистики и визуализации
    """
    
    def __init__(self,
                 data_dir: str = None,
                 db_path: str = None,
                 config: Config = None,
                 device: str = None,
                 llm_generator: LLMGenerator = None):
        """
        Инициализация пайплайна.
        
        Args:
            data_dir: Директория с данными
            db_path: Путь к векторной базе
            config: Конфигурация
            device: Устройство для вычислений (cuda/cpu)
            llm_generator: Генератор ответов (опционально)
        """
        self.config = config or Config
        self.data_dir = Path(data_dir or self.config.DEFAULT_DATA_DIR)
        self.db_path = Path(db_path or self.config.DEFAULT_DB_DIR)
        
        # Переопределяем устройство, если указано
        if device:
            self.config.DEVICE = device
            self.config.USE_FP16 = True if device == "cuda" else False
        
        # Создаем директории
        self.config.setup_directories()
        JupyterConfig.setup_directories()
        
        # Инициализируем компоненты
        self.data_loader = DataLoader(self.data_dir)
        self.embedder = MultilingualE5Embedder(
            device=self.config.DEVICE, 
            use_fp16=self.config.USE_FP16
        )
        
        self.storage = None  
        
        self.llm_generator = llm_generator
        
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        self.logger.info(f"🚀 RAG Pipeline инициализирован на устройстве: {self.config.DEVICE}")
    
    def setup_logging(self):
        """Настраивает логирование."""
        log_file = Path(self.config.DEFAULT_LOGS_DIR) / f"rag_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Удаляем существующие обработчики
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        logging.basicConfig(
            level=getattr(logging, self.config.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    # ========== МЕТОДЫ РАБОТЫ С ДАННЫМИ ==========
    
    def load_and_process_data(self,
                            file_pattern: str = "*.txt",
                            max_files: int = None,
                            save_intermediate: bool = True) -> pd.DataFrame:
        """
        Загружает и обрабатывает данные.
        
        Args:
            file_pattern: Шаблон файлов
            max_files: Максимальное количество файлов
            save_intermediate: Сохранять ли промежуточные результаты
        
        Returns:
            DataFrame с обработанными данными
        """
        self.logger.info("📥 Загрузка данных...")
        
        # Загружаем данные
        df = self.data_loader.load_files_to_dataframe(
            file_pattern=file_pattern,
            max_files=max_files,
            process_chunks=True
        )
        
        if df.empty:
            self.logger.error("❌ Не удалось загрузить данные")
            return df
        
        # Сохраняем промежуточные результаты
        if save_intermediate:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(JupyterConfig.RESULTS_DIR) / f"processed_data_{timestamp}.parquet"
            self.data_loader.save_dataframe(df, output_path)
            self.logger.info(f"💾 Промежуточные данные сохранены: {output_path}")
        
        # Отображаем информацию о данных
        display_df = df.head(JupyterConfig.DISPLAY_TABLE_ROWS).copy()
        display_df['text_preview'] = display_df['text'].apply(
            lambda x: x[:JupyterConfig.DISPLAY_MAX_CHARS] + '...' if len(x) > JupyterConfig.DISPLAY_MAX_CHARS else x
        )
        
        print(f"\n📊 Загружено {len(df)} чанков из {df['source_file'].nunique()} файлов")
        print(f"🔤 Примеры данных:")
        display(display_df[['id', 'source_file', 'chunk_index', 'total_chunks', 'text_preview']])
        
        return df
    
    def initialize_storage(self, collection_name: str = None) -> ChromaStorage:
        """
        Инициализирует векторное хранилище.
        
        Args:
            collection_name: Название коллекции
        
        Returns:
            Инициализированное хранилище
        """
        self.logger.info("📦 Инициализация векторного хранилища...")
        
        # ✅ Используем ChromaStorage класс из старой версии
        self.storage = ChromaStorage(
            db_path=str(self.db_path),
            collection_name=collection_name or self.config.COLLECTION_NAME,
            embedding_model=self.embedder
        )
        
        info = self.storage.get_collection_info()
        self.logger.info(f"✅ Хранилище инициализировано: {info}")
        
        # Отображаем информацию о коллекции
        html = f"""
        <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 10px 0; background-color: #f8f9fa;">
            <h3 style="color: #2196F3; margin-top: 0;">📦 Информация о векторном хранилище</h3>
            <ul style="line-height: 1.6;">
                <li><strong>Название коллекции:</strong> {info['collection_name']}</li>
                <li><strong>Количество документов:</strong> {info['document_count']}</li>
                <li><strong>Размерность эмбеддингов:</strong> {info['embedding_dimension']}</li>
                <li><strong>Модель эмбеддингов:</strong> {info['model']}</li>
                <li><strong>Путь к базе:</strong> {info['db_path']}</li>
            </ul>
        </div>
        """
        display(HTML(html))
        
        return self.storage
    
    def build_vector_store(self,
                         df: pd.DataFrame,
                         clear_existing: bool = False) -> List[str]:
        """
        Строит векторное хранилище из DataFrame.
        
        Args:
            df: DataFrame с данными
            clear_existing: Очистить ли существующую коллекцию
        
        Returns:
            Список ID добавленных документов
        """
        if self.storage is None:
            self.initialize_storage()
        
        if df.empty:
            self.logger.error("❌ DataFrame пуст")
            return []
        
        # Очищаем коллекцию, если нужно
        if clear_existing:
            self.logger.info("🧹 Очистка существующей коллекции...")
            self.storage.clear_collection()
        
        # Подготавливаем данные
        texts = df["text"].tolist()
        ids = df["id"].tolist()
        
        # Создаем метаданные
        metadatas = []
        for _, row in df.iterrows():
            metadata = {k: v for k, v in row.items() if k not in ["id", "text"]}
            metadata["id"] = row["id"]
            metadatas.append(metadata)
        
        self.logger.info(f"🚀 Добавление {len(texts)} документов в векторное хранилище...")
        
        # Добавляем в хранилище
        added_ids = self.storage.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            batch_size=self.config.BATCH_SIZE
        )
        
        # Получаем информацию о коллекции
        info = self.storage.get_collection_info()
        self.logger.info(f"✅ Векторное хранилище построено: {info}")
        
        # Отображаем статистику
        html = f"""
        <div style="border: 2px solid #4CAF50; border-radius: 8px; padding: 15px; margin: 10px 0; background-color: #e8f5e9;">
            <h3 style="color: #4CAF50; margin-top: 0;">✅ Векторное хранилище успешно построено!</h3>
            <ul style="line-height: 1.6;">
                <li><strong>Добавлено документов:</strong> {len(added_ids)}</li>
                <li><strong>Общее количество в коллекции:</strong> {info['document_count']}</li>
                <li><strong>Размерность векторов:</strong> {info['embedding_dimension']}</li>
                <li><strong>Модель:</strong> {info['model']}</li>
            </ul>
        </div>
        """
        display(HTML(html))
        
        return added_ids
    
    # ========== МЕТОДЫ ПОИСКА ==========
    
    def search(self,
              query: str,
              top_k: int = None,
              score_threshold: float = None,
              return_df: bool = False) -> Union[List[SearchResult], pd.DataFrame]:
        """
        Выполняет поиск в векторном хранилище.
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            score_threshold: Порог сходства
            return_df: Вернуть результаты в виде DataFrame
        
        Returns:
            Результаты поиска как список SearchResult или DataFrame
        """
        if self.storage is None:
            self.logger.error("❌ Хранилище не инициализировано")
            return [] if not return_df else pd.DataFrame()
        
        top_k = top_k or self.config.DEFAULT_TOP_K
        score_threshold = score_threshold or self.config.SIMILARITY_THRESHOLD
        
        self.logger.info(f"🔍 Поиск запроса: '{query}'")
        
        # ✅ Используем метод search из ChromaStorage
        results = self.storage.search(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold
        )
        
        # Логируем результаты
        if results:
            self.logger.info(f"✅ Найдено {len(results)} результатов:")
            for i, result in enumerate(results[:3]):  # Показываем только первые 3
                self.logger.info(f"  {i+1}. Score: {result.similarity_score:.4f}, Source: {result.metadata.get('source_file', 'unknown')}")
        else:
            self.logger.info("⚠️ Результаты не найдены")
        
        # Отображаем результаты в интерактивном формате
        if results:
            self._display_search_results(query, results, score_threshold)
        
        if return_df:
            return self.results_to_dataframe(results)
        
        return results
    
    def _display_search_results(self, query: str, results: List[SearchResult], score_threshold: float):
        """Отображает результаты поиска в интерактивном формате."""
        html = f"""
        <div style="border: 1px solid #2196F3; border-radius: 8px; padding: 15px; margin: 10px 0;">
            <h3 style="color: #2196F3; margin-top: 0;">🔍 Результаты поиска для: "{query}"</h3>
            <p style="color: #666;"><strong>Найдено:</strong> {len(results)} результатов | <strong>Порог сходства:</strong> {score_threshold}</p>
        """
        
        for i, result in enumerate(results):
            color = '#e8f5e9' if result.similarity_score > 0.8 else '#fff8e1' if result.similarity_score > 0.6 else '#ffebee'
            border_color = '#4CAF50' if result.similarity_score > 0.8 else '#FFA726' if result.similarity_score > 0.6 else '#EF5350'
            
            html += f"""
            <div style="border: 2px solid {border_color}; border-radius: 8px; padding: 15px; margin: 10px 0; background-color: {color};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <strong style="font-size: 1.1em; color: #1a237e;">Результат #{i+1}</strong>
                    <span style="background-color: {border_color}; color: white; padding: 3px 8px; border-radius: 12px; font-weight: bold;">
                        Score: {result.similarity_score:.4f}
                    </span>
                </div>
                <div style="background-color: white; padding: 10px; border-radius: 5px; margin: 5px 0; border-left: 3px solid {border_color};">
                    {result.content[:300]}...
                </div>
                <div style="color: #666; font-size: 0.9em; margin-top: 8px;">
                    <strong>Источник:</strong> {result.metadata.get('source_file', 'unknown')} | 
                    <strong>Чанк:</strong> {result.metadata.get('chunk_index', 'N/A')}/{result.metadata.get('total_chunks', 'N/A')}
                </div>
            </div>
            """
        
        html += "</div>"
        display(HTML(html))
    
    def results_to_dataframe(self, results: List[SearchResult]) -> pd.DataFrame:
        """
        Конвертирует результаты поиска в DataFrame.
        
        Args:
            results: Список результатов поиска
        
        Returns:
            DataFrame с результатами
        """
        if not results:
            return pd.DataFrame()
        
        data = []
        for result in results:
            row = {
                'id': result.id,
                'content': result.content,
                'similarity_score': result.similarity_score,
                'source_file': result.metadata.get('source_file', ''),
                'chunk_index': result.metadata.get('chunk_index', ''),
                'total_chunks': result.metadata.get('total_chunks', ''),
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        return df
    
    # ========== LLM ГЕНЕРАЦИЯ ==========
    
    def set_llm_generator(self, llm_generator: LLMGenerator):
        """Устанавливает LLM генератор."""
        self.llm_generator = llm_generator
        self.logger.info("✅ LLM генератор установлен")
    
    def query(self,
             query: str,
             top_k: int = None,
             score_threshold: float = None,
             return_details: bool = False,
             display_result: bool = True) -> Union[str, Dict]:
        """
        ✨ Полный RAG запрос: поиск + генерация ответа.
        
        Args:
            query: Вопрос пользователя
            top_k: Количество контекстов для поиска
            score_threshold: Порог сходства
            return_details: Возвращать детали (контексты, источники)
            display_result: Отображать результат в интерактивном формате
        
        Returns:
            Ответ или словарь с деталями
        """
        if self.llm_generator is None:
            raise ValueError("❌ LLM генератор не установлен. Используйте set_llm_generator()")
        
        if self.storage is None:
            raise ValueError("❌ Хранилище не инициализировано. Используйте initialize_storage()")
        
        top_k = top_k or self.config.DEFAULT_TOP_K
        score_threshold = score_threshold or self.config.SIMILARITY_THRESHOLD
        
        self.logger.info(f"💬 RAG запрос: '{query}'")
        
        # 1. Поиск контекстов
        search_results = self.search(query, top_k=top_k, score_threshold=score_threshold)
        
        if not search_results:
            answer = "Извините, не удалось найти релевантную информацию для ответа на ваш вопрос."
            if display_result:
                self._display_query_result(query, answer, [], [], [])
            if return_details:
                return {"answer": answer, "contexts": [], "sources": [], "scores": []}
            return answer
        
        # 2. Извлекаем контексты
        contexts = [r.content for r in search_results]
        sources = [r.metadata.get('source_file', 'unknown') for r in search_results]
        scores = [r.similarity_score for r in search_results]
        
        # 3. Генерация ответа
        self.logger.info("🤖 Генерация ответа...")
        answer = self.llm_generator.generate_answer(query, contexts)
        
        # 4. Отображаем результат
        if display_result:
            self._display_query_result(query, answer, contexts, sources, scores)
        
        if return_details:
            return {
                "answer": answer,
                "contexts": contexts,
                "sources": sources,
                "scores": scores,
                "search_results": search_results
            }
        
        return answer
    
    def _display_query_result(self, query: str, answer: str, contexts: List[str], 
                             sources: List[str], scores: List[float]):
        """Отображает результат RAG запроса в интерактивном формате."""
        html = f"""
        <div style="border: 2px solid #673AB7; border-radius: 12px; padding: 20px; margin: 15px 0; background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);">
            <h2 style="color: #4A148C; margin-top: 0;">💬 RAG Ответ</h2>
            
            <div style="background: white; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 5px solid #673AB7;">
                <strong style="color: #666; display: block; margin-bottom: 8px;">❓ Вопрос:</strong>
                <div style="color: #1a237e; font-size: 1.1em;">{query}</div>
            </div>
            
            <div style="background: #e8f5e9; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 5px solid #4CAF50;">
                <strong style="color: #666; display: block; margin-bottom: 8px;">✅ Ответ:</strong>
                <div style="color: #1b5e20; line-height: 1.6;">{answer}</div>
            </div>
        """
        
        if contexts:
            html += f"""
            <div style="margin-top: 20px;">
                <h3 style="color: #4A148C;">📚 Использованные контексты ({len(contexts)}):</h3>
            """
            
            for i, (ctx, src, score) in enumerate(zip(contexts, sources, scores)):
                color = '#4CAF50' if score > 0.8 else '#FFA726' if score > 0.6 else '#EF5350'
                html += f"""
                <div style="background: white; padding: 12px; margin: 8px 0; border-radius: 6px; border-left: 4px solid {color};">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <strong style="color: #666;">Контекст #{i+1}</strong>
                        <span style="background: {color}; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.9em;">
                            Score: {score:.4f}
                        </span>
                    </div>
                    <div style="color: #333; font-size: 0.95em; margin: 8px 0;">{ctx[:200]}...</div>
                    <div style="color: #999; font-size: 0.85em;">📄 Источник: {src}</div>
                </div>
                """
            
            html += "</div>"
        
        html += "</div>"
        display(HTML(html))
    
    # ========== ОЦЕНКА КАЧЕСТВА ==========
    
    def evaluate_search(self,
                       test_queries: List[str],
                       expected_results: List[List[str]] = None,
                       display_results: bool = True) -> Dict[str, Any]:
        """
        Оценивает качество поиска.
        
        Args:
            test_queries: Тестовые запросы
            expected_results: Ожидаемые результаты (опционально)
            display_results: Отображать ли результаты интерактивно
        
        Returns:
            Метрики оценки
        """
        self.logger.info(f"🧪 Оценка поиска на {len(test_queries)} запросах...")
        
        metrics = {
            "total_queries": len(test_queries),
            "results_per_query": [],
            "avg_similarity_score": [],
            "execution_times": [],
            "successful_queries": 0
        }
        
        evaluation_results = []
        
        for query in tqdm(test_queries, desc="Оценка запросов"):
            start_time = datetime.now()
            try:
                results = self.search(query, top_k=5, return_df=False)
                execution_time = (datetime.now() - start_time).total_seconds()
                
                metrics["results_per_query"].append(len(results))
                metrics["execution_times"].append(execution_time)
                
                if results:
                    metrics["successful_queries"] += 1
                    avg_similarity = np.mean([r.similarity_score for r in results])
                    metrics["avg_similarity_score"].append(avg_similarity)
                    
                    # Сохраняем для детального анализа
                    eval_result = {
                        'query': query,
                        'results_count': len(results),
                        'avg_similarity': avg_similarity,
                        'execution_time': execution_time,
                        'top_result': results[0].content[:100] + '...' if results else None
                    }
                    evaluation_results.append(eval_result)
                
            except Exception as e:
                self.logger.error(f"❌ Ошибка при поиске '{query}': {e}")
                metrics["results_per_query"].append(0)
                metrics["execution_times"].append(0)
        
        # Вычисляем агрегированные метрики
        metrics["avg_results_per_query"] = np.mean(metrics["results_per_query"]) if metrics["results_per_query"] else 0
        metrics["avg_execution_time"] = np.mean(metrics["execution_times"]) if metrics["execution_times"] else 0
        metrics["success_rate"] = metrics["successful_queries"] / metrics["total_queries"] if metrics["total_queries"] > 0 else 0
        
        if metrics["avg_similarity_score"]:
            metrics["avg_similarity"] = np.mean(metrics["avg_similarity_score"])
        
        self.logger.info(f"✅ Оценка завершена: {metrics}")
        
        # Отображаем результаты в интерактивном формате
        if display_results:
            self.display_evaluation_metrics(metrics, evaluation_results)
        
        return metrics
    
    def display_evaluation_metrics(self, metrics: Dict[str, Any], evaluation_results: List[Dict]):
        """Отображает метрики оценки в интерактивном формате."""
        html = f"""
        <div style="border: 2px solid #2196F3; border-radius: 8px; padding: 20px; margin: 15px 0; background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);">
            <h2 style="color: #1565C0; margin-top: 0; text-align: center;">📊 Метрики оценки поиска</h2>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0;">
        """
        
        metric_items = [
            ('Всего запросов', metrics['total_queries'], '#1565C0'),
            ('Успешных запросов', metrics['successful_queries'], '#4CAF50'),
            ('Скорость успеха', f"{metrics['success_rate']:.1%}", '#FFA726'),
            ('Среднее время', f"{metrics['avg_execution_time']:.3f} сек", '#9C27B0'),
            ('Результатов на запрос', f"{metrics['avg_results_per_query']:.1f}", '#009688'),
            ('Среднее сходство', f"{metrics.get('avg_similarity', 0):.4f}", '#E91E63')
        ]
        
        for title, value, color in metric_items:
            html += f"""
            <div style="text-align: center; padding: 15px; background: {color}15; border-radius: 8px; border: 1px solid {color}50;">
                <div style="font-size: 2.5em; color: {color}; margin-bottom: 8px;">{value}</div>
                <div style="font-weight: 500; color: #333;">{title}</div>
            </div>
            """
        
        html += """
            </div>
            
            <h3 style="color: #1565C0; margin-top: 25px;">📈 Детальные результаты по запросам:</h3>
            <div style="margin-top: 15px;">
        """
        
        for i, result in enumerate(evaluation_results[:10]):  # Показываем только первые 10
            success_color = '#4CAF50' if result['results_count'] > 0 else '#EF5350'
            html += f"""
            <div style="background: white; padding: 12px; margin: 8px 0; border-radius: 6px; border-left: 4px solid {success_color};">
                <div style="display: flex; justify-content: space-between;">
                    <strong style="color: #1a237e;">Запрос #{i+1}:</strong>
                    <span style="color: {success_color}; font-weight: bold;">{result['results_count']} результатов</span>
                </div>
                <div style="color: #666; margin: 5px 0; font-style: italic;">"{result['query']}"</div>
                <div style="display: flex; justify-content: space-between; font-size: 0.9em; color: #666;">
                    <span>⏰ Время: {result['execution_time']:.3f} сек</span>
                    <span>⭐ Сходство: {result['avg_similarity']:.4f}</span>
                </div>
                <div style="color: #666; margin-top: 5px; font-size: 0.9em;">
                    <strong>Топ результат:</strong> {result['top_result'] or 'Нет результатов'}
                </div>
            </div>
            """
        
        html += """
            </div>
        </div>
        """
        
        display(HTML(html))
    
    # ========== ЭКСПОРТ И СОХРАНЕНИЕ ==========
    
    def save_pipeline_state(self, filename: str = None):
        """
        Сохраняет состояние пайплайна.
        
        Args:
            filename: Имя файла для сохранения
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_pipeline_state_{timestamp}.pkl"
        
        filepath = Path(JupyterConfig.RESULTS_DIR) / filename
        
        state = {
            'config': asdict(self.config),
            'data_dir': str(self.data_dir),
            'db_path': str(self.db_path),
            'collection_info': self.storage.get_collection_info() if self.storage else None,
            'has_llm': self.llm_generator is not None,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"💾 Состояние пайплайна сохранено: {filepath}")
        return filepath
    
    def export_results(self,
                     results: List[SearchResult],
                     output_format: str = "json",
                     filename: str = None) -> str:
        """
        Экспортирует результаты поиска.
        
        Args:
            results: Результаты поиска
            output_format: Формат экспорта (json/csv/html)
            filename: Имя файла для сохранения
        
        Returns:
            Путь к сохраненному файлу
        """
        if not results:
            self.logger.warning("⚠️ Нет результатов для экспорта")
            return ""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"search_results_{timestamp}"
        
        filepath = Path(JupyterConfig.RESULTS_DIR) / filename
        
        if output_format == "json":
            result_dicts = [r.to_dict() for r in results]
            filepath = filepath.with_suffix('.json')
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_dicts, f, ensure_ascii=False, indent=2)
            print(f"💾 Результаты сохранены в JSON: {filepath}")
            return str(filepath)
            
        elif output_format == "csv":
            df = self.results_to_dataframe(results)
            filepath = filepath.with_suffix('.csv')
            df.to_csv(filepath, index=False, encoding='utf-8')
            print(f"💾 Результаты сохранены в CSV: {filepath}")
            return str(filepath)
            
        elif output_format == "html":
            html_content = self.generate_html_report(results)
            filepath = filepath.with_suffix('.html')
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"💾 HTML отчет сохранен: {filepath}")
            return str(filepath)
            
        else:
            raise ValueError(f"Неподдерживаемый формат: {output_format}")
    
    def generate_html_report(self, results: List[SearchResult]) -> str:
        """Генерирует HTML отчет с результатами поиска."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>RAG Search Results</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.12); }
                .header { text-align: center; color: #1a237e; margin-bottom: 30px; }
                .result { border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 15px 0; background: #f8f9fa; }
                .score { background: #4CAF50; color: white; padding: 3px 8px; border-radius: 12px; font-weight: bold; float: right; }
                .content { background: white; padding: 10px; margin: 10px 0; border-left: 3px solid #2196F3; }
                .metadata { color: #666; font-size: 0.9em; margin-top: 8px; }
                .source { color: #1565C0; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔍 RAG Search Results</h1>
                    <p>Generated on """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
                </div>
        """
        
        for i, result in enumerate(results):
            html += f"""
                <div class="result">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3>Result #{i+1}</h3>
                        <span class="score">Score: {result.similarity_score:.4f}</span>
                    </div>
                    <div class="content">
                        {result.content.replace('\n', '<br>')}
                    </div>
                    <div class="metadata">
                        <span class="source">Source: {result.metadata.get('source_file', 'unknown')}</span> | 
                        Chunk: {result.metadata.get('chunk_index', 'N/A')}/{result.metadata.get('total_chunks', 'N/A')} |
                        ID: {result.id[:12]}...
                    </div>
                </div>
            """
        
        html += """
            </div>
        </body>
        </html>
        """
        return html
```


```python
def get_device_info():
    """Возвращает информацию об устройстве."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "device": device,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None
    }
```


```python
def initialize_rag_pipeline(
    data_dir: str = "data/chunks",
    db_path: str = "vector_db",
    device: str = None,
    setup_llm: bool = True,
    max_files: int = 1000,
    build_store: bool = False,
    reset_db: bool = False
) -> RAGPipeline:
    """
    Полная инициализация RAG пайплайна.
    
    Args:
        data_dir: Директория с данными
        db_path: Путь к векторной базе
        device: Устройство (auto определяет автоматически)
        setup_llm: Настроить ли LLM генератор
        max_files: Максимум файлов для загрузки
        reset_db: Пересоздать базу данных (решает проблемы с повреждённой БД)
    
    Returns:
        Инициализированный RAG пайплайн
    """
    try:
        # Определяем устройство
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🔧 Инициализация RAG пайплайна на устройстве: {device}")
        
        # Показываем информацию об устройстве
        device_info = get_device_info()
        print(f"\nℹ️  Информация об устройстве:")
        print(f"   • Устройство: {device_info['device']}")
        if device_info['gpu_available']:
            print(f"   • GPU: {device_info['gpu_name']}")
            print(f"   • Память GPU: {device_info['gpu_memory']:.2f} GB")
            print(f"   • CUDA версия: {device_info['cuda_version']}")
        
        # ✅ Если требуется - удаляем старую базу
        if reset_db:
            db_path_obj = Path(db_path)
            if db_path_obj.exists():
                print(f"\n🗑️ Удаление старой базы данных: {db_path}")
                shutil.rmtree(db_path_obj)
                print("✅ Старая база удалена")
        
        # Создаем пайплайн
        pipeline = RAGPipeline(
            data_dir=data_dir,
            db_path=db_path,
            device=device
        )
        
        # Настраиваем LLM генератор
        if setup_llm:
            print("\n🤖 Настройка LLM генератора...")
            generation_client = UniversalLLMClient(
                model=Config.GENERATION_MODEL,
                api_key=Config.LLM_API_KEY,
                api_base=Config.LLM_API_BASE,
                temperature=Config.GENERATION_TEMPERATURE,
                max_tokens=Config.GENERATION_MAX_TOKENS
            )
            llm_generator = LLMGenerator(llm_client=generation_client)
            pipeline.set_llm_generator(llm_generator)
        
        # Загружаем данные
        print("\n📥 Загрузка данных...")
        df = pipeline.load_and_process_data(
            file_pattern="*.txt",
            max_files=max_files,
            save_intermediate=True
        )
        
        if not df.empty:
            # Инициализируем хранилище
            print("\n📦 Инициализация векторного хранилища...")
            pipeline.initialize_storage()

            if build_store:
                # Строим векторное хранилище
                print("\n🚀 Построение векторного хранилища...")
                pipeline.build_vector_store(df, clear_existing=True)
            
            print("\n✅ Пайплайн успешно инициализирован и готов к работе!")
            print("\n📝 Доступные методы:")
            print("   • pipeline.search(query) - поиск по векторной базе")
            print("   • pipeline.query(query) - полный RAG запрос с генерацией ответа")
            print("   • pipeline.evaluate_search(queries) - оценка качества поиска")
            
            return pipeline
        else:
            print("❌ Не удалось загрузить данные")
            return None
            
    except Exception as e:
        print(f"❌ Ошибка инициализации пайплайна: {e}")
        import traceback
        traceback.print_exc()
        return None
```

## Проход по конвейеру

### Инициализация


```python
import hashlib
```


```python
pipeline = initialize_rag_pipeline(
    data_dir="data/chunks",
    db_path="vector_db",
    setup_llm=True,
    max_files=1000,
    # build_store=True,
    # reset_db=True
)
```

### Создание тестовых данных


```python
test_cases = [
    {
        "query": "Tell me all about I101 and its relations in Attempto Controlled English",
        "ground_truth": """i101 is a length_measure_with_unit. It represents a measurement involving 
        a length and a unit. i101 has two key components: i101_value_component (the numerical value) 
        and i17 (the unit component). i101 also serves as a conversion factor for i103."""
    },
    {
        "query": "What is i103 and how does it relate to i101?",
        "ground_truth": """i103 is a conversion_based_unit that uses i101 as its conversion factor. 
        This means i101 provides the necessary factor for converting between i103 and other units."""
    },
    {
        "query": "Explain the relationship between i101, i17, and i101_value_component",
        "ground_truth": """i101 is a length_measure_with_unit that consists of two parts: 
        i17 (unit component) and i101_value_component (value component). The measure i101 
        combines these two to represent a complete length measurement."""
    }
]
```


```python
test_queries = [test_cases[0]['query'], test_cases[1]['query'], test_cases[2]['query']]
test_queries
```




    ['Tell me all about I101 and its relations in Attempto Controlled English',
     'What is i103 and how does it relate to i101?',
     'Explain the relationship between i101, i17, and i101_value_component']



### Генерация ответов через RAG


```python
question = "Tell me all about I101 and its relations in Attempto Controlled English"
```


```python
if pipeline:
    # 1. Простой поиск
    results = pipeline.search(question, 
                              top_k=5)
```


```python
# 2. Полный RAG запрос
answer = pipeline.query(question, display_result=True)
```


```python
# 3. Оценка качества
metrics = pipeline.evaluate_search(test_queries)
```

## Метрики RAGAS


```python
def prepare_rag_evaluation_dataset(
    pipeline,  # RAGPipeline
    test_cases: List[Dict[str, str]],
    top_k: int = 5
) -> Tuple[List[str], List[str], List[List[str]], List[str]]:
    """
    Подготавливает данные для оценки RAGAS, получая ответы и контексты от RAG системы.
    
    Args:
        pipeline: Инициализированный RAG пайплайн
        test_cases: Список тестовых кейсов с query и ground_truth
        top_k: Количество контекстов для извлечения
    
    Returns:
        Кортеж (queries, answers, contexts_list, ground_truths)
    """
    queries = []
    answers = []
    contexts_list = []
    ground_truths = []
    
    print(f"🔄 Генерация ответов для {len(test_cases)} тестовых запросов...")
    
    for test_case in tqdm(test_cases, desc="Обработка запросов"):
        query = test_case["query"]
        ground_truth = test_case.get("ground_truth", "")
        
        # Получаем ответ и контексты от RAG системы
        try:
            result = pipeline.query(
                query=query,
                top_k=top_k,
                return_details=True,
                display_result=False  # Не показываем при генерации датасета
            )
            
            # Сохраняем данные
            queries.append(query)
            answers.append(result["answer"])
            contexts_list.append(result["contexts"])
            ground_truths.append(ground_truth)
            
            print(f"✅ Запрос обработан: {query[:50]}...")
            
        except Exception as e:
            print(f"❌ Ошибка при обработке запроса '{query[:50]}...': {e}")
            # Добавляем пустые значения, чтобы не нарушить индексацию
            queries.append(query)
            answers.append("")
            contexts_list.append([])
            ground_truths.append(ground_truth)
    
    print(f"\n✅ Датасет подготовлен:")
    print(f"   • Запросов: {len(queries)}")
    print(f"   • С контекстами: {sum(1 for c in contexts_list if c)}")
    print(f"   • С ground truth: {sum(1 for gt in ground_truths if gt)}")
    
    return queries, answers, contexts_list, ground_truths
```


```python
def evaluate_rag_with_ragas(
    pipeline,  # RAGPipeline
    test_cases: List[Dict[str, str]],
    judge_model: str = "openai/gpt-5-mini",
    judge_api_key: str = None,
    judge_api_base: str = "https://api.vsegpt.ru/v1",
    embedding_model: str = "emb-openai/text-embedding-3-large",
    top_k: int = 5,
    metrics: List[str] = None
) -> Dict[str, Any]:
    """
    Полная оценка RAG системы с использованием RAGAS 0.4+.
    
    Args:
        pipeline: Инициализированный RAG пайплайн
        test_cases: Тестовые кейсы
        judge_model: Модель для оценки (без префикса провайдера)
        judge_api_key: API ключ для Judge LLM
        judge_api_base: Base URL для Judge LLM API
        embedding_model: Модель embeddings
        top_k: Количество контекстов
        metrics: Список метрик (None = базовые)
    
    Returns:
        Результаты оценки
    """
    if not RAGAS_AVAILABLE:
        raise ImportError("RAGAS не установлена. Установите: pip install ragas --upgrade")
    
    # 1. Подготавливаем датасет (получаем ответы от RAG)
    print("=" * 80)
    print("📊 ШАГ 1: Подготовка датасета")
    print("=" * 80)
    queries, answers, contexts_list, ground_truths = prepare_rag_evaluation_dataset(
        pipeline, test_cases, top_k
    )
    
    # 2. Создаем оценщика (Judge LLM)
    print("\n" + "=" * 80)
    print("🤖 ШАГ 2: Инициализация Judge LLM для оценки")
    print("=" * 80)
    
    # Если не указан API ключ, пытаемся получить из окружения
    if judge_api_key is None:
        judge_api_key = os.getenv("OPENAI_API_KEY")
        if not judge_api_key:
            raise ValueError("Необходимо указать judge_api_key или установить OPENAI_API_KEY")
    
    evaluation_client = UniversalLLMClient(
        model=judge_model,
        api_key=judge_api_key,
        api_base=judge_api_base,
        temperature=0.0,  # Детерминированные оценки
        max_tokens=32000
    )
    
    # Определяем метрики
    if metrics is None:
        # Проверяем, есть ли ground truth
        has_ground_truth = any(gt and gt.strip() for gt in ground_truths)
        
        if has_ground_truth:
            # Используем все доступные метрики
            metrics = [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
                "context_relevance",
                "response_groundedness",
                "answer_accuracy"
            ]
        else:
            # Используем только базовые метрики (без ground truth)
            metrics = [
                "faithfulness",
                "context_relevance",
                "response_groundedness"
            ]
    
    evaluator = UniversalRAGEvaluator(
        judge_llm_client=evaluation_client,
        embedding_model=embedding_model,
        metrics=metrics
    )
    
    # 3. Запускаем оценку
    print("\n" + "=" * 80)
    print("🚀 ШАГ 3: Запуск оценки RAGAS")
    print("=" * 80)
    print(f"⚙️  Метрики: {', '.join(metrics)}")
    print(f"⏰ Это может занять несколько минут (LLM вызовы для каждой метрики)...\n")
    
    # Выполняем оценку
    results = evaluator.evaluate(
        queries=queries,
        answers=answers,
        contexts=contexts_list,
        ground_truths=ground_truths,
        show_progress=True
    )
    
    # 4. Отображаем результаты
    print("\n" + "=" * 80)
    print("📊 ШАГ 4: Результаты оценки")
    print("=" * 80)
    
    evaluator.display_results(max_examples=len(test_cases))
    
    # 5. Сохраняем результаты
    print("\n" + "=" * 80)
    print("💾 ШАГ 5: Сохранение результатов")
    print("=" * 80)
    
    results_path = evaluator.save_results()
    print(f"✅ Детальные результаты сохранены: {results_path}")
    
    # Сохраняем также агрегированные метрики
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = Path("results") / f"ragas_metrics_{timestamp}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ Агрегированные метрики сохранены: {metrics_path}")
    
    return {
        "aggregated_metrics": results,
        "detailed_results": evaluator.get_detailed_results(),
        "evaluator": evaluator
    }
```


```python
test_cases = [
    {
        "query": "Tell me all about I101 and its relations in Attempto Controlled English",
        "ground_truth": """Okay, let's break down the information about i101 and its relationships as expressed in Attempto Controlled English (ACE) using the provided axioms.

Here's a description, presented in a structured way:

1. What is i101?
According to the axioms, i101 is a length_measure_with_unit. This means i101 represents a measurement involving a length and a unit.

2. Components of i101:
i101 has two key components:
- Value Component: i101_value_component is a length_measure
- Unit Component: i17 is the unit associated with i101

3. i101's Role in Unit Conversion:
i101 serves as a conversion factor for conversion_based_unit i103."""
    },
    {
        "query": "What is the relationship between i101 and i103?",
        "ground_truth": "i101 serves as a conversion factor for i103, where i103 is a conversion_based_unit."
    },
    {
        "query": "What type of measurement is i101?",
        "ground_truth": "i101 is a length_measure_with_unit, which means it represents a length measurement with an associated unit."
    },
    {
        "query": "What is i101_value_component?",
        "ground_truth": "i101_value_component is a length_measure that represents the numerical value component of the i101 measurement."
    },
    {
        "query": "What unit is associated with i101?",
        "ground_truth": "The unit i17 is associated with i101 through the measure_with_unit_has_unit_component relationship."
    }
]
```


```python
evaluation_results = evaluate_rag_with_ragas(
    pipeline=pipeline,
    test_cases=test_cases,
    judge_model="openai/gpt-5-mini",
    judge_api_key="sk-or-vv-xxx",
    judge_api_base="https://api.vsegpt.ru/v1",
    embedding_model="emb-openai/text-embedding-3-large",
    top_k=5,
    metrics=None  # None = автоопределение на основе ground truth
)
```


```python
# Результаты доступны в:
# - evaluation_results["aggregated_metrics"] - агрегированные метрики
# - evaluation_results["detailed_results"] - детальная таблица
# - evaluation_results["evaluator"] - объект evaluator для дополнительного анализа
```


```python
for metric, value in evaluation_results["aggregated_metrics"].items():
    print(f"  • {metric}: {value:.4f}")
```

      • faithfulness_mean: 1.0000
      • faithfulness_std: 0.0000
      • faithfulness_min: 1.0000
      • faithfulness_max: 1.0000
      • faithfulness_count: 5.0000
      • answer_relevancy_mean: 0.4306
      • answer_relevancy_std: 0.3782
      • answer_relevancy_min: 0.0000
      • answer_relevancy_max: 0.9426
      • answer_relevancy_count: 5.0000
      • context_precision_mean: 0.1000
      • context_precision_std: 0.1225
      • context_precision_min: 0.0000
      • context_precision_max: 0.2500
      • context_precision_count: 5.0000
      • context_recall_mean: 0.2500
      • context_recall_std: 0.2500
      • context_recall_min: 0.0000
      • context_recall_max: 0.5000
      • context_recall_count: 4.0000
      • context_relevance_mean: 0.6000
      • context_relevance_std: 0.4062
      • context_relevance_min: 0.0000
      • context_relevance_max: 1.0000
      • context_relevance_count: 5.0000
      • response_groundedness_mean: 1.0000
      • response_groundedness_std: 0.0000
      • response_groundedness_min: 1.0000
      • response_groundedness_max: 1.0000
      • response_groundedness_count: 5.0000
      • answer_accuracy_mean: 0.4000
      • answer_accuracy_std: 0.3391
      • answer_accuracy_min: 0.0000
      • answer_accuracy_max: 0.7500
      • answer_accuracy_count: 5.0000
      • overall_mean: 0.5401
    


```python
evaluation_results["aggregated_metrics"]
```




    {'faithfulness_mean': 1.0,
     'faithfulness_std': 0.0,
     'faithfulness_min': 1.0,
     'faithfulness_max': 1.0,
     'faithfulness_count': 5,
     'answer_relevancy_mean': 0.43056041541699397,
     'answer_relevancy_std': 0.3781564037159313,
     'answer_relevancy_min': 0.0,
     'answer_relevancy_max': 0.9426166505197333,
     'answer_relevancy_count': 5,
     'context_precision_mean': 0.09999999999,
     'context_precision_std': 0.12247448712691146,
     'context_precision_min': 0.0,
     'context_precision_max': 0.249999999975,
     'context_precision_count': 5,
     'context_recall_mean': 0.25,
     'context_recall_std': 0.25,
     'context_recall_min': 0.0,
     'context_recall_max': 0.5,
     'context_recall_count': 4,
     'context_relevance_mean': 0.6,
     'context_relevance_std': 0.406201920231798,
     'context_relevance_min': 0.0,
     'context_relevance_max': 1.0,
     'context_relevance_count': 5,
     'response_groundedness_mean': 1.0,
     'response_groundedness_std': 0.0,
     'response_groundedness_min': 1.0,
     'response_groundedness_max': 1.0,
     'response_groundedness_count': 5,
     'answer_accuracy_mean': 0.4,
     'answer_accuracy_std': 0.33911649915626346,
     'answer_accuracy_min': 0.0,
     'answer_accuracy_max': 0.75,
     'answer_accuracy_count': 5,
     'overall_mean': 0.5400800593438563}



## Оптимизируем оценку


```python
def prepare_rag_evaluation_dataset(
    pipeline,  
    test_cases: List[Dict[str, str]],
    top_k: int = 5,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Подготавливает датасет для оценки RAGAS.
    
    Args:
        pipeline: Инициализированный RAG пайплайн
        test_cases: Список тестовых кейсов с query и ground_truth
        top_k: Количество контекстов для извлечения
        save_path: Путь для сохранения датасета (optional)
    
    Returns:
        DataFrame с данными для оценки
    """
    data = []
    
    print(f"🔄 Генерация ответов для {len(test_cases)} тестовых запросов...")
    
    for idx, test_case in enumerate(tqdm(test_cases, desc="Обработка запросов")):
        query = test_case["query"]
        ground_truth = test_case.get("ground_truth", "")
        metadata = test_case.get("metadata", {})
        
        try:
            # Получаем ответ и контексты от RAG системы
            result = pipeline.query(
                query=query,
                top_k=top_k,
                return_details=True,
                display_result=False
            )
            
            # Формируем запись
            row = {
                'idx': idx,
                'query': query,
                'answer': result["answer"],
                'contexts': '|||'.join(result["contexts"]),  # Объединяем контексты
                'ground_truth': ground_truth,
                'num_contexts': len(result["contexts"]),
                'timestamp': datetime.now().isoformat(),
            }
            
            # Добавляем метаданные
            row.update(metadata)
            
            data.append(row)
            
            print(f"✅ [{idx+1}/{len(test_cases)}] Обработан: {query[:50]}...")
            
        except Exception as e:
            print(f"❌ Ошибка при обработке запроса '{query[:50]}...': {e}")
            
            # Добавляем пустую запись
            data.append({
                'idx': idx,
                'query': query,
                'answer': "",
                'contexts': "",
                'ground_truth': ground_truth,
                'num_contexts': 0,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
            })
    
    # Создаем DataFrame
    df = pd.DataFrame(data)
    
    # Сохраняем, если указан путь
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем в Parquet (эффективнее для больших данных)
        if save_path.suffix == '.parquet':
            df.to_parquet(save_path, index=False, compression='gzip')
        # Или в CSV
        elif save_path.suffix == '.csv':
            df.to_csv(save_path, index=False, encoding='utf-8')
        # JSON
        elif save_path.suffix == '.json':
            df.to_json(save_path, orient='records', lines=True, force_ascii=False)
        else:
            # По умолчанию Parquet
            save_path = save_path.with_suffix('.parquet')
            df.to_parquet(save_path, index=False, compression='gzip')
        
        print(f"\n💾 Датасет сохранен: {save_path}")
        print(f"   Формат: {save_path.suffix}")
        print(f"   Размер: {save_path.stat().st_size / 1024:.1f} KB")
    
    print(f"\n✅ Датасет подготовлен:")
    print(f"   • Всего записей: {len(df)}")
    print(f"   • Успешных: {df['answer'].notna().sum()}")
    print(f"   • С ground truth: {df['ground_truth'].notna().sum()}")
    print(f"   • С ошибками: {df.get('error', pd.Series()).notna().sum()}")
    
    return df
```


```python
def load_evaluation_dataset(filepath: str) -> pd.DataFrame:
    """
    Загружает датасет для оценки из файла.
    
    Args:
        filepath: Путь к файлу датасета
    
    Returns:
        DataFrame с данными
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Файл не найден: {filepath}")
    
    print(f"📂 Загрузка датасета: {filepath}")
    
    # Определяем формат и загружаем
    if filepath.suffix == '.parquet':
        df = pd.read_parquet(filepath)
    elif filepath.suffix == '.csv':
        df = pd.read_csv(filepath, encoding='utf-8')
    elif filepath.suffix == '.json':
        df = pd.read_json(filepath, orient='records', lines=True)
    else:
        raise ValueError(f"Неподдерживаемый формат: {filepath.suffix}")
    
    print(f"✅ Датасет загружен:")
    print(f"   • Записей: {len(df)}")
    print(f"   • Колонок: {len(df.columns)}")
    print(f"   • Размер: {filepath.stat().st_size / 1024:.1f} KB")
    
    return df
```


```python
def dataset_to_ragas_format(df: pd.DataFrame) -> Tuple[List[str], List[str], List[List[str]], List[str]]:
    """
    Конвертирует DataFrame в формат для RAGAS оценки.
    
    Args:
        df: DataFrame с данными
    
    Returns:
        Кортеж (queries, answers, contexts_list, ground_truths)
    """
    queries = df['query'].tolist()
    answers = df['answer'].tolist()
    
    # Разделяем контексты
    contexts_list = []
    for contexts_str in df['contexts']:
        if pd.notna(contexts_str) and contexts_str:
            contexts = contexts_str.split('|||')
            contexts_list.append(contexts)
        else:
            contexts_list.append([])
    
    ground_truths = df['ground_truth'].fillna("").tolist()
    
    return queries, answers, contexts_list, ground_truths
```


```python
def evaluate_rag_with_ragas(
    dataset: Optional[Union[pd.DataFrame, str]] = None,
    pipeline = None,
    test_cases: List[Dict[str, str]] = None,
    judge_model: str = "openai/gpt-5-mini",
    judge_api_key: str = None,
    judge_api_base: str = "https://api.vsegpt.ru/v1",
    embedding_model: str = "emb-openai/text-embedding-3-large",
    top_k: int = 5,
    metrics: List[str] = None,
    max_tokens: int = 32000,  # ✅ Добавлен параметр
    save_dataset_path: Optional[str] = None,
    enable_timing: bool = True,  # ✅ Новый параметр
    exclude_slow_metrics: bool = False,  # ✅ Новый параметр
    max_metric_time: float = 15.0  # ✅ Новый параметр
) -> Dict[str, Any]:
    """
    Полная оценка RAG системы с использованием RAGAS 0.4+.
    
    Args:
        dataset: DataFrame или путь к файлу с готовым датасетом
        pipeline: RAG пайплайн (если нужно сгенерировать датасет)
        test_cases: Тестовые кейсы (если нужно сгенерировать датасет)
        judge_model: Модель для оценки
        judge_api_key: API ключ для Judge LLM
        judge_api_base: Base URL для Judge LLM API
        embedding_model: Модель embeddings
        top_k: Количество контекстов
        metrics: Список метрик (None = автоопределение)
        max_tokens: Максимальное количество токенов для LLM
        save_dataset_path: Путь для сохранения сгенерированного датасета
        enable_timing: Включить анализ времени выполнения
        exclude_slow_metrics: Автоматически исключить медленные метрики
        max_metric_time: Максимальное время для метрики (если exclude_slow_metrics=True)
    
    Returns:
        Результаты оценки
    """
    if not RAGAS_AVAILABLE:
        raise ImportError("RAGAS не установлена. Установите: pip install ragas --upgrade")
    
    # 1. Получаем или создаем датасет
    print("=" * 80)
    print("📊 ШАГ 1: Подготовка датасета")
    print("=" * 80)
    
    if dataset is not None:
        # Загружаем существующий датасет
        if isinstance(dataset, str):
            df = load_evaluation_dataset(dataset)
        else:
            df = dataset
            print(f"✅ Использован предоставленный DataFrame ({len(df)} записей)")
    else:
        # Генерируем новый датасет
        if pipeline is None or test_cases is None:
            raise ValueError("Необходимо указать либо dataset, либо pipeline+test_cases")
        
        df = prepare_rag_evaluation_dataset(
            pipeline=pipeline,
            test_cases=test_cases,
            top_k=top_k,
            save_path=save_dataset_path
        )
    
    # Конвертируем в формат RAGAS
    queries, answers, contexts_list, ground_truths = dataset_to_ragas_format(df)
    
    # 2. Создаем оценщика (Judge LLM)
    print("\n" + "=" * 80)
    print("🤖 ШАГ 2: Инициализация Judge LLM для оценки")
    print("=" * 80)
    
    if judge_api_key is None:
        judge_api_key = os.getenv("OPENAI_API_KEY")
        if not judge_api_key:
            raise ValueError("Необходимо указать judge_api_key или установить OPENAI_API_KEY")
    
    # ✅ Передаем max_tokens
    evaluation_client = UniversalLLMClient(
        model=judge_model,
        api_key=judge_api_key,
        api_base=judge_api_base,
        temperature=0.0,
        max_tokens=max_tokens  # ✅ Теперь используется
    )
    
    # Определяем метрики
    if metrics is None:
        has_ground_truth = any(gt and gt.strip() for gt in ground_truths)
        
        if has_ground_truth:
            metrics = [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
                "context_relevance",
                "response_groundedness",
                "answer_accuracy"
            ]
        else:
            metrics = [
                "faithfulness",
                "context_relevance",
                "response_groundedness"
            ]
    
    # ✅ Создаем evaluator с поддержкой timing
    evaluator = UniversalRAGEvaluator(
        judge_llm_client=evaluation_client,
        embedding_model=embedding_model,
        metrics=metrics,
        enable_timing=enable_timing  # ✅ Включаем анализ времени
    )
    
    # 3. Запускаем оценку
    print("\n" + "=" * 80)
    print("🚀 ШАГ 3: Запуск оценки RAGAS")
    print("=" * 80)
    print(f"⚙️  Метрики: {', '.join(metrics)}")
    print(f"⚙️  max_tokens: {max_tokens}")
    print(f"⏰ Это может занять несколько минут (LLM вызовы для каждой метрики)...\n")
    
    # Выполняем оценку
    results = evaluator.evaluate(
        queries=queries,
        answers=answers,
        contexts=contexts_list,
        ground_truths=ground_truths,
        show_progress=True
    )
    
    # 4. Анализ производительности
    if enable_timing:
        print("\n" + "=" * 80)
        print("⏱️ ШАГ 3.5: Анализ производительности метрик")
        print("=" * 80)
        
        evaluator.display_timing_analysis()
        
        # Рекомендации по оптимизации
        if exclude_slow_metrics:
            recommended_metrics = evaluator.get_recommended_metrics(
                max_avg_time=max_metric_time,
                exclude_redundant=True
            )
            
            print(f"\n💡 Рекомендация: используйте метрики {recommended_metrics}")
            print(f"   для ускорения оценки в {len(metrics)/len(recommended_metrics):.1f}x раз")
    
    # 5. Отображаем результаты
    print("\n" + "=" * 80)
    print("📊 ШАГ 4: Результаты оценки")
    print("=" * 80)
    
    evaluator.display_results(max_examples=len(queries))
    
    # 6. Сохраняем результаты
    print("\n" + "=" * 80)
    print("💾 ШАГ 5: Сохранение результатов")
    print("=" * 80)
    
    results_path = evaluator.save_results()
    print(f"✅ Детальные результаты сохранены: {results_path}")
    
    # Сохраняем агрегированные метрики
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = Path("results") / f"ragas_metrics_{timestamp}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ Агрегированные метрики сохранены: {metrics_path}")
    
    # ✅ Сохраняем статистику по времени
    if enable_timing:
        timing_df = evaluator.get_timing_statistics()
        timing_path = Path("results") / f"ragas_timing_{timestamp}.csv"
        timing_df.to_csv(timing_path, index=False)
        print(f"✅ Статистика производительности: {timing_path}")
    
    return {
        "aggregated_metrics": results,
        "detailed_results": evaluator.get_detailed_results(),
        "evaluator": evaluator,
        "dataset": df,
        "timing_statistics": evaluator.get_timing_statistics() if enable_timing else None
    }
```

### Создаем датасет и сохраняем


```python
dataset_df = prepare_rag_evaluation_dataset(
    pipeline=pipeline,
    test_cases=test_cases,
    top_k=5,
    save_path="datasets/rag_evaluation_dataset.parquet"
)
```

### Оценка с существующим датасетом


```python
results = evaluate_rag_with_ragas(
    dataset="datasets/rag_evaluation_dataset.parquet",  # Загрузка датасета
    judge_model="openai/gpt-5-mini",
    judge_api_key="sk-or-vv-xxx",
    judge_api_base="https://api.vsegpt.ru/v1",
    embedding_model="emb-openai/text-embedding-3-large",
    max_tokens=32000,
    enable_timing=True,  # Анализ производительности
    exclude_slow_metrics=False,  # Пока все метрики
    metrics=None  # Автоопределение
)
```


```python
recommended = results['evaluator'].get_recommended_metrics(
    max_avg_time=10.0,
    exclude_redundant=True
)
```

    2026-01-09 19:49:33,506 - __main__ - INFO - ✅ Рекомендуемые метрики: 
    

### Быстрая оценка с оптимизированными метриками


```python
metrics = [
    # "faithfulness",
    "answer_relevancy",
    # "context_precision",
    "context_recall",
    "context_relevance",
    # "response_groundedness",
    "answer_accuracy"
]
```


```python
start_time = time.time()
results_fast = evaluate_rag_with_ragas(
    dataset="datasets/rag_evaluation_dataset.parquet",
    judge_model="openai/gpt-5-mini",
    judge_api_key="sk-or-vv-xxx",
    max_tokens=32000,
    metrics=metrics, #recommended,  # Только быстрые метрики
    enable_timing=True
)
elapsed_time = time.time() - start_time
print(f"Время исполнения: {elapsed_time} сек.")
```

## Загрузка датасета


```python
def load_test_cases_from_file(filepath: str) -> List[Dict[str, str]]:
    """
    Загружает тестовые кейсы из сохраненного файла.
    
    Args:
        filepath: Путь к файлу (CSV, Parquet или JSON)
    
    Returns:
        Список тестовых кейсов
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"❌ Файл не найден: {filepath}")
    
    print(f"📂 Загрузка тестовых кейсов: {filepath}")
    
    # JSON - прямая загрузка
    if filepath.suffix == '.json':
        with open(filepath, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
        print(f"✅ Загружено {len(test_cases)} тестовых кейсов из JSON")
        return test_cases
    
    # CSV или Parquet - конвертируем из DataFrame
    if filepath.suffix == '.csv':
        df = pd.read_csv(filepath, encoding='utf-8')
    elif filepath.suffix == '.parquet':
        df = pd.read_parquet(filepath)
    else:
        raise ValueError(f"❌ Неподдерживаемый формат: {filepath.suffix}")
    
    # Конвертируем в список словарей
    test_cases = []
    for _, row in df.iterrows():
        test_case = {
            "query": row["query"],
            "ground_truth": row["ground_truth"],
            "metadata": {
                "object_id": row.get("object_id", ""),
                "num_axioms": int(row.get("num_axioms", 0)),
            }
        }
        
        # Добавляем аксиомы если есть
        if "axioms" in row and pd.notna(row["axioms"]):
            axioms_str = str(row["axioms"])
            test_case["metadata"]["axioms"] = axioms_str.split("\n") if axioms_str else []
        
        test_cases.append(test_case)
    
    print(f"✅ Загружено {len(test_cases)} тестовых кейсов")
    return test_cases
```


```python
loaded_test_cases = load_test_cases_from_file("./datasets/rag_test_cases_20260109_222314.json")

print(f"✅ Загружено {len(loaded_test_cases)} тестовых кейсов")
```

    📂 Загрузка тестовых кейсов: datasets\rag_test_cases_20260109_222314.json
    ✅ Загружено 198 тестовых кейсов из JSON
    ✅ Загружено 198 тестовых кейсов
    

### Генерация данных для оценки RAGAS


```python
# Используем созданные test_cases для генерации ответов от RAG системы
dataset_df = prepare_rag_evaluation_dataset(
    pipeline=pipeline,  # Ваш RAG pipeline
    test_cases=loaded_test_cases,
    top_k=5,
    save_path="./datasets/rag_evaluation_dataset_full.parquet"
)
```

### Оценка RAGAS


```python
metrics = [
    # "faithfulness",
    "answer_relevancy",
    # "context_precision",
    "context_recall",
    "context_relevance",
    # "response_groundedness",
    "answer_accuracy"
]
```


```python
evaluation_results = evaluate_rag_with_ragas(
    dataset=dataset_df,
    judge_model="openai/gpt-5-mini",
    judge_api_key="sk-or-vv-xxx",
    judge_api_base="https://api.vsegpt.ru/v1",
    max_tokens=32000,
    enable_timing=True,
    metrics=metrics
)
```

# RAGAS метрики для GraphRAG

### Адаптер GraphRAG для RAGAS


```python
class GraphRAGAdapter:
    """
    Адаптер для интеграции Microsoft GraphRAG с RAGAS.
    
    Преобразует структурированные данные GraphRAG (сущности, связи)
    в формат, совместимый с метриками RAGAS.
    """
    
    def __init__(self, search_engine):
        """
        Args:
            search_engine: Экземпляр LocalSearch из GraphRAG
        """
        self.search_engine = search_engine
        self.logger = logging.getLogger(__name__)
    
    async def query(self, 
                   question: str,
                   return_details: bool = True) -> Dict[str, Any]:
        """
        Выполняет запрос к GraphRAG и возвращает результат в формате для RAGAS.
        
        Args:
            question: Вопрос пользователя
            return_details: Возвращать ли детали контекста
        
        Returns:
            Словарь с ответом и контекстом
        """
        start_time = time.time()
        
        # Выполняем поиск в GraphRAG
        result = await self.search_engine.search(question)
        
        elapsed_time = time.time() - start_time
        
        # Формируем ответ
        answer = result.response
        
        # Извлекаем и форматируем контекст
        contexts = []
        context_data = {}
        
        if return_details and hasattr(result, 'context_data'):
            contexts, context_data = self._extract_contexts(result.context_data)
        
        return {
            "answer": answer,
            "contexts": contexts,
            "context_data": context_data,
            "elapsed_time": elapsed_time,
            "raw_result": result
        }
    
    def _extract_contexts(self, context_data: Dict) -> Tuple[List[str], Dict]:
        """
        Извлекает и форматирует контекст из GraphRAG context_data.
        
        GraphRAG предоставляет:
        - entities: DataFrame с релевантными сущностями
        - relationships: DataFrame с релевантными связями
        - reports: Отчеты сообществ
        - sources: Текстовые единицы
        
        Преобразуем их в список строк (contexts) для RAGAS.
        
        Args:
            context_data: Словарь с контекстными данными от GraphRAG
        
        Returns:
            Кортеж (список контекстных строк, словарь метаданных)
        """
        contexts = []
        metadata = {
            "num_entities": 0,
            "num_relationships": 0,
            "num_reports": 0,
            "num_sources": 0
        }
        
        # 1. Извлекаем сущности
        if "entities" in context_data and context_data["entities"] is not None:
            entities_df = context_data["entities"]
            
            if not entities_df.empty:
                metadata["num_entities"] = len(entities_df)
                
                # Форматируем каждую сущность
                for _, entity in entities_df.iterrows():
                    entity_context = self._format_entity(entity)
                    if entity_context:
                        contexts.append(entity_context)
        
        # 2. Извлекаем связи
        if "relationships" in context_data and context_data["relationships"] is not None:
            relationships_df = context_data["relationships"]
            
            if not relationships_df.empty:
                metadata["num_relationships"] = len(relationships_df)
                
                # Форматируем связи
                for _, rel in relationships_df.iterrows():
                    rel_context = self._format_relationship(rel)
                    if rel_context:
                        contexts.append(rel_context)
        
        # 3. Извлекаем отчеты сообществ
        if "reports" in context_data and context_data["reports"] is not None:
            reports_df = context_data["reports"]
            
            if not reports_df.empty:
                metadata["num_reports"] = len(reports_df)
                
                for _, report in reports_df.iterrows():
                    report_context = self._format_report(report)
                    if report_context:
                        contexts.append(report_context)
        
        # 4. Извлекаем текстовые единицы (sources)
        if "sources" in context_data and context_data["sources"] is not None:
            sources_df = context_data["sources"]
            
            if not sources_df.empty:
                metadata["num_sources"] = len(sources_df)
                
                for _, source in sources_df.iterrows():
                    source_context = self._format_source(source)
                    if source_context:
                        contexts.append(source_context)
        
        self.logger.info(
            f"📊 Извлечено контекстов: {len(contexts)} "
            f"(entities: {metadata['num_entities']}, "
            f"relationships: {metadata['num_relationships']}, "
            f"reports: {metadata['num_reports']}, "
            f"sources: {metadata['num_sources']})"
        )
        
        return contexts, metadata
    
    def _format_entity(self, entity: pd.Series) -> str:
        """
        Форматирует сущность в текстовую строку.
        
        Типичные поля в GraphRAG entity:
        - title/name: название сущности
        - type: тип сущности
        - description: описание
        - rank: важность
        """
        parts = []
        
        # Название
        title = entity.get('title') or entity.get('name', 'Unknown Entity')
        parts.append(f"Entity: {title}")
        
        # Тип
        if 'type' in entity and pd.notna(entity['type']):
            parts.append(f"Type: {entity['type']}")
        
        # Описание
        if 'description' in entity and pd.notna(entity['description']):
            desc = str(entity['description']).strip()
            if desc:
                parts.append(f"Description: {desc}")
        
        # Ранг/важность
        if 'rank' in entity and pd.notna(entity['rank']):
            parts.append(f"Rank: {entity['rank']}")
        
        return " | ".join(parts) if parts else ""
    
    def _format_relationship(self, relationship: pd.Series) -> str:
        """
        Форматирует связь в текстовую строку.
        
        Типичные поля:
        - source: исходная сущность
        - target: целевая сущность
        - description: описание связи
        - weight: вес связи
        """
        parts = []
        
        source = relationship.get('source', 'Unknown')
        target = relationship.get('target', 'Unknown')
        
        parts.append(f"Relationship: {source} -> {target}")
        
        # Описание связи
        if 'description' in relationship and pd.notna(relationship['description']):
            desc = str(relationship['description']).strip()
            if desc:
                parts.append(f"Description: {desc}")
        
        # Вес
        if 'weight' in relationship and pd.notna(relationship['weight']):
            parts.append(f"Weight: {relationship['weight']}")
        
        return " | ".join(parts) if parts else ""
    
    def _format_report(self, report: pd.Series) -> str:
        """
        Форматирует отчет сообщества.
        
        Типичные поля:
        - title: название отчета
        - summary: краткое содержание
        - full_content: полное содержание
        - rank: важность
        """
        parts = []
        
        # Заголовок
        if 'title' in report and pd.notna(report['title']):
            parts.append(f"Community Report: {report['title']}")
        
        # Содержание (используем summary если есть, иначе full_content)
        content = None
        if 'summary' in report and pd.notna(report['summary']):
            content = str(report['summary']).strip()
        elif 'full_content' in report and pd.notna(report['full_content']):
            content = str(report['full_content']).strip()
        
        if content:
            # Ограничиваем длину для читабельности
            if len(content) > 500:
                content = content[:497] + "..."
            parts.append(f"Content: {content}")
        
        return " | ".join(parts) if parts else ""
    
    def _format_source(self, source: pd.Series) -> str:
        """
        Форматирует текстовую единицу (source).
        
        Типичные поля:
        - text: текст
        - id: идентификатор
        """
        if 'text' in source and pd.notna(source['text']):
            text = str(source['text']).strip()
            
            # Добавляем ID если есть
            if 'id' in source and pd.notna(source['id']):
                return f"Source [{source['id']}]: {text}"
            else:
                return f"Source: {text}"
        
        return ""
```


```python
def sync_query_graphrag(adapter: GraphRAGAdapter, question: str) -> Dict[str, Any]:
    """
    Синхронная обертка для асинхронного query GraphRAG.
    
    Args:
        adapter: GraphRAGAdapter экземпляр
        question: Вопрос
    
    Returns:
        Результат запроса
    """
    try:
        # Применяем nest_asyncio если нужно (для Jupyter)
        import nest_asyncio
        nest_asyncio.apply()
    except:
        pass
    
    # Выполняем асинхронный запрос
    try:
        loop = asyncio.get_running_loop()
        result = loop.run_until_complete(adapter.query(question))
    except RuntimeError:
        result = asyncio.run(adapter.query(question))
    
    return result
```

### Подготовка датасета для GraphRAG


```python
def prepare_graphrag_evaluation_dataset(
    graphrag_adapter: GraphRAGAdapter,
    test_cases: List[Dict[str, str]],
    save_path: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Подготавливает датасет для оценки GraphRAG с RAGAS.
    
    Args:
        graphrag_adapter: Адаптер GraphRAG
        test_cases: Список тестовых кейсов с query и ground_truth
        save_path: Путь для сохранения датасета
        verbose: Подробный вывод
    
    Returns:
        DataFrame с данными для оценки
    """
    data = []
    
    print(f"🔄 Генерация ответов GraphRAG для {len(test_cases)} запросов...")
    
    iterator = tqdm(test_cases, desc="Обработка запросов") if verbose else test_cases
    
    for idx, test_case in enumerate(iterator):
        query = test_case["query"]
        ground_truth = test_case.get("ground_truth", "")
        metadata = test_case.get("metadata", {})
        
        try:
            # Получаем ответ от GraphRAG
            result = sync_query_graphrag(graphrag_adapter, query)
            
            # Формируем запись
            row = {
                'idx': idx,
                'query': query,
                'answer': result["answer"],
                'contexts': '|||'.join(result["contexts"]),  # Объединяем контексты
                'ground_truth': ground_truth,
                'num_contexts': len(result["contexts"]),
                'num_entities': result["context_data"].get("num_entities", 0),
                'num_relationships': result["context_data"].get("num_relationships", 0),
                'num_reports': result["context_data"].get("num_reports", 0),
                'num_sources': result["context_data"].get("num_sources", 0),
                'elapsed_time': result["elapsed_time"],
                'timestamp': datetime.now().isoformat(),
            }
            
            # Добавляем исходные метаданные
            row.update(metadata)
            
            data.append(row)
            
            if verbose:
                tqdm.write(
                    f"✅ [{idx+1}/{len(test_cases)}] {query[:50]}... "
                    f"(контекстов: {row['num_contexts']}, время: {row['elapsed_time']:.1f}с)"
                )
            
        except Exception as e:
            print(f"❌ Ошибка при обработке запроса '{query[:50]}...': {e}")
            
            # Добавляем пустую запись
            data.append({
                'idx': idx,
                'query': query,
                'answer': "",
                'contexts': "",
                'ground_truth': ground_truth,
                'num_contexts': 0,
                'num_entities': 0,
                'num_relationships': 0,
                'num_reports': 0,
                'num_sources': 0,
                'elapsed_time': 0,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
            })
    
    # Создаем DataFrame
    df = pd.DataFrame(data)
    
    # Сохраняем если указан путь
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_path.suffix == '.parquet':
            df.to_parquet(save_path, index=False, compression='gzip')
        elif save_path.suffix == '.csv':
            df.to_csv(save_path, index=False, encoding='utf-8')
        else:
            save_path = save_path.with_suffix('.parquet')
            df.to_parquet(save_path, index=False, compression='gzip')
        
        print(f"\n💾 Датасет сохранен: {save_path}")
        print(f"   Размер: {save_path.stat().st_size / 1024:.1f} KB")
    
    # Статистика
    print(f"\n✅ Датасет подготовлен:")
    print(f"   • Всего записей: {len(df)}")
    print(f"   • Успешных: {df['answer'].notna().sum()}")
    print(f"   • С ground truth: {df['ground_truth'].notna().sum()}")
    print(f"   • С ошибками: {df.get('error', pd.Series()).notna().sum()}")
    print(f"   • Среднее контекстов: {df['num_contexts'].mean():.1f}")
    print(f"   • Среднее время: {df['elapsed_time'].mean():.1f}с")
    
    return df
```

### Анализ совместимости метрик RAGAS с GraphRAG


```python
def analyze_graphrag_ragas_compatibility(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Анализирует совместимость метрик RAGAS с данными GraphRAG.
    
    Args:
        df: DataFrame с данными GraphRAG
    
    Returns:
        Отчет о совместимости
    """
    from IPython.display import display, HTML
    
    compatibility = {
        "faithfulness": {
            "compatible": True,
            "required_fields": ["answer", "contexts"],
            "notes": "✅ Полностью совместима. Использует контексты из сущностей/связей/отчетов."
        },
        "answer_relevancy": {
            "compatible": True,
            "required_fields": ["query", "answer"],
            "notes": "✅ Полностью совместима. Не требует контекстов, использует embeddings."
        },
        "context_precision": {
            "compatible": True,
            "required_fields": ["query", "contexts", "ground_truth"],
            "notes": "✅ Совместима при наличии ground_truth. Оценивает релевантность контекстов."
        },
        "context_recall": {
            "compatible": True,
            "required_fields": ["contexts", "ground_truth"],
            "notes": "✅ Совместима при наличии ground_truth. Проверяет полноту контекста."
        },
        "context_relevance": {
            "compatible": True,
            "required_fields": ["query", "contexts"],
            "notes": "✅ Полностью совместима. Оценивает релевантность контекстов запросу."
        },
        "response_groundedness": {
            "compatible": True,
            "required_fields": ["answer", "contexts"],
            "notes": "✅ Полностью совместима. Проверяет обоснованность ответа контекстом."
        },
        "answer_accuracy": {
            "compatible": True,
            "required_fields": ["query", "answer", "ground_truth"],
            "notes": "✅ Совместима при наличии ground_truth. Сравнивает ответ с эталоном."
        }
    }
    
    # Проверяем доступность данных
    has_contexts = df['contexts'].notna().sum() > 0
    has_ground_truth = df['ground_truth'].notna().sum() > 0
    avg_contexts = df['num_contexts'].mean()
    
    # Рекомендуемые метрики
    recommended_metrics = []
    
    if has_contexts:
        recommended_metrics.extend([
            "faithfulness",
            "context_relevance",
            "response_groundedness"
        ])
    
    if has_contexts and has_ground_truth:
        recommended_metrics.extend([
            "context_precision",
            "context_recall"
        ])
    
    if has_ground_truth:
        recommended_metrics.append("answer_accuracy")
    
    # Always available
    recommended_metrics.append("answer_relevancy")
    
    # Удаляем дубликаты
    recommended_metrics = list(dict.fromkeys(recommended_metrics))
    
    # HTML отчет
    html = f"""
    <div style="border: 2px solid #2196F3; border-radius: 12px; padding: 20px; margin: 15px 0;
                background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);">
        <h2 style="color: #1565C0; margin-top: 0;">🔍 Анализ совместимости RAGAS с GraphRAG</h2>
        
        <div style="background: white; padding: 15px; border-radius: 8px; margin: 15px 0;">
            <h3 style="color: #1565C0;">📊 Статистика данных</h3>
            <ul style="line-height: 1.8;">
                <li>📝 Всего записей: <strong>{len(df)}</strong></li>
                <li>📦 Записей с контекстами: <strong>{df['contexts'].notna().sum()}</strong> ({df['contexts'].notna().sum()/len(df)*100:.1f}%)</li>
                <li>✅ Записей с ground truth: <strong>{df['ground_truth'].notna().sum()}</strong> ({df['ground_truth'].notna().sum()/len(df)*100:.1f}%)</li>
                <li>📊 Среднее контекстов на запрос: <strong>{avg_contexts:.1f}</strong></li>
                <li>🔗 Среднее сущностей: <strong>{df['num_entities'].mean():.1f}</strong></li>
                <li>🔗 Среднее связей: <strong>{df['num_relationships'].mean():.1f}</strong></li>
            </ul>
        </div>
        
        <div style="background: white; padding: 15px; border-radius: 8px; margin: 15px 0;">
            <h3 style="color: #1565C0;">✅ Рекомендуемые метрики RAGAS</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
    """
    
    for metric in recommended_metrics:
        info = compatibility.get(metric, {})
        html += f"""
                <div style="background: #E8F5E9; padding: 10px; border-radius: 5px; border-left: 4px solid #4CAF50;">
                    <strong style="color: #2E7D32;">{metric}</strong>
                    <div style="font-size: 0.85em; color: #666; margin-top: 5px;">
                        {info.get('notes', '')}
                    </div>
                </div>
        """
    
    html += """
            </div>
        </div>
        
        <div style="background: #FFF9C4; padding: 15px; border-radius: 8px; margin: 15px 0;">
            <h3 style="color: #F57F17;">⚠️ Особенности GraphRAG</h3>
            <ul style="line-height: 1.8; color: #666;">
                <li>GraphRAG использует <strong>структурированный контекст</strong> (сущности + связи + отчеты)</li>
                <li>Контексты могут быть <strong>более семантически богатыми</strong>, чем простые текстовые чанки</li>
                <li>Метрики <strong>context_precision</strong> и <strong>context_recall</strong> оценивают качество графовой структуры</li>
                <li><strong>answer_relevancy</strong> использует embeddings и не зависит от типа контекста</li>
            </ul>
        </div>
    </div>
    """
    
    display(HTML(html))
    
    return {
        "compatibility": compatibility,
        "recommended_metrics": recommended_metrics,
        "has_contexts": has_contexts,
        "has_ground_truth": has_ground_truth,
        "stats": {
            "total_records": len(df),
            "avg_contexts": avg_contexts,
            "avg_entities": df['num_entities'].mean(),
            "avg_relationships": df['num_relationships'].mean()
        }
    }
```

### Оценка GraphRAG с RAGAS


```python
def evaluate_graphrag_with_ragas(
    dataset: Optional[Union[pd.DataFrame, str]] = None,
    graphrag_adapter: GraphRAGAdapter = None,
    test_cases: List[Dict[str, str]] = None,
    judge_model: str = "openai/gpt-5-mini",
    judge_api_key: str = None,
    judge_api_base: str = "https://api.vsegpt.ru/v1",
    embedding_model: str = "emb-openai/text-embedding-3-large",
    metrics: List[str] = None,
    max_tokens: int = 32000,
    save_dataset_path: Optional[str] = None,
    enable_timing: bool = True,
    analyze_compatibility: bool = True
) -> Dict[str, Any]:
    """
    Полная оценка GraphRAG системы с использованием RAGAS 0.4+.
    
    Args:
        dataset: DataFrame или путь к файлу с готовым датасетом
        graphrag_adapter: GraphRAGAdapter (если нужно сгенерировать датасет)
        test_cases: Тестовые кейсы (если нужно сгенерировать датасет)
        judge_model: Модель для оценки
        judge_api_key: API ключ для Judge LLM
        judge_api_base: Base URL для Judge LLM API
        embedding_model: Модель embeddings
        metrics: Список метрик (None = автоопределение)
        max_tokens: Максимальное количество токенов для LLM
        save_dataset_path: Путь для сохранения сгенерированного датасета
        enable_timing: Включить анализ времени выполнения
        analyze_compatibility: Провести анализ совместимости
    
    Returns:
        Результаты оценки
    """
    if not RAGAS_AVAILABLE:
        raise ImportError("RAGAS не установлена. Установите: pip install ragas --upgrade")
    
    # 1. Получаем или создаем датасет
    print("=" * 80)
    print("📊 ШАГ 1: Подготовка датасета GraphRAG")
    print("=" * 80)
    
    if dataset is not None:
        # Загружаем существующий датасет
        if isinstance(dataset, str):
            df = load_evaluation_dataset(dataset)
        else:
            df = dataset
            print(f"✅ Использован предоставленный DataFrame ({len(df)} записей)")
    else:
        # Генерируем новый датасет
        if graphrag_adapter is None or test_cases is None:
            raise ValueError("Необходимо указать либо dataset, либо graphrag_adapter+test_cases")
        
        df = prepare_graphrag_evaluation_dataset(
            graphrag_adapter=graphrag_adapter,
            test_cases=test_cases,
            save_path=save_dataset_path,
            verbose=True
        )
    
    # Анализ совместимости
    if analyze_compatibility:
        print("\n" + "=" * 80)
        print("🔍 ШАГ 1.5: Анализ совместимости RAGAS")
        print("=" * 80)
        
        compatibility_result = analyze_graphrag_ragas_compatibility(df)
        
        # Если метрики не указаны, используем рекомендуемые
        if metrics is None:
            metrics = compatibility_result["recommended_metrics"]
            print(f"\n✅ Автоматически выбраны метрики: {', '.join(metrics)}")
    
    # Конвертируем в формат RAGAS
    queries, answers, contexts_list, ground_truths = dataset_to_ragas_format(df)
    
    # 2. Создаем оценщика (Judge LLM)
    print("\n" + "=" * 80)
    print("🤖 ШАГ 2: Инициализация Judge LLM для оценки")
    print("=" * 80)
    
    if judge_api_key is None:
        judge_api_key = os.getenv("OPENAI_API_KEY")
        if not judge_api_key:
            raise ValueError("Необходимо указать judge_api_key или установить OPENAI_API_KEY")
    
    evaluation_client = UniversalLLMClient(
        model=judge_model,
        api_key=judge_api_key,
        api_base=judge_api_base,
        temperature=0.0,
        max_tokens=max_tokens
    )
    
    # Определяем метрики если не указаны
    if metrics is None:
        has_ground_truth = any(gt and gt.strip() for gt in ground_truths)
        
        if has_ground_truth:
            metrics = [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
                "context_relevance",
                "response_groundedness",
                "answer_accuracy"
            ]
        else:
            metrics = [
                "faithfulness",
                "context_relevance",
                "response_groundedness",
                "answer_relevancy"
            ]
    
    evaluator = UniversalRAGEvaluator(
        judge_llm_client=evaluation_client,
        embedding_model=embedding_model,
        metrics=metrics,
        enable_timing=enable_timing
    )
    
    # 3. Запускаем оценку
    print("\n" + "=" * 80)
    print("🚀 ШАГ 3: Запуск оценки RAGAS для GraphRAG")
    print("=" * 80)
    print(f"⚙️  Метрики: {', '.join(metrics)}")
    print(f"⚙️  max_tokens: {max_tokens}")
    print(f"⏰ Это может занять несколько минут...\n")
    
    results = evaluator.evaluate(
        queries=queries,
        answers=answers,
        contexts=contexts_list,
        ground_truths=ground_truths,
        show_progress=True
    )
    
    # 4. Анализ производительности
    if enable_timing:
        print("\n" + "=" * 80)
        print("⏱️ ШАГ 3.5: Анализ производительности метрик")
        print("=" * 80)
        
        evaluator.display_timing_analysis()
    
    # 5. Отображаем результаты
    print("\n" + "=" * 80)
    print("📊 ШАГ 4: Результаты оценки GraphRAG")
    print("=" * 80)
    
    evaluator.display_results(max_examples=min(10, len(queries)))
    
    # 6. Сохраняем результаты
    print("\n" + "=" * 80)
    print("💾 ШАГ 5: Сохранение результатов")
    print("=" * 80)
    
    results_path = evaluator.save_results()
    print(f"✅ Детальные результаты сохранены: {results_path}")
    
    # Сохраняем агрегированные метрики
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = Path("results") / f"graphrag_ragas_metrics_{timestamp}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ Агрегированные метрики: {metrics_path}")
    
    # Сохраняем статистику по времени
    if enable_timing:
        timing_df = evaluator.get_timing_statistics()
        timing_path = Path("results") / f"graphrag_ragas_timing_{timestamp}.csv"
        timing_df.to_csv(timing_path, index=False)
        print(f"✅ Статистика производительности: {timing_path}")
    
    # Сравнительная статистика GraphRAG
    print("\n" + "=" * 80)
    print("📊 Статистика GraphRAG")
    print("=" * 80)
    print(f"   • Среднее контекстов на запрос: {df['num_contexts'].mean():.1f}")
    print(f"   • Среднее сущностей: {df['num_entities'].mean():.1f}")
    print(f"   • Среднее связей: {df['num_relationships'].mean():.1f}")
    print(f"   • Среднее отчетов: {df['num_reports'].mean():.1f}")
    print(f"   • Среднее время запроса: {df['elapsed_time'].mean():.1f}с")
    
    return {
        "aggregated_metrics": results,
        "detailed_results": evaluator.get_detailed_results(),
        "evaluator": evaluator,
        "dataset": df,
        "timing_statistics": evaluator.get_timing_statistics() if enable_timing else None,
        "graphrag_stats": {
            "avg_contexts": df['num_contexts'].mean(),
            "avg_entities": df['num_entities'].mean(),
            "avg_relationships": df['num_relationships'].mean(),
            "avg_reports": df['num_reports'].mean(),
            "avg_query_time": df['elapsed_time'].mean()
        }
    }
      
```

### Сравнение RAG vs GraphRAG


```python
def compare_rag_vs_graphrag(
    rag_results: Dict[str, Any],
    graphrag_results: Dict[str, Any],
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Сравнивает результаты оценки обычного RAG и GraphRAG.
    
    Args:
        rag_results: Результаты оценки RAG
        graphrag_results: Результаты оценки GraphRAG
        save_path: Путь для сохранения сравнения
    
    Returns:
        DataFrame со сравнением
    """
    from IPython.display import display, HTML
    
    # Извлекаем метрики
    rag_metrics = rag_results["aggregated_metrics"]
    graphrag_metrics = graphrag_results["aggregated_metrics"]
    
    # Общие метрики
    common_metrics = set(
        [k.replace('_mean', '') for k in rag_metrics.keys() if k.endswith('_mean')]
    ) & set(
        [k.replace('_mean', '') for k in graphrag_metrics.keys() if k.endswith('_mean')]
    )
    
    # Создаем сравнительную таблицу
    comparison_data = []
    
    for metric in sorted(common_metrics):
        rag_mean = rag_metrics.get(f"{metric}_mean", 0)
        graphrag_mean = graphrag_metrics.get(f"{metric}_mean", 0)
        
        difference = graphrag_mean - rag_mean
        improvement = (difference / rag_mean * 100) if rag_mean > 0 else 0
        
        comparison_data.append({
            "Metric": metric,
            "RAG": f"{rag_mean:.3f}",
            "GraphRAG": f"{graphrag_mean:.3f}",
            "Difference": f"{difference:+.3f}",
            "Improvement %": f"{improvement:+.1f}%",
            "Winner": "GraphRAG" if graphrag_mean > rag_mean else ("RAG" if rag_mean > graphrag_mean else "Tie")
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Сохраняем
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_comparison.to_csv(save_path, index=False)
        print(f"💾 Сравнение сохранено: {save_path}")
    
    # HTML визуализация
    html = f"""
    <div style="border: 2px solid #9C27B0; border-radius: 12px; padding: 20px; margin: 15px 0;
                background: linear-gradient(135deg, #F3E5F5 0%, #E1BEE7 100%);">
        <h2 style="color: #6A1B9A; margin-top: 0;">⚖️ Сравнение RAG vs GraphRAG</h2>
        
        <table style="width: 100%; border-collapse: collapse; background: white; margin: 15px 0;">
            <thead>
                <tr style="background: #9C27B0; color: white;">
                    <th style="padding: 12px; text-align: left;">Метрика</th>
                    <th style="padding: 12px; text-align: center;">RAG</th>
                    <th style="padding: 12px; text-align: center;">GraphRAG</th>
                    <th style="padding: 12px; text-align: center;">Разница</th>
                    <th style="padding: 12px; text-align: center;">Улучшение</th>
                    <th style="padding: 12px; text-align: center;">Победитель</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for _, row in df_comparison.iterrows():
        # Определяем цвет строки
        if row['Winner'] == 'GraphRAG':
            bg_color = "#C8E6C9"
            winner_badge = "🏆 GraphRAG"
            winner_color = "#2E7D32"
        elif row['Winner'] == 'RAG':
            bg_color = "#FFE0B2"
            winner_badge = "🏆 RAG"
            winner_color = "#E65100"
        else:
            bg_color = "#E0E0E0"
            winner_badge = "🤝 Tie"
            winner_color = "#666"
        
        html += f"""
            <tr style="background: {bg_color};">
                <td style="padding: 10px; font-weight: bold;">{row['Metric']}</td>
                <td style="padding: 10px; text-align: center;">{row['RAG']}</td>
                <td style="padding: 10px; text-align: center;">{row['GraphRAG']}</td>
                <td style="padding: 10px; text-align: center; font-weight: bold;">{row['Difference']}</td>
                <td style="padding: 10px; text-align: center; font-weight: bold;">{row['Improvement %']}</td>
                <td style="padding: 10px; text-align: center;">
                    <span style="background: {winner_color}; color: white; padding: 4px 8px; border-radius: 12px; font-size: 0.9em;">
                        {winner_badge}
                    </span>
                </td>
            </tr>
        """
    
    html += """
            </tbody>
        </table>
        
        <div style="margin-top: 20px; padding: 15px; background: white; border-radius: 8px;">
            <h3 style="color: #6A1B9A;">📈 Общий результат</h3>
    """
    
    # Подсчет побед
    graphrag_wins = len(df_comparison[df_comparison['Winner'] == 'GraphRAG'])
    rag_wins = len(df_comparison[df_comparison['Winner'] == 'RAG'])
    ties = len(df_comparison[df_comparison['Winner'] == 'Tie'])
    
    html += f"""
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top: 15px;">
                <div style="text-align: center; padding: 15px; background: #C8E6C9; border-radius: 8px;">
                    <div style="font-size: 2em; font-weight: bold; color: #2E7D32;">{graphrag_wins}</div>
                    <div style="color: #666;">GraphRAG побед</div>
                </div>
                <div style="text-align: center; padding: 15px; background: #FFE0B2; border-radius: 8px;">
                    <div style="font-size: 2em; font-weight: bold; color: #E65100;">{rag_wins}</div>
                    <div style="color: #666;">RAG побед</div>
                </div>
                <div style="text-align: center; padding: 15px; background: #E0E0E0; border-radius: 8px;">
                    <div style="font-size: 2em; font-weight: bold; color: #666;">{ties}</div>
                    <div style="color: #666;">Ничьих</div>
                </div>
            </div>
        </div>
    </div>
    """
    
    display(HTML(html))
    
    print("\n📊 Детальная таблица:")
    display(df_comparison)
    
    return df_comparison
```

## Расчёт метрик


```python
import logging
logging.basicConfig(level=logging.INFO)
```


```python
print("🚀 Инициализация GraphRAG...")
```

    🚀 Инициализация GraphRAG...
    


```python
import tiktoken

from graphrag.config.models.vector_store_schema_config import VectorStoreSchemaConfig
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore
```


```python
from graphrag.config.enums import ModelType
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.manager import ModelManager
from graphrag.tokenizer.get_tokenizer import get_tokenizer
```


```python
directory = 'C:\\Users\\glvv2\\vkr\\graph_local3' # os.getcwd()

INPUT_DIR = os.path.join(directory, "output")
LANCEDB_URI = os.path.join(INPUT_DIR, "lancedb")

COMMUNITY_REPORT_TABLE = "community_reports"
ENTITY_TABLE = "entities"
COMMUNITY_TABLE = "communities"
RELATIONSHIP_TABLE = "relationships"
COVARIATE_TABLE = "covariates"
TEXT_UNIT_TABLE = "text_units"
COMMUNITY_LEVEL = 2
```


```python
api_key="sk-or-vv-xxx"

llm_model = "google/gemma-3-27b-it"
embedding_model = "emb-openai/text-embedding-3-large"

chat_config = LanguageModelConfig(
    api_key=api_key,
    api_base="https://api.vsegpt.ru/v1",
    type=ModelType.Chat,
    model_provider="openai",
    model=llm_model,
    max_retries=20,
)

chat_model = ModelManager().get_or_create_chat_model(
    name="local_search",
    model_type=ModelType.Chat,
    config=chat_config,
)

embedding_config = LanguageModelConfig(
    api_key=api_key,
    api_base="https://api.vsegpt.ru/v1",
    type=ModelType.Embedding,
    model_provider="openai",
    model=embedding_model,
    max_retries=20,
)

text_embedder = ModelManager().get_or_create_embedding_model(
    name="local_search_embedding",
    model_type=ModelType.Embedding,
    config=embedding_config,
)

tokenizer = get_tokenizer(chat_config)

description_embedding_store = LanceDBVectorStore(
    vector_store_schema_config=VectorStoreSchemaConfig(
        index_name="default-entity-description"
    )
)
description_embedding_store.connect(db_uri=LANCEDB_URI)

entities = pd.read_parquet(f"{directory}/output/entities.parquet")
communities = pd.read_parquet(f"{directory}/output/communities.parquet")
community_reports = pd.read_parquet(f"{directory}/output/community_reports.parquet")
text_units = pd.read_parquet(f"{directory}/output/text_units.parquet")
relationships = pd.read_parquet(f"{directory}/output/relationships.parquet")

entities_d = read_indexer_entities(entities, communities, None) #, COMMUNITY_LEVEL)
relationships_d = read_indexer_relationships(relationships)
reports_d = read_indexer_reports(community_reports, communities, None) # , COMMUNITY_LEVEL)
text_units_d = read_indexer_text_units(text_units)

context_builder = LocalSearchMixedContext(
    community_reports=reports_d,
    text_units=text_units_d,
    entities=entities_d,
    relationships=relationships_d,
    covariates=None, #covariates,
    entity_text_embeddings=description_embedding_store,
    embedding_vectorstore_key=EntityVectorStoreKey.ID,
    text_embedder=text_embedder,
    tokenizer=tokenizer,
)

local_context_params = {
    "text_unit_prop": 0.5,
    "community_prop": 0.1,
    "conversation_history_max_turns": 5,
    "conversation_history_user_turns_only": True,
    "top_k_mapped_entities": 10,
    "top_k_relationships": 10,
    "include_entity_rank": True,
    "include_relationship_weight": True,
    "include_community_rank": False,
    "return_candidate_context": False, #True,
    "embedding_vectorstore_key": EntityVectorStoreKey.ID,
    "max_tokens": 24_000,
}

model_params = {
    "max_tokens": 3_000,
    "temperature": 0.0,
}

search_engine = LocalSearch(
    model=chat_model,
    context_builder=context_builder,
    tokenizer=tokenizer,
    model_params=model_params,
    context_builder_params=local_context_params,
    response_type="multiple paragraphs",
)
```


```python
start = time.time()

result = await search_engine.search("Draw conclusions about the i101 object in Attempto Controlled English")

end = time.time()
print(f"Длительность: {(end - start):.3f} сек.")
```

    [92m08:23:06 - LiteLLM:INFO[0m: utils.py:1575 - Wrapper: Completed Call, calling success_handler
    2026-01-10 08:23:06,010 - LiteLLM - INFO - Wrapper: Completed Call, calling success_handler
    [92m08:23:06 - LiteLLM:INFO[0m: utils.py:3749 - 
    LiteLLM completion() model= google/gemma-3-27b-it; provider = openai
    2026-01-10 08:23:06,734 - LiteLLM - INFO - 
    LiteLLM completion() model= google/gemma-3-27b-it; provider = openai
    

    Длительность: 14.509 сек.
    


```python
print("\n📦 Создание адаптера GraphRAG для RAGAS...")

graphrag_adapter = GraphRAGAdapter(search_engine=search_engine)
```

    
    📦 Создание адаптера GraphRAG для RAGAS...
    


```python
print("\n📂 Загрузка тестовых кейсов...")

loaded_test_cases = load_test_cases_from_file("./datasets/rag_test_cases_20260109_222314.json")

print(f"✅ Загружено {len(loaded_test_cases)} тестовых кейсов")
```

    
    📂 Загрузка тестовых кейсов...
    📂 Загрузка тестовых кейсов: datasets\rag_test_cases_20260109_222314.json
    ✅ Загружено 198 тестовых кейсов из JSON
    ✅ Загружено 198 тестовых кейсов
    


```python
print("\n📊 Генерация датасета GraphRAG")

graphrag_dataset = prepare_graphrag_evaluation_dataset(
    graphrag_adapter=graphrag_adapter,
    test_cases=loaded_test_cases, #test_cases,
    save_path="./datasets/graphrag_evaluation_dataset.parquet",
    verbose=True
)
```


```python
print("🔍 Анализ совместимости")

compatibility = analyze_graphrag_ragas_compatibility(graphrag_dataset)
```


```python
graphrag_dataset.contexts[0][:150]
```




    'Entity: Unknown Entity|||Entity: Unknown Entity | Description: I90 is a context-dependent shape representation and has a representation relation (i91)'




```python
metrics_eq = [
    # "faithfulness",
    "answer_relevancy",
    # "context_precision",
    "context_recall",
    "context_relevance",
    # "response_groundedness",
    "answer_accuracy"
]
```


```python
print("🚀 Расчёт метрик RAGAS для GraphRAG")

graphrag_evaluation_results_eq = evaluate_graphrag_with_ragas(
    dataset=graphrag_dataset,
    judge_model="openai/gpt-5-mini",
    judge_api_key="sk-or-vv-xxx",
    judge_api_base="https://api.vsegpt.ru/v1",
    embedding_model="emb-openai/text-embedding-3-large",
    max_tokens=32000,
    enable_timing=True,
    analyze_compatibility=True,
    metrics=metrics_eq
)
```


```python
print("⚖️ Сравнение RAG vs. GraphRAG")

try:
    # Загрузка результатов RAG
    # with open("results/ragas_metrics_20260109_183330.json", 'r') as f:
    #     rag_aggregated = json.load(f)
    # rag_results = {"aggregated_metrics": rag_aggregated}
    
    # Сравнение
    comparison_df_eq = compare_rag_vs_graphrag(
        rag_results=evaluation_results, # rag_results,
        graphrag_results=graphrag_evaluation_results_eq,
        save_path="./results/rag_vs_graphrag_comparison_eq.csv"
    )
    
except Exception as e:
    print(f"⚠️  Сравнение недоступно: {e}")
```

    ⚖️ Сравнение RAG vs. GraphRAG
    💾 Сравнение сохранено: results\rag_vs_graphrag_comparison_eq.csv
    



<div style="border: 2px solid #9C27B0; border-radius: 12px; padding: 20px; margin: 15px 0;
            background: linear-gradient(135deg, #F3E5F5 0%, #E1BEE7 100%);">
    <h2 style="color: #6A1B9A; margin-top: 0;">⚖️ Сравнение RAG vs GraphRAG</h2>

    <table style="width: 100%; border-collapse: collapse; background: white; margin: 15px 0;">
        <thead>
            <tr style="background: #9C27B0; color: white;">
                <th style="padding: 12px; text-align: left;">Метрика</th>
                <th style="padding: 12px; text-align: center;">RAG</th>
                <th style="padding: 12px; text-align: center;">GraphRAG</th>
                <th style="padding: 12px; text-align: center;">Разница</th>
                <th style="padding: 12px; text-align: center;">Улучшение</th>
                <th style="padding: 12px; text-align: center;">Победитель</th>
            </tr>
        </thead>
        <tbody>

        <tr style="background: #C8E6C9;">
            <td style="padding: 10px; font-weight: bold;">answer_accuracy</td>
            <td style="padding: 10px; text-align: center;">0.367</td>
            <td style="padding: 10px; text-align: center;">0.486</td>
            <td style="padding: 10px; text-align: center; font-weight: bold;">+0.119</td>
            <td style="padding: 10px; text-align: center; font-weight: bold;">+32.3%</td>
            <td style="padding: 10px; text-align: center;">
                <span style="background: #2E7D32; color: white; padding: 4px 8px; border-radius: 12px; font-size: 0.9em;">
                    🏆 GraphRAG
                </span>
            </td>
        </tr>

        <tr style="background: #C8E6C9;">
            <td style="padding: 10px; font-weight: bold;">answer_relevancy</td>
            <td style="padding: 10px; text-align: center;">0.359</td>
            <td style="padding: 10px; text-align: center;">0.483</td>
            <td style="padding: 10px; text-align: center; font-weight: bold;">+0.124</td>
            <td style="padding: 10px; text-align: center; font-weight: bold;">+34.7%</td>
            <td style="padding: 10px; text-align: center;">
                <span style="background: #2E7D32; color: white; padding: 4px 8px; border-radius: 12px; font-size: 0.9em;">
                    🏆 GraphRAG
                </span>
            </td>
        </tr>

        <tr style="background: #C8E6C9;">
            <td style="padding: 10px; font-weight: bold;">context_recall</td>
            <td style="padding: 10px; text-align: center;">0.116</td>
            <td style="padding: 10px; text-align: center;">0.727</td>
            <td style="padding: 10px; text-align: center; font-weight: bold;">+0.611</td>
            <td style="padding: 10px; text-align: center; font-weight: bold;">+525.8%</td>
            <td style="padding: 10px; text-align: center;">
                <span style="background: #2E7D32; color: white; padding: 4px 8px; border-radius: 12px; font-size: 0.9em;">
                    🏆 GraphRAG
                </span>
            </td>
        </tr>

        <tr style="background: #C8E6C9;">
            <td style="padding: 10px; font-weight: bold;">context_relevance</td>
            <td style="padding: 10px; text-align: center;">0.551</td>
            <td style="padding: 10px; text-align: center;">0.975</td>
            <td style="padding: 10px; text-align: center; font-weight: bold;">+0.424</td>
            <td style="padding: 10px; text-align: center; font-weight: bold;">+77.1%</td>
            <td style="padding: 10px; text-align: center;">
                <span style="background: #2E7D32; color: white; padding: 4px 8px; border-radius: 12px; font-size: 0.9em;">
                    🏆 GraphRAG
                </span>
            </td>
        </tr>

        <tr style="background: #C8E6C9;">
            <td style="padding: 10px; font-weight: bold;">overall</td>
            <td style="padding: 10px; text-align: center;">0.348</td>
            <td style="padding: 10px; text-align: center;">0.668</td>
            <td style="padding: 10px; text-align: center; font-weight: bold;">+0.320</td>
            <td style="padding: 10px; text-align: center; font-weight: bold;">+91.8%</td>
            <td style="padding: 10px; text-align: center;">
                <span style="background: #2E7D32; color: white; padding: 4px 8px; border-radius: 12px; font-size: 0.9em;">
                    🏆 GraphRAG
                </span>
            </td>
        </tr>

        </tbody>
    </table>

    <div style="margin-top: 20px; padding: 15px; background: white; border-radius: 8px;">
        <h3 style="color: #6A1B9A;">📈 Общий результат</h3>

        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top: 15px;">
            <div style="text-align: center; padding: 15px; background: #C8E6C9; border-radius: 8px;">
                <div style="font-size: 2em; font-weight: bold; color: #2E7D32;">5</div>
                <div style="color: #666;">GraphRAG побед</div>
            </div>
            <div style="text-align: center; padding: 15px; background: #FFE0B2; border-radius: 8px;">
                <div style="font-size: 2em; font-weight: bold; color: #E65100;">0</div>
                <div style="color: #666;">RAG побед</div>
            </div>
            <div style="text-align: center; padding: 15px; background: #E0E0E0; border-radius: 8px;">
                <div style="font-size: 2em; font-weight: bold; color: #666;">0</div>
                <div style="color: #666;">Ничьих</div>
            </div>
        </div>
    </div>
</div>



    
    📊 Детальная таблица:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>RAG</th>
      <th>GraphRAG</th>
      <th>Difference</th>
      <th>Improvement %</th>
      <th>Winner</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>answer_accuracy</td>
      <td>0.367</td>
      <td>0.486</td>
      <td>+0.119</td>
      <td>+32.3%</td>
      <td>GraphRAG</td>
    </tr>
    <tr>
      <th>1</th>
      <td>answer_relevancy</td>
      <td>0.359</td>
      <td>0.483</td>
      <td>+0.124</td>
      <td>+34.7%</td>
      <td>GraphRAG</td>
    </tr>
    <tr>
      <th>2</th>
      <td>context_recall</td>
      <td>0.116</td>
      <td>0.727</td>
      <td>+0.611</td>
      <td>+525.8%</td>
      <td>GraphRAG</td>
    </tr>
    <tr>
      <th>3</th>
      <td>context_relevance</td>
      <td>0.551</td>
      <td>0.975</td>
      <td>+0.424</td>
      <td>+77.1%</td>
      <td>GraphRAG</td>
    </tr>
    <tr>
      <th>4</th>
      <td>overall</td>
      <td>0.348</td>
      <td>0.668</td>
      <td>+0.320</td>
      <td>+91.8%</td>
      <td>GraphRAG</td>
    </tr>
  </tbody>
</table>
</div>



```python
print("\n📊 Результаты GraphRAG:")
for metric, value in graphrag_evaluation_results_eq["aggregated_metrics"].items():
    if metric.endswith('_mean'):
        metric_name = metric.replace('_mean', '')
        print(f"   • {metric_name}: {value:.3f}")
```

    
    📊 Результаты GraphRAG:
       • answer_relevancy: 0.483
       • context_recall: 0.727
       • context_relevance: 0.975
       • answer_accuracy: 0.486
       • overall: 0.668
    


```python
print("\n📈 Статистика GraphRAG:")
stats = graphrag_evaluation_results_eq["graphrag_stats"]
print(f"   • Среднее контекстов: {stats['avg_contexts']:.1f}")
print(f"   • Среднее сущностей: {stats['avg_entities']:.1f}")
print(f"   • Среднее связей: {stats['avg_relationships']:.1f}")
print(f"   • Среднее время запроса: {stats['avg_query_time']:.1f}с")
```

    
    📈 Статистика GraphRAG:
       • Среднее контекстов: 80.7
       • Среднее сущностей: 20.0
       • Среднее связей: 45.9
       • Среднее время запроса: 14.4с
    


```python
print("\n💾 Файлы сохранены в директории 'results/'")
```

    
    💾 Файлы сохранены в директории 'results/'
    


```python

```
