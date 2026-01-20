import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import time
from collections import defaultdict

from src.utils.logger import setup_logger
from src.config.config import LLMConfig

logger = setup_logger("ragas_evaluator")

@dataclass
class EvaluationResult:
    """Результат оценки."""
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
    
    metadata: Dict[str, Any] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
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

class RAGASEvaluator:
    """
    Оценщик RAG систем с использованием RAGAS.
    """
    
    # Доступные метрики
    AVAILABLE_METRICS = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "context_relevance",
        "response_groundedness",
        "answer_accuracy",
    ]
    
    def __init__(
        self,
        llm_config: LLMConfig,
        embedding_model: str = "emb-openai/text-embedding-3-large",
        metrics: Optional[List[str]] = None,
        enable_timing: bool = True
    ):
        """
        Инициализация оценщика.
        
        Args:
            llm_config: Конфигурация LLM
            embedding_model: Модель для embeddings
            metrics: Список метрик для оценки
            enable_timing: Включить анализ времени выполнения
        """
        self.llm_config = llm_config
        self.embedding_model = embedding_model
        self.enable_timing = enable_timing
        
        # Базовые метрики
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
        
        # Определяем метрики
        if metrics is None:
            self.metrics_to_use = self.basic_metrics
        else:
            self.metrics_to_use = metrics
        
        self._validate_metrics()
        
        # Инициализируем RAGAS компоненты
        self._init_ragas_components()
        
        # Отслеживание времени
        self.metric_timings = defaultdict(list)
        
        logger.info(f"RAGAS оценщик инициализирован с метриками: {', '.join(self.metrics_to_use)}")
    
    def _validate_metrics(self):
        """Проверяет валидность выбранных метрик."""
        invalid_metrics = set(self.metrics_to_use) - set(self.AVAILABLE_METRICS)
        if invalid_metrics:
            raise ValueError(f"Неизвестные метрики: {invalid_metrics}")
    
    def _init_ragas_components(self):
        """Инициализирует компоненты RAGAS."""
        try:
            # Импортируем RAGAS
            from ragas.llms import llm_factory
            from ragas.embeddings.base import embedding_factory
            from ragas.metrics.collections import (
                Faithfulness,
                AnswerRelevancy,
                ContextPrecision,
                ContextRecall,
                ContextRelevance,
                ResponseGroundedness,
                AnswerAccuracy,
            )
            
            # Создаем LLM клиент
            import openai
            from openai import AsyncOpenAI
            
            self.llm_client = openai.OpenAI(
                api_key=self.llm_config.api_key,
                base_url=self.llm_config.api_base,
                timeout=120,
                max_retries=3
            )
            
            self.async_llm_client = AsyncOpenAI(
                api_key=self.llm_config.api_key,
                base_url=self.llm_config.api_base,
                timeout=120,
                max_retries=3
            )
            
            # Создаем RAGAS LLM
            self.ragas_llm = llm_factory(
                model=self.llm_config.evaluation_model,
                client=self.async_llm_client,
                temperature=self.llm_config.evaluation_temperature,
                max_tokens=self.llm_config.evaluation_max_tokens,
            )
            
            # Создаем embeddings только если нужны
            self.ragas_embeddings = None
            if any(m in self.metrics_to_use for m in self.embedding_metrics):
                self.ragas_embeddings = embedding_factory(
                    "openai",
                    model=self.embedding_model,
                    client=self.async_llm_client
                )
                logger.info(f"Embeddings инициализированы: {self.embedding_model}")
            
            # Инициализируем метрики
            self.metric_scorers = {}
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
                            logger.warning(f"Пропущена метрика {metric_name} (нет embeddings)")
                            continue
                        scorer = metric_class(llm=self.ragas_llm, embeddings=self.ragas_embeddings)
                    else:
                        scorer = metric_class(llm=self.ragas_llm)
                    
                    self.metric_scorers[metric_name] = scorer
                    logger.info(f"Метрика {metric_name} инициализирована")
                    
                except Exception as e:
                    logger.warning(f"Ошибка инициализации метрики {metric_name}: {e}")
            
        except ImportError as e:
            logger.error(f"Ошибка импорта RAGAS: {e}")
            raise
        except Exception as e:
            logger.error(f"Ошибка инициализации RAGAS компонентов: {e}")
            raise
    
    async def _evaluate_single_async(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
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
                logger.warning(f"Ошибка при оценке {metric_name}: {e}")
                scores[metric_name] = None

            finally:
                # Сохраняем время выполнения
                if self.enable_timing:
                    elapsed_time = time.time() - start_time
                    self.metric_timings[metric_name].append(elapsed_time)
        
        return scores
    
    def evaluate(
        self,
        queries: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None
    ) -> EvaluationResult:
        """
        Выполняет оценку RAG системы.
        
        Args:
            queries: Список вопросов
            answers: Список ответов
            contexts: Список списков контекстов
            ground_truths: Список правильных ответов (опционально)
            metrics: Список метрик для оценки (опционально)
        
        Returns:
            Агрегированный результат оценки
        """
        if len(queries) != len(answers) or len(queries) != len(contexts):
            raise ValueError("Длины queries, answers и contexts должны совпадать")
        
        if ground_truths is None:
            ground_truths = [None] * len(queries)
        elif len(ground_truths) != len(queries):
            raise ValueError("Длина ground_truths должна совпадать с queries")
        
        logger.info(f"Запуск оценки RAGAS на {len(queries)} примерах...")
        
        # Оцениваем каждый пример
        all_scores = {metric: [] for metric in self.metric_scorers.keys()}
        results = []
        
        for i, (query, answer, context, ground_truth) in enumerate(zip(queries, answers, contexts, ground_truths)):
            logger.info(f"Оценка примера {i+1}/{len(queries)}")
            
            # Создаем результат
            result = EvaluationResult(
                query=query,
                answer=answer,
                context=context,
                ground_truth=ground_truth
            )
            
            # Оцениваем
            try:
                # Выполняем асинхронную оценку
                import asyncio
                scores = asyncio.run(
                    self._evaluate_single_async(query, answer, context, ground_truth)
                )
                
                # Сохраняем метрики
                for metric_name, score in scores.items():
                    setattr(result, metric_name, score)
                    if score is not None:
                        all_scores[metric_name].append(score)
                
            except Exception as e:
                logger.error(f"Ошибка при оценке примера: {e}")
            
            results.append(result)
        
        # Вычисляем агрегированные метрики
        aggregated_result = EvaluationResult(
            query="AGGREGATED",
            answer="AGGREGATED",
            context=[],
            ground_truth=None
        )
        
        for metric_name in self.metric_scorers.keys():
            if metric_name in all_scores and all_scores[metric_name]:
                scores = [s for s in all_scores[metric_name] if s is not None]
                if scores:
                    setattr(aggregated_result, metric_name, float(np.mean(scores)))
        
        # Добавляем метаданные с детальной информацией
        aggregated_result.metadata = {
            "num_examples": len(results),
            "individual_results": [r.to_dict() for r in results],
            "metric_statistics": {}
        }
        
        for metric_name, scores in all_scores.items():
            valid_scores = [s for s in scores if s is not None]
            if valid_scores:
                aggregated_result.metadata["metric_statistics"][metric_name] = {
                    "mean": float(np.mean(valid_scores)),
                    "std": float(np.std(valid_scores)),
                    "min": float(np.min(valid_scores)),
                    "max": float(np.max(valid_scores)),
                    "count": len(valid_scores)
                }
        
        logger.info("Оценка RAGAS завершена успешно!")
        return aggregated_result
    