import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging
from datetime import datetime
import json

from src.config.config import PipelineConfig
from src.rag.vector_store import ChromaStorage, SearchResult
from src.rag.text_processor import TextProcessor
from src.utils.logger import setup_logger
from src.utils.dataset_loader import save_evaluation_dataset
from src.evaluation.ragas_evaluator import RAGASEvaluator, EvaluationResult

logger = setup_logger("rag_pipeline")

class RAGPipeline:
    """
    RAG пайплайн для поиска и генерации ответов.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Инициализация пайплайна.
        
        Args:
            config: Конфигурация пайплайна
        """
        if config is None:
            config = PipelineConfig()
        
        self.config = config
        self.vector_store = None
        self.text_processor = TextProcessor(config.chunking)
        self.llm_client = None
        self.evaluator = None
        
        # Инициализируем векторное хранилище
        self._init_vector_store()
        
        logger.info(f"RAG пайплайн инициализирован на устройстве: {config.device}")
    
    def _init_vector_store(self):
        """Инициализирует векторное хранилище."""
        try:
            self.vector_store = ChromaStorage(
                db_path=self.config.data.db_path,
                collection_name=self.config.vector_store.collection_name,
                embedding_config=self.config.embeddings,
                reset_db=False
            )
            
            info = self.vector_store.get_collection_info()
            logger.info(f"Векторное хранилище инициализировано: {info}")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации векторного хранилища: {e}")
            raise
    
    def load_and_process_data(
        self,
        file_pattern: str = "*.txt",
        max_files: Optional[int] = None,
        save_intermediate: bool = True
    ) -> pd.DataFrame:
        """
        Загружает и обрабатывает данные.
        
        Args:
            file_pattern: Шаблон файлов
            max_files: Максимальное количество файлов
            save_intermediate: Сохранять ли промежуточные результаты
        
        Returns:
            DataFrame с обработанными данными
        """
        logger.info("Загрузка данных...")
        
        data_dir = Path(self.config.data.data_dir)
        files = list(data_dir.glob(file_pattern))
        
        if max_files:
            files = files[:max_files]
        
        if not files:
            logger.warning(f"Файлы по шаблону {file_pattern} не найдены в {data_dir}")
            return pd.DataFrame()
        
        logger.info(f"Найдено {len(files)} файлов")
        
        # Обрабатываем файлы
        all_chunks = []
        
        for file_path in files:
            chunks = self.text_processor.process_file(file_path)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            logger.error("Не удалось загрузить ни одного файла")
            return pd.DataFrame()
        
        # Создаем DataFrame
        data = []
        for i, chunk in enumerate(all_chunks):
            record = {
                "id": f"chunk_{i}_{datetime.now().timestamp()}",
                "text": chunk["text"],
                **chunk["metadata"]
            }
            data.append(record)
        
        df = pd.DataFrame(data)
        logger.info(f"Создан DataFrame с {len(df)} записями")
        
        # Сохраняем промежуточные результаты
        if save_intermediate:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(self.config.data.results_dir) / f"processed_data_{timestamp}.parquet"
            save_evaluation_dataset(df, str(output_path))
            logger.info(f"Промежуточные данные сохранены: {output_path}")
        
        return df
    
    def build_vector_store(
        self,
        df: pd.DataFrame,
        clear_existing: bool = False
    ) -> List[str]:
        """
        Строит векторное хранилище из DataFrame.
        
        Args:
            df: DataFrame с данными
            clear_existing: Очистить ли существующую коллекцию
        
        Returns:
            Список ID добавленных документов
        """
        if self.vector_store is None:
            self._init_vector_store()
        
        if df.empty:
            logger.error("DataFrame пуст")
            return []
        
        # Очищаем коллекцию, если нужно
        if clear_existing:
            logger.info("Очистка существующей коллекции...")
            # Реализация очистки коллекции
        
        # Подготавливаем данные
        texts = df["text"].tolist()
        ids = df["id"].tolist()
        
        # Создаем метаданные
        metadatas = []
        for _, row in df.iterrows():
            metadata = {k: v for k, v in row.items() if k not in ["id", "text"]}
            metadata["id"] = row["id"]
            metadatas.append(metadata)
        
        logger.info(f"Добавление {len(texts)} документов в векторное хранилище...")
        
        # Добавляем в хранилище
        added_ids = self.vector_store.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            batch_size=self.config.batch_size
        )
        
        # Получаем информацию о коллекции
        info = self.vector_store.get_collection_info()
        logger.info(f"Векторное хранилище построено: {info}")
        
        return added_ids
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Выполняет поиск в векторном хранилище.
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            score_threshold: Порог сходства
        
        Returns:
            Результаты поиска
        """
        if self.vector_store is None:
            logger.error("Векторное хранилище не инициализировано")
            return []
        
        top_k = top_k or self.config.vector_store.top_k
        score_threshold = score_threshold or self.config.vector_store.similarity_threshold
        
        logger.info(f"Поиск запроса: '{query}'")
        
        # Выполняем поиск
        results = self.vector_store.search(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold
        )
        
        # Логируем результаты
        if results:
            logger.info(f"Найдено {len(results)} результатов:")
            for i, result in enumerate(results[:3]):  # Показываем только первые 3
                logger.info(f"  {i+1}. Score: {result.similarity_score:.4f}, Source: {result.metadata.get('source_file', 'unknown')}")
        else:
            logger.info("Результаты не найдены")
        
        return results
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        return_details: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        Выполняет полный RAG запрос: поиск + генерация ответа.
        
        Args:
            query: Вопрос пользователя
            top_k: Количество контекстов для поиска
            score_threshold: Порог сходства
            return_details: Возвращать детали (контексты, источники)
        
        Returns:
            Ответ или словарь с деталями
        """
        if self.llm_client is None:
            raise ValueError("LLM клиент не установлен. Установите его с помощью set_llm_client()")
        
        if self.vector_store is None:
            raise ValueError("Векторное хранилище не инициализировано")
        
        top_k = top_k or self.config.vector_store.top_k
        score_threshold = score_threshold or self.config.vector_store.similarity_threshold
        
        logger.info(f"RAG запрос: '{query}'")
        
        # 1. Поиск контекстов
        search_results = self.search(query, top_k=top_k, score_threshold=score_threshold)
        
        if not search_results:
            answer = "Извините, не удалось найти релевантную информацию для ответа на ваш вопрос."
            if return_details:
                return {
                    "answer": answer,
                    "contexts": [],
                    "sources": [],
                    "scores": []
                }
            return answer
        
        # 2. Извлекаем контексты
        contexts = [r.content for r in search_results]
        sources = [r.metadata.get('source_file', 'unknown') for r in search_results]
        scores = [r.similarity_score for r in search_results]
        
        # 3. Генерация ответа
        logger.info("Генерация ответа...")
        
        # Формируем промпт
        combined_context = "\n\n".join([
            f"[Контекст {i+1}]:\n{ctx}" 
            for i, ctx in enumerate(contexts[:5])  # Берем топ-5
        ])
        
        system_prompt = """Ты - полезный ассистент, который отвечает на вопросы пользователя на основе предоставленного контекста.

КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:
1. Используй ТОЛЬКО информацию из предоставленного контекста
2. Если в контексте нет ответа на вопрос, честно скажи об этом
3. НЕ добавляй информацию, которой нет в контексте
4. Отвечай четко и по делу
5. Если уместно, цитируй факты из контекста
6. Отвечай на том же языке, что и вопрос пользователя"""

        user_prompt = f"""Контекст для ответа:
{combined_context}

Вопрос пользователя:
{query}

Ответ на основе контекста:"""

        answer = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=self.config.llm.generation_temperature,
            max_tokens=self.config.llm.generation_max_tokens
        )
        
        if return_details:
            return {
                "answer": answer,
                "contexts": contexts,
                "sources": sources,
                "scores": scores,
                "search_results": search_results
            }
        
        return answer
    
    def set_llm_client(self, llm_client):
        """Устанавливает LLM клиент."""
        self.llm_client = llm_client
        logger.info("LLM клиент установлен")
    
    def evaluate(
        self,
        test_cases: List[Dict[str, Any]],
        output_dir: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> EvaluationResult:
        """
        Оценивает качество RAG системы с использованием RAGAS.
        
        Args:
            test_cases: Тестовые кейсы
            output_dir: Директория для сохранения результатов
            top_k: Количество контекстов для поиска
        
        Returns:
            Результаты оценки
        """
        if self.evaluator is None:
            self.evaluator = RAGASEvaluator(
                llm_config=self.config.llm,
                embedding_model=self.config.ragas.embedding_model
            )
        
        logger.info(f"Оценка RAG системы на {len(test_cases)} тестовых кейсах...")
        
        # Генерируем ответы для всех тестовых кейсов
        queries = []
        answers = []
        contexts_list = []
        ground_truths = []
        
        for test_case in test_cases:
            query = test_case["query"]
            ground_truth = test_case.get("ground_truth", "")
            
            # Получаем ответ и контексты
            result = self.query(
                query=query,
                top_k=top_k,
                return_details=True
            )
            
            queries.append(query)
            answers.append(result["answer"])
            contexts_list.append(result["contexts"])
            ground_truths.append(ground_truth)
        
        # Выполняем оценку
        evaluation_result = self.evaluator.evaluate(
            queries=queries,
            answers=answers,
            contexts=contexts_list,
            ground_truths=ground_truths,
            metrics=self.config.ragas.metrics
        )
        
        # Сохраняем результаты
        if output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(output_dir) / f"rag_evaluation_{timestamp}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_result.to_dict(), f, ensure_ascii=False, indent=2)
            
            logger.info(f"Результаты оценки сохранены: {output_path}")
        
        return evaluation_result
    