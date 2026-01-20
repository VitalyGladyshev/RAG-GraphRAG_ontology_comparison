import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from datetime import datetime
import json
import time

from src.config.config import PipelineConfig, GraphRAGConfig
from src.utils.logger import setup_logger
from src.utils.dataset_loader import save_evaluation_dataset
from src.evaluation.ragas_evaluator import RAGASEvaluator, EvaluationResult

logger = setup_logger("graphrag_pipeline")

class GraphRAGAdapter:
    """
    Адаптер для интеграции Microsoft GraphRAG с RAGAS.
    """
    
    def __init__(self, search_engine):
        """
        Инициализация адаптера.
        
        Args:
            search_engine: Экземпляр LocalSearch из GraphRAG
        """
        self.search_engine = search_engine
        logger.info("GraphRAG адаптер инициализирован")
    
    async def _query_async(self, question: str) -> Dict[str, Any]:
        """
        Асинхронный запрос к GraphRAG.
        
        Args:
            question: Вопрос пользователя
        
        Returns:
            Результат в формате для RAGAS
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
        
        if hasattr(result, 'context_data'):
            contexts, context_data = self._extract_contexts(result.context_data)
        
        return {
            "answer": answer,
            "contexts": contexts,
            "context_data": context_data,
            "elapsed_time": elapsed_time,
            "raw_result": result
        }
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Синхронный запрос к GraphRAG.
        
        Args:
            question: Вопрос пользователя
        
        Returns:
            Результат в формате для RAGAS
        """
        try:
            # Применяем nest_asyncio если нужно (для Jupyter)
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            pass
        
        # Выполняем асинхронный запрос
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            result = loop.run_until_complete(self._query_async(question))
        except RuntimeError:
            import asyncio
            result = asyncio.run(self._query_async(question))
        
        return result
    
    def _extract_contexts(self, context_data: Dict) -> Tuple[List[str], Dict]:
        """
        Извлекает и форматирует контекст из GraphRAG context_data.
        
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
        
        logger.info(
            f"Извлечено контекстов: {len(contexts)} "
            f"(entities: {metadata['num_entities']}, "
            f"relationships: {metadata['num_relationships']}, "
            f"reports: {metadata['num_reports']}, "
            f"sources: {metadata['num_sources']})"
        )
        
        return contexts, metadata
    
    def _format_entity(self, entity: pd.Series) -> str:
        """
        Форматирует сущность в текстовую строку.
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
        """
        parts = []
        
        # Заголовок
        if 'title' in report and pd.notna(report['title']):
            parts.append(f"Community Report: {report['title']}")
        
        # Содержание
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
        """
        if 'text' in source and pd.notna(source['text']):
            text = str(source['text']).strip()
            
            # Добавляем ID если есть
            if 'id' in source and pd.notna(source['id']):
                return f"Source [{source['id']}]: {text}"
            else:
                return f"Source: {text}"
        
        return ""

class GraphRAGPipeline:
    """
    GraphRAG пайплайн для работы с графовой структурой данных.
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
        self.graphrag_adapter = None
        self.search_engine = None
        self.evaluator = None
        
        logger.info("GraphRAG пайплайн инициализирован")
    
    def init_from_directory(self, input_dir: str, llm_config: Optional[Dict] = None):
        """
        Инициализирует GraphRAG из директории с обработанными данными.
        
        Args:
            input_dir: Директория с данными GraphRAG
            llm_config: Конфигурация LLM (опционально)
        """
        try:
            logger.info(f"Инициализация GraphRAG из директории: {input_dir}")
            
            # Импортируем зависимости GraphRAG
            from graphrag.config.models.vector_store_schema_config import VectorStoreSchemaConfig
            from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
            from graphrag.query.indexer_adapters import (
                read_indexer_covariates,
                read_indexer_entities,
                read_indexer_relationships,
                read_indexer_reports,
                read_indexer_text_units,
            )
            from graphrag.query.structured_search.local_search.mixed_context import (
                LocalSearchMixedContext,
            )
            from graphrag.query.structured_search.local_search.search import LocalSearch
            from graphrag.vector_stores.lancedb import LanceDBVectorStore
            from graphrag.config.enums import ModelType
            from graphrag.config.models.language_model_config import LanguageModelConfig
            from graphrag.language_model.manager import ModelManager
            from graphrag.tokenizer.get_tokenizer import get_tokenizer
            
            # Загружаем данные
            entities = pd.read_parquet(f"{input_dir}/entities.parquet")
            communities = pd.read_parquet(f"{input_dir}/communities.parquet")
            community_reports = pd.read_parquet(f"{input_dir}/community_reports.parquet")
            text_units = pd.read_parquet(f"{input_dir}/text_units.parquet")
            relationships = pd.read_parquet(f"{input_dir}/relationships.parquet")
            
            # Инициализируем LLM
            if llm_config is None:
                llm_config = {
                    "api_key": self.config.llm.api_key,
                    "api_base": self.config.llm.api_base,
                    "model": self.config.llm.generation_model
                }
            
            # Инициализируем компоненты
            chat_config = LanguageModelConfig(
                api_key=llm_config["api_key"],
                api_base=llm_config["api_base"],
                type=ModelType.Chat,
                model_provider="openai",
                model=llm_config["model"],
                max_retries=20,
            )
            
            chat_model = ModelManager().get_or_create_chat_model(
                name="local_search",
                model_type=ModelType.Chat,
                config=chat_config,
            )
            
            embedding_config = LanguageModelConfig(
                api_key=llm_config["api_key"],
                api_base=llm_config["api_base"],
                type=ModelType.Embedding,
                model_provider="openai",
                model=self.config.ragas.embedding_model,
                max_retries=20,
            )
            
            text_embedder = ModelManager().get_or_create_embedding_model(
                name="local_search_embedding",
                model_type=ModelType.Embedding,
                config=embedding_config,
            )
            
            tokenizer = get_tokenizer(chat_config)
            
            lancedb_uri = f"{input_dir}/lancedb"
            description_embedding_store = LanceDBVectorStore(
                vector_store_schema_config=VectorStoreSchemaConfig(
                    index_name="default-entity-description"
                )
            )
            description_embedding_store.connect(db_uri=lancedb_uri)
            
            # Читаем индексированные данные
            entities_d = read_indexer_entities(entities, communities, None)
            relationships_d = read_indexer_relationships(relationships)
            reports_d = read_indexer_reports(community_reports, communities, None)
            text_units_d = read_indexer_text_units(text_units)
            
            # Создаем контекстный билдер
            context_builder = LocalSearchMixedContext(
                community_reports=reports_d,
                text_units=text_units_d,
                entities=entities_d,
                relationships=relationships_d,
                covariates=None,
                entity_text_embeddings=description_embedding_store,
                embedding_vectorstore_key=EntityVectorStoreKey.ID,
                text_embedder=text_embedder,
                tokenizer=tokenizer,
            )
            
            # Параметры контекста
            context_params = {
                "text_unit_prop": 0.5,
                "community_prop": 0.1,
                "conversation_history_max_turns": 5,
                "conversation_history_user_turns_only": True,
                "top_k_mapped_entities": 10,
                "top_k_relationships": 10,
                "include_entity_rank": True,
                "include_relationship_weight": True,
                "include_community_rank": False,
                "return_candidate_context": False,
                "embedding_vectorstore_key": EntityVectorStoreKey.ID,
                "max_tokens": 24000,
            }
            
            # Параметры модели
            model_params = {
                "max_tokens": 3000,
                "temperature": 0.0,
            }
            
            # Создаем поисковый движок
            self.search_engine = LocalSearch(
                model=chat_model,
                context_builder=context_builder,
                tokenizer=tokenizer,
                model_params=model_params,
                context_builder_params=context_params,
                response_type="multiple paragraphs",
            )
            
            # Создаем адаптер
            self.graphrag_adapter = GraphRAGAdapter(self.search_engine)
            
            logger.info("GraphRAG успешно инициализирован")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации GraphRAG: {e}")
            raise
    
    def query(self, question: str, return_details: bool = False) -> Union[str, Dict[str, Any]]:
        """
        Выполняет запрос к GraphRAG.
        
        Args:
            question: Вопрос пользователя
            return_details: Возвращать детали (контексты, метаданные)
        
        Returns:
            Ответ или словарь с деталями
        """
        if self.graphrag_adapter is None:
            raise ValueError("GraphRAG не инициализирован. Используйте init_from_directory()")
        
        logger.info(f"GraphRAG запрос: '{question}'")
        
        # Получаем ответ от GraphRAG
        result = self.graphrag_adapter.query(question)
        
        if return_details:
            return result
        
        return result["answer"]
    
    def evaluate(
        self,
        test_cases: List[Dict[str, Any]],
        output_dir: Optional[str] = None
    ) -> EvaluationResult:
        """
        Оценивает качество GraphRAG системы с использованием RAGAS.
        
        Args:
            test_cases: Тестовые кейсы
            output_dir: Директория для сохранения результатов
        
        Returns:
            Результаты оценки
        """
        if self.evaluator is None:
            self.evaluator = RAGASEvaluator(
                llm_config=self.config.llm,
                embedding_model=self.config.ragas.embedding_model
            )
        
        logger.info(f"Оценка GraphRAG системы на {len(test_cases)} тестовых кейсах...")
        
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
                question=query,
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
            output_path = Path(output_dir) / f"graphrag_evaluation_{timestamp}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_result.to_dict(), f, ensure_ascii=False, indent=2)
            
            logger.info(f"Результаты оценки сохранены: {output_path}")
        
        return evaluation_result
    