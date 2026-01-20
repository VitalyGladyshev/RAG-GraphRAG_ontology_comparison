import chromadb
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging
from dataclasses import dataclass

from src.config.config import EmbeddingConfig
from src.rag.text_processor import MultilingualE5Embedder
from src.utils.logger import setup_logger

logger = setup_logger("vector_store")

@dataclass
class SearchResult:
    """Результат поиска в векторной базе."""
    id: str
    content: str
    similarity_score: float
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует результат в словарь."""
        return {
            "id": self.id,
            "content": self.content,
            "similarity_score": self.similarity_score,
            "metadata": self.metadata
        }

class ChromaStorage:
    """
    Класс для работы с векторным хранилищем ChromaDB.
    Поддерживает multilingual-e5-large модель.
    """
    
    def __init__(
        self,
        db_path: str,
        collection_name: str = "document_chunks",
        embedding_config: Optional[EmbeddingConfig] = None,
        create_if_missing: bool = True,
        reset_db: bool = False
    ):
        """
        Инициализация хранилища.
        
        Args:
            db_path: Путь к базе данных
            collection_name: Название коллекции
            embedding_config: Конфигурация эмбеддингов
            create_if_missing: Создавать ли коллекцию, если её нет
            reset_db: Пересоздать ли базу данных
        """
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.create_if_missing = create_if_missing
        
        # Инициализируем модель эмбеддингов
        if embedding_config is None:
            embedding_config = EmbeddingConfig()
        
        self.embedding_config = embedding_config
        self.embedding_model = MultilingualE5Embedder(
            model_name=embedding_config.model_name,
            device="cuda" if embedding_config.use_fp16 else "cpu",
            use_fp16=embedding_config.use_fp16
        )
        
        # Пересоздаём базу, если требуется
        if reset_db and self.db_path.exists():
            logger.warning(f"Удаление базы данных: {self.db_path}")
            import shutil
            shutil.rmtree(self.db_path)
        
        # Создаем директорию для базы данных
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.client = None
        self.collection = None
        self._init_chroma()
    
    def _init_chroma(self):
        """Инициализирует ChromaDB."""
        try:
            # Инициализация клиента
            self.client = chromadb.PersistentClient(
                path=str(self.db_path)
            )
            
            logger.info(f"ChromaDB клиент инициализирован: {self.db_path}")
            
            # Создаем или получаем коллекцию
            self._init_collection()
            
        except Exception as e:
            logger.error(f"Ошибка инициализации ChromaDB: {e}")
            raise
    
    def _init_collection(self):
        """Инициализирует коллекцию."""
        try:
            # Пытаемся получить существующую коллекцию
            try:
                self.collection = self.client.get_collection(self.collection_name)
                logger.info(f"Коллекция '{self.collection_name}' загружена")
            except Exception:
                if self.create_if_missing:
                    # Создаем новую коллекцию
                    self.collection = self.client.create_collection(
                        name=self.collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    logger.info(f"Создана коллекция '{self.collection_name}'")
                else:
                    raise ValueError(f"Коллекция '{self.collection_name}' не существует")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации коллекции: {e}")
            raise
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 32
    ) -> List[str]:
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
            logger.warning("Список текстов пуст")
            return []
        
        # Генерируем ID, если не указаны
        if ids is None:
            import hashlib
            ids = [f"doc_{hashlib.md5(text.encode()).hexdigest()[:12]}" 
                  for text in texts]
        
        # Подготавливаем метаданные
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Проверяем длину списков
        if len(texts) != len(ids) or len(texts) != len(metadatas):
            raise ValueError("Длины texts, ids и metadatas должны совпадать")
        
        logger.info(f"Добавление {len(texts)} документов...")
        
        # Добавляем батчами
        added_ids = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            
            # Создаем эмбеддинги
            batch_embeddings = self.embedding_model.embed_texts(
                batch_texts, 
                text_type="passage",
                show_progress=False,
                batch_size=batch_size
            )
            
            # Добавляем в коллекцию
            self.collection.add(
                documents=batch_texts,
                embeddings=batch_embeddings.tolist(),
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            added_ids.extend(batch_ids)
        
        logger.info(f"Добавлено {len(added_ids)} документов")
        return added_ids
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.7,
        filter_meta: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
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
        
        logger.info(f"Поиск: '{query[:50]}...'")
        
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
            
            logger.info(f"Найдено {len(search_results)} результатов")
            return search_results
            
        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Возвращает информацию о коллекции."""
        try:
            count = self.collection.count()
            collection_metadata = self.collection.metadata or {}
            
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_dimension": self.embedding_config.dimension,
                "db_path": str(self.db_path),
                "metadata": collection_metadata,
                "model": self.embedding_config.model_name
            }
        except Exception as e:
            logger.error(f"Ошибка получения информации о коллекции: {e}")
            return {}
        