import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.logger import setup_logger
from src.config.config import EmbeddingConfig, ChunkingConfig

logger = setup_logger("text_processor")

class MultilingualE5Embedder:
    """Класс для работы с моделью multilingual-e5-large."""
    
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        device: str = "cuda",
        use_fp16: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Инициализация модели эмбеддингов.
        
        Args:
            model_name: Название модели на Hugging Face
            device: Устройство для вычислений (cuda/cpu)
            use_fp16: Использовать ли FP16 для GPU
            cache_dir: Директория для кэширования модели
        """
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16
        self.cache_dir = cache_dir or "models/cache"
        
        # Создаем директорию для кэша
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Загружает модель и токенизатор."""
        try:
            logger.info(f"Загрузка модели {self.model_name} на устройство {self.device}...")
            
            # Импортируем зависимости
            from sentence_transformers import SentenceTransformer
            from transformers import AutoTokenizer
            
            # Загружаем модель
            model_kwargs = {}
            if self.device == "cuda" and self.use_fp16:
                model_kwargs["torch_dtype"] = "float16"
            
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=str(self.cache_dir),
                **model_kwargs
            )
            
            # Загружаем токенизатор
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir)
            )
            
            logger.info(f"Модель загружена на {self.device}")
            logger.info(f"Размерность эмбеддингов: {self.get_embedding_dimension()}")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Возвращает размерность эмбеддингов модели."""
        if self.model is None:
            return 1024  # Значение по умолчанию
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
    
    def embed_texts(
        self,
        texts: List[str],
        text_type: str = "passage",
        batch_size: int = 32,
        show_progress: bool = True
    ) -> Any:
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
            return []
        
        # Форматируем тексты
        formatted_texts = [self.format_text(text, text_type) for text in texts]
        
        logger.info(f"Создание эмбеддингов для {len(texts)} текстов...")
        
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
            
            logger.info(f"Создано {len(embeddings)} эмбеддингов")
            return embeddings
            
        except Exception as e:
            logger.error(f"Ошибка создания эмбеддингов: {e}")
            raise
    
    def embed_query(self, query: str) -> Any:
        """
        Создает эмбеддинг для одного запроса.
        
        Args:
            query: Текст запроса
        
        Returns:
            Эмбеддинг запроса
        """
        return self.embed_texts([query], text_type="query", show_progress=False)[0]

class TextProcessor:
    """Обработчик текста для создания чанков."""
    
    def __init__(self, chunking_config: Optional[ChunkingConfig] = None):
        """
        Инициализация процессора текста.
        
        Args:
            chunking_config: Конфигурация чанкинга
        """
        if chunking_config is None:
            chunking_config = ChunkingConfig()
        
        self.chunk_size = chunking_config.chunk_size
        self.chunk_overlap = chunking_config.chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=chunking_config.separators
        )
    
    def split_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Разделяет текст на чанки.
        
        Args:
            text: Исходный текст
            metadata: Базовые метаданные для всех чанков
        
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
        
        logger.info(f"Разделено на {len(chunks)} чанков")
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
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read().strip()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                logger.error(f"Не удалось прочитать файл {file_path}")
                return []
            
            if not content:
                logger.warning(f"Файл {file_path} пустой")
                return []
            
            # Создаем базовые метаданные
            base_metadata = {
                "source_file": file_path.name,
                "file_path": str(file_path),
                "file_size": len(content),
            }
            
            # Разделяем на чанки
            chunks = self.split_text(content, base_metadata)
            return chunks
            
        except Exception as e:
            logger.error(f"Ошибка обработки файла {file_path}: {e}")
            return []
        