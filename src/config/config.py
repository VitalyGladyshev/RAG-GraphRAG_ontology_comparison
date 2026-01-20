from dataclasses import dataclass
from pathlib import Path
import torch
from typing import Optional, List, Dict, Any

@dataclass
class DataConfig:
    """Конфигурация данных."""
    data_dir: str = "data/chunks"
    db_path: str = "vector_db"
    model_cache: str = "models/cache"
    logs_dir: str = "logs"
    results_dir: str = "results"
    
    def __post_init__(self):
        """Создание необходимых директорий."""
        for directory in [self.data_dir, self.db_path, 
                         self.model_cache, self.logs_dir,
                         self.results_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)

@dataclass
class EmbeddingConfig:
    """Конфигурация эмбеддингов."""
    model_name: str = "intfloat/multilingual-e5-large"
    dimension: int = 1024
    max_sequence_length: int = 512
    use_fp16: bool = torch.cuda.is_available()

@dataclass
class ChunkingConfig:
    """Конфигурация чанкинга текста."""
    chunk_size: int = 500
    chunk_overlap: int = 50
    separators: List[str] = None
    
    def __post_init__(self):
        if self.separators is None:
            self.separators = ["\n\n", "\n", ".  ", "  ", " "]

@dataclass
class VectorStoreConfig:
    """Конфигурация векторного хранилища."""
    collection_name: str = "document_chunks"
    distance_metric: str = "cosine"
    similarity_threshold: float = 0.7
    top_k: int = 5

@dataclass
class LLMConfig:
    """Конфигурация LLM."""
    api_base: str = "https://api.vsegpt.ru/v1"
    api_key: str = ""
    generation_model: str = "google/gemma-3-27b-it"
    evaluation_model: str = "openai/gpt-5-mini"
    generation_temperature: float = 0.7
    generation_max_tokens: int = 1024
    evaluation_temperature: float = 0.0
    evaluation_max_tokens: int = 512

@dataclass
class GraphRAGConfig:
    """Конфигурация GraphRAG."""
    input_dir: str = "output"
    lancedb_uri: str = "output/lancedb"
    community_report_table: str = "community_reports"
    entity_table: str = "entities"
    community_table: str = "communities"
    relationship_table: str = "relationships"
    covariate_table: str = "covariates"
    text_unit_table: str = "text_units"
    community_level: int = 2

@dataclass
class RAGASConfig:
    """Конфигурация RAGAS оценки."""
    metrics: List[str] = None
    embedding_model: str = "emb-openai/text-embedding-3-large"
    enable_timing: bool = True
    max_metric_time: float = 15.0
    exclude_slow_metrics: bool = False
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
                "context_relevance",
                "response_groundedness",
                "answer_accuracy"
            ]

@dataclass
class PipelineConfig:
    """Общая конфигурация пайплайна."""
    data: DataConfig = None
    embeddings: EmbeddingConfig = None
    chunking: ChunkingConfig = None
    vector_store: VectorStoreConfig = None
    llm: LLMConfig = None
    graphrag: GraphRAGConfig = None
    ragas: RAGASConfig = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    log_level: str = "INFO"
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.embeddings is None:
            self.embeddings = EmbeddingConfig()
        if self.chunking is None:
            self.chunking = ChunkingConfig()
        if self.vector_store is None:
            self.vector_store = VectorStoreConfig()
        if self.llm is None:
            self.llm = LLMConfig()
        if self.graphrag is None:
            self.graphrag = GraphRAGConfig()
        if self.ragas is None:
            self.ragas = RAGASConfig()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PipelineConfig":
        """Создание конфигурации из словаря."""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    