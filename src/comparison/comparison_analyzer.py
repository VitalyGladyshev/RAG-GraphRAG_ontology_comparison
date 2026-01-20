import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging
from datetime import datetime

from src.utils.logger import setup_logger
from src.evaluation.ragas_evaluator import EvaluationResult

logger = setup_logger("comparison_analyzer")

class ComparisonAnalyzer:
    """
    Анализатор для сравнения результатов RAG и GraphRAG.
    """
    
    def __init__(self):
        """Инициализация анализатора."""
        pass
    
    def compare_results(
        self,
        rag_result: EvaluationResult,
        graphrag_result: EvaluationResult,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Сравнивает результаты RAG и GraphRAG.
        
        Args:
            rag_result: Результаты оценки RAG
            graphrag_result: Результаты оценки GraphRAG
            metrics: Список метрик для сравнения (опционально)
        
        Returns:
            DataFrame со сравнением
        """
        if metrics is None:
            metrics = [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
                "context_relevance",
                "response_groundedness",
                "answer_accuracy"
            ]
        
        comparison_data = []
        
        for metric in metrics:
            rag_value = getattr(rag_result, metric, None)
            graphrag_value = getattr(graphrag_result, metric, None)
            
            if rag_value is not None and graphrag_value is not None:
                difference = graphrag_value - rag_value
                improvement = (difference / rag_value * 100) if rag_value > 0 else 0
                
                comparison_data.append({
                    "Metric": metric,
                    "RAG": rag_value,
                    "GraphRAG": graphrag_value,
                    "Difference": difference,
                    "Improvement_%": improvement,
                    "Winner": "GraphRAG" if graphrag_value > rag_value else 
                             ("RAG" if rag_value > graphrag_value else "Tie")
                })
        
        return pd.DataFrame(comparison_data)
    
    def save_comparison(
        self,
        comparison_df: pd.DataFrame,
        output_path: str,
        rag_stats: Optional[Dict] = None,
        graphrag_stats: Optional[Dict] = None
    ) -> Path:
        """
        Сохраняет результаты сравнения.
        
        Args:
            comparison_df: DataFrame со сравнением
            output_path: Путь для сохранения
            rag_stats: Статистика RAG (опционально)
            graphrag_stats: Статистика GraphRAG (опционально)
        
        Returns:
            Путь к сохраненному файлу
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Подготавливаем данные для сохранения
        result_data = {
            "comparison": comparison_df.to_dict(orient="records"),
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "rag_stats": rag_stats or {},
                "graphrag_stats": graphrag_stats or {}
            }
        }
        
        # Сохраняем в JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        # Также сохраняем CSV для удобства
        csv_path = output_path.with_suffix('.csv')
        comparison_df.to_csv(csv_path, index=False)
        
        logger.info(f"Результаты сравнения сохранены: {output_path}")
        logger.info(f"CSV версия: {csv_path}")
        
        return output_path
    