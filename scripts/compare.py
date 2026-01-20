import argparse
import logging
from pathlib import Path
import sys
import json
from datetime import datetime

from src.comparison.comparison_analyzer import ComparisonAnalyzer
from src.utils.logger import setup_logger
from src.evaluation.ragas_evaluator import EvaluationResult

def load_evaluation_result(file_path: str) -> EvaluationResult:
    """Загружает результаты оценки из файла."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Создаем EvaluationResult из данных
    result = EvaluationResult(
        query="AGGREGATED",
        answer="AGGREGATED",
        context=[],
        ground_truth=None
    )
    
    # Устанавливаем метрики
    metrics = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "context_relevance",
        "response_groundedness",
        "answer_accuracy"
    ]
    
    for metric in metrics:
        if metric in data:
            setattr(result, metric, float(data[metric]))
    
    return result

def main():
    """Основная функция для сравнения результатов."""
    parser = argparse.ArgumentParser(description="Сравнение результатов RAG и GraphRAG систем")
    
    # Аргументы для файлов результатов
    parser.add_argument('--rag-result', type=str, required=True,
                        help='Путь к файлу с результатами RAG оценки')
    parser.add_argument('--graphrag-result', type=str, required=True,
                        help='Путь к файлу с результатами GraphRAG оценки')
    
    # Аргументы для сохранения
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Директория для сохранения результатов сравнения')
    
    # Аргументы для логирования
    parser.add_argument('--log-level', type=str, default='INFO',
                        help='Уровень логирования')
    
    args = parser.parse_args()
    
    # Настройка логирования
    logger = setup_logger("compare", None, args.log_level)
    
    try:
        # Загрузка результатов оценки
        logger.info(f"Загрузка результатов RAG из: {args.rag_result}")
        rag_result = load_evaluation_result(args.rag_result)
        
        logger.info(f"Загрузка результатов GraphRAG из: {args.graphrag_result}")
        graphrag_result = load_evaluation_result(args.graphrag_result)
        
        # Сравнение результатов
        logger.info("Выполнение сравнения результатов...")
        analyzer = ComparisonAnalyzer()
        comparison_df = analyzer.compare_results(rag_result, graphrag_result)
        
        # Вывод результатов сравнения
        print("\n" + "="*80)
        print("СРАВНЕНИЕ RAG И GraphRAG")
        print("="*80)
        
        # Сводная статистика
        graphrag_wins = len(comparison_df[comparison_df['Winner'] == 'GraphRAG'])
        rag_wins = len(comparison_df[comparison_df['Winner'] == 'RAG'])
        ties = len(comparison_df[comparison_df['Winner'] == 'Tie'])
        
        print(f"📊 Сводная статистика:")
        print(f"   GraphRAG побед: {graphrag_wins}")
        print(f"   RAG побед:      {rag_wins}")
        print(f"   Ничьих:         {ties}")
        print()
        
        # Детальное сравнение по метрикам
        print("📈 Детальное сравнение по метрикам:")
        for _, row in comparison_df.iterrows():
            metric_name = row['Metric'].replace('_', ' ').title()
            print(f"\n{metric_name}:")
            print(f"   RAG:       {row['RAG']:.4f}")
            print(f"   GraphRAG:  {row['GraphRAG']:.4f}")
            print(f"   Разница:   {row['Difference']:+.4f} ({row['Improvement_%']:+.1f}%)")
            print(f"   Победитель: {row['Winner']}")
        
        # Сохранение результатов
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Сохранение в JSON
        json_path = output_dir / f"comparison_{timestamp}.json"
        analyzer.save_comparison(
            comparison_df,
            str(json_path),
            rag_stats={"avg_score": rag_result.get_average_score()},
            graphrag_stats={"avg_score": graphrag_result.get_average_score()}
        )
        
        # Сохранение в CSV для удобства анализа
        csv_path = output_dir / f"comparison_{timestamp}.csv"
        comparison_df.to_csv(csv_path, index=False)
        
        logger.info(f"Результаты сравнения сохранены:")
        logger.info(f"  JSON: {json_path}")
        logger.info(f"  CSV:  {csv_path}")
        
        print(f"\n✅ Результаты сравнения сохранены в:")
        print(f"   {json_path}")
        print(f"   {csv_path}")
        
        logger.info("Сравнение успешно завершено")
        
    except Exception as e:
        logger.error(f"Ошибка выполнения сравнения: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
    