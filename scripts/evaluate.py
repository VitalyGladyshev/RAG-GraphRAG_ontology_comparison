import argparse
import logging
from pathlib import Path
import sys
import json
from datetime import datetime

from src.config.config import PipelineConfig
from src.rag.rag_pipeline import RAGPipeline
from src.graphrag.graphrag_pipeline import GraphRAGPipeline
from src.utils.logger import setup_logger
from src.utils.dataset_loader import load_test_cases_from_file
from src.comparison.comparison_analyzer import ComparisonAnalyzer

def evaluate_rag(args, config):
    """Оценивает RAG систему."""
    logger = logging.getLogger("evaluate")
    
    # Инициализация пайплайна
    pipeline = RAGPipeline(config)
    
    # Загрузка тестовых кейсов
    test_cases = load_test_cases_from_file(args.dataset, args.max_cases)
    logger.info(f"Загружено {len(test_cases)} тестовых кейсов")
    
    # Установка LLM клиента
    from src.evaluation.ragas_evaluator import RAGASEvaluator
    evaluator = RAGASEvaluator(config.llm)
    pipeline.set_llm_client(evaluator.llm_client)
    
    # Оценка
    logger.info("Запуск оценки RAG системы...")
    result = pipeline.evaluate(
        test_cases=test_cases,
        output_dir=args.output_dir,
        top_k=args.top_k
    )
    
    # Вывод результатов
    print("\n" + "="*80)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ RAG")
    print("="*80)
    for metric in ["faithfulness", "answer_relevancy", "context_precision", 
                  "context_recall", "context_relevance", "response_groundedness", 
                  "answer_accuracy"]:
        value = getattr(result, metric, None)
        if value is not None:
            print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
    
    avg_score = result.get_average_score()
    print(f"\nСредний балл: {avg_score:.4f}")
    print("="*80)
    
    return result

def evaluate_graphrag(args, config):
    """Оценивает GraphRAG систему."""
    logger = logging.getLogger("evaluate")
    
    # Инициализация пайплайна
    pipeline = GraphRAGPipeline(config)
    
    # Инициализация из директории
    pipeline.init_from_directory(
        input_dir=args.graphrag_dir,
        llm_config={
            "api_key": config.llm.api_key,
            "api_base": config.llm.api_base,
            "model": config.llm.generation_model
        }
    )
    
    # Загрузка тестовых кейсов
    test_cases = load_test_cases_from_file(args.dataset, args.max_cases)
    logger.info(f"Загружено {len(test_cases)} тестовых кейсов")
    
    # Оценка
    logger.info("Запуск оценки GraphRAG системы...")
    result = pipeline.evaluate(
        test_cases=test_cases,
        output_dir=args.output_dir
    )
    
    # Вывод результатов
    print("\n" + "="*80)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ GraphRAG")
    print("="*80)
    for metric in ["faithfulness", "answer_relevancy", "context_precision", 
                  "context_recall", "context_relevance", "response_groundedness", 
                  "answer_accuracy"]:
        value = getattr(result, metric, None)
        if value is not None:
            print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
    
    avg_score = result.get_average_score()
    print(f"\nСредний балл: {avg_score:.4f}")
    print("="*80)
    
    return result

def main():
    """Основная функция для оценки систем."""
    parser = argparse.ArgumentParser(description="Оценка RAG и GraphRAG систем с использованием RAGAS")
    
    # Общие аргументы
    parser.add_argument('--dataset', type=str, required=True,
                        help='Путь к файлу с тестовыми кейсами')
    parser.add_argument('--api-key', type=str, required=True,
                        help='API ключ для LLM')
    parser.add_argument('--api-base', type=str, default='https://api.vsegpt.ru/v1',
                        help='Base URL API')
    parser.add_argument('--model', type=str, default='google/gemma-3-27b-it',
                        help='Модель для генерации')
    parser.add_argument('--max-cases', type=int,
                        help='Максимальное количество тестовых кейсов')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Директория для сохранения результатов')
    parser.add_argument('--log-level', type=str, default='INFO',
                        help='Уровень логирования')
    
    # Аргументы для RAG
    parser.add_argument('--data-dir', type=str, default='data/chunks',
                        help='Директория с текстовыми файлами для RAG')
    parser.add_argument('--db-path', type=str, default='vector_db',
                        help='Путь к векторной базе данных для RAG')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Количество контекстов для поиска в RAG')
    
    # Аргументы для GraphRAG
    parser.add_argument('--graphrag-dir', type=str,
                        help='Директория с обработанными данными GraphRAG')
    
    # Режимы работы
    parser.add_argument('--mode', type=str, choices=['rag', 'graphrag', 'compare'],
                        required=True, help='Режим работы: rag, graphrag или compare')
    
    args = parser.parse_args()
    
    # Настройка логирования
    logger = setup_logger("evaluate", None, args.log_level)
    
    try:
        # Загрузка конфигурации
        config = PipelineConfig()
        config.llm.api_key = args.api_key
        config.llm.api_base = args.api_base
        config.llm.generation_model = args.model
        config.log_level = args.log_level
        
        # Режимы работы
        if args.mode == 'rag':
            result = evaluate_rag(args, config)
        
        elif args.mode == 'graphrag':
            if not args.graphrag_dir:
                raise ValueError("Для режима graphrag требуется указать --graphrag-dir")
            result = evaluate_graphrag(args, config)
        
        elif args.mode == 'compare':
            if not args.graphrag_dir:
                raise ValueError("Для режима compare требуется указать --graphrag-dir")
            
            # Оценка RAG
            logger.info("Оценка RAG системы...")
            rag_result = evaluate_rag(args, config)
            
            # Оценка GraphRAG
            logger.info("Оценка GraphRAG системы...")
            graphrag_result = evaluate_graphrag(args, config)
            
            # Сравнение результатов
            logger.info("Сравнение результатов...")
            analyzer = ComparisonAnalyzer()
            comparison_df = analyzer.compare_results(rag_result, graphrag_result)
            
            # Вывод сравнения
            print("\n" + "="*80)
            print("СРАВНЕНИЕ RAG И GraphRAG")
            print("="*80)
            
            # Подсчет побед
            graphrag_wins = len(comparison_df[comparison_df['Winner'] == 'GraphRAG'])
            rag_wins = len(comparison_df[comparison_df['Winner'] == 'RAG'])
            ties = len(comparison_df[comparison_df['Winner'] == 'Tie'])
            
            print(f"GraphRAG побед: {graphrag_wins}")
            print(f"RAG побед: {rag_wins}")
            print(f"Ничьих: {ties}")
            print()
            
            # Вывод детального сравнения
            for _, row in comparison_df.iterrows():
                print(f"{row['Metric'].replace('_', ' ').title()}:")
                print(f"  RAG:       {row['RAG']:.4f}")
                print(f"  GraphRAG:  {row['GraphRAG']:.4f}")
                print(f"  Разница:   {row['Difference']:+.4f} ({row['Improvement_%']:+.1f}%)")
                print(f"  Победитель: {row['Winner']}")
                print()
            
            # Сохранение результатов сравнения
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            comparison_path = Path(args.output_dir) / f"comparison_{timestamp}.json"
            analyzer.save_comparison(
                comparison_df, 
                str(comparison_path),
                rag_stats={"avg_score": rag_result.get_average_score()},
                graphrag_stats={"avg_score": graphrag_result.get_average_score()}
            )
            
            logger.info(f"Результаты сравнения сохранены: {comparison_path}")
        
        logger.info("Оценка успешно завершена")
        
    except Exception as e:
        logger.error(f"Ошибка выполнения оценки: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
    