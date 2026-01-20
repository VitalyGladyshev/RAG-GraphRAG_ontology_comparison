import argparse
import logging
from pathlib import Path
import sys

from src.config.config import PipelineConfig
from src.rag.rag_pipeline import RAGPipeline
from src.utils.logger import setup_logger
from src.utils.dataset_loader import load_test_cases_from_file

def main():
    """Основная функция для запуска RAG пайплайна."""
    parser = argparse.ArgumentParser(description="RAG пайплайн для поиска и генерации ответов")
    
    # Аргументы для данных
    parser.add_argument('--data-dir', type=str, default='data/chunks',
                        help='Директория с текстовыми файлами')
    parser.add_argument('--db-path', type=str, default='vector_db',
                        help='Путь к векторной базе данных')
    
    # Аргументы для LLM
    parser.add_argument('--api-key', type=str, required=True,
                        help='API ключ для LLM')
    parser.add_argument('--api-base', type=str, default='https://api.vsegpt.ru/v1',
                        help='Base URL API')
    parser.add_argument('--model', type=str, default='google/gemma-3-27b-it',
                        help='Модель для генерации')
    
    # Аргументы для поиска
    parser.add_argument('--query', type=str,
                        help='Запрос для поиска и генерации ответа')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Количество возвращаемых результатов')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Порог сходства для фильтрации результатов')
    
    # Аргументы для обработки данных
    parser.add_argument('--rebuild-index', action='store_true',
                        help='Перестроить индекс заново')
    parser.add_argument('--max-files', type=int,
                        help='Максимальное количество обрабатываемых файлов')
    
    # Аргументы для логирования
    parser.add_argument('--log-level', type=str, default='INFO',
                        help='Уровень логирования')
    parser.add_argument('--log-file', type=str,
                        help='Файл для сохранения логов')
    
    args = parser.parse_args()
    
    # Настройка логирования
    logger = setup_logger("run_rag", args.log_file, args.log_level)
    
    try:
        # Загрузка конфигурации
        config = PipelineConfig()
        config.data.data_dir = args.data_dir
        config.data.db_path = args.db_path
        config.llm.api_key = args.api_key
        config.llm.api_base = args.api_base
        config.llm.generation_model = args.model
        config.vector_store.top_k = args.top_k
        config.vector_store.similarity_threshold = args.threshold
        config.log_level = args.log_level
        
        # Инициализация пайплайна
        pipeline = RAGPipeline(config)
        
        # Перестроение индекса при необходимости
        if args.rebuild_index:
            logger.info("Перестроение индекса...")
            df = pipeline.load_and_process_data(max_files=args.max_files)
            if not df.empty:
                pipeline.build_vector_store(df, clear_existing=True)
                logger.info("Индекс успешно перестроен")
            else:
                logger.error("Не удалось загрузить данные для перестроения индекса")
                sys.exit(1)
        
        # Выполнение запроса
        if args.query:
            logger.info(f"Выполнение запроса: {args.query}")
            
            # Установка LLM клиента
            from src.evaluation.ragas_evaluator import RAGASEvaluator
            evaluator = RAGASEvaluator(config.llm)
            pipeline.set_llm_client(evaluator.llm_client)
            
            # Выполнение запроса
            result = pipeline.query(
                query=args.query,
                top_k=args.top_k,
                score_threshold=args.threshold,
                return_details=True
            )
            
            # Вывод результата
            print("\n" + "="*80)
            print("РЕЗУЛЬТАТ ЗАПРОСА")
            print("="*80)
            print(f"Вопрос: {args.query}")
            print(f"\nОтвет: {result['answer']}")
            
            if result['contexts']:
                print(f"\nНайдено контекстов: {len(result['contexts'])}")
                print("\nКонтексты:")
                for i, (ctx, source, score) in enumerate(zip(
                    result['contexts'], 
                    result['sources'], 
                    result['scores']
                )):
                    print(f"\nКонтекст #{i+1} (Источник: {source}, Score: {score:.4f}):")
                    print(f"{ctx[:300]}..." if len(ctx) > 300 else ctx)
            
            print("="*80)
        
        logger.info("RAG пайплайн успешно завершен")
        
    except Exception as e:
        logger.error(f"Ошибка выполнения RAG пайплайна: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
    