import argparse
import logging
from pathlib import Path
import sys

from src.config.config import PipelineConfig
from src.graphrag.graphrag_pipeline import GraphRAGPipeline
from src.utils.logger import setup_logger

def main():
    """Основная функция для запуска GraphRAG пайплайна."""
    parser = argparse.ArgumentParser(description="GraphRAG пайплайн для работы с графовой структурой данных")
    
    # Аргументы для данных
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Директория с обработанными данными GraphRAG')
    
    # Аргументы для LLM
    parser.add_argument('--api-key', type=str, required=True,
                        help='API ключ для LLM')
    parser.add_argument('--api-base', type=str, default='https://api.vsegpt.ru/v1',
                        help='Base URL API')
    parser.add_argument('--model', type=str, default='google/gemma-3-27b-it',
                        help='Модель для генерации')
    
    # Аргументы для запроса
    parser.add_argument('--query', type=str,
                        help='Запрос для выполнения')
    
    # Аргументы для логирования
    parser.add_argument('--log-level', type=str, default='INFO',
                        help='Уровень логирования')
    parser.add_argument('--log-file', type=str,
                        help='Файл для сохранения логов')
    
    args = parser.parse_args()
    
    # Настройка логирования
    logger = setup_logger("run_graphrag", args.log_file, args.log_level)
    
    try:
        # Загрузка конфигурации
        config = PipelineConfig()
        config.llm.api_key = args.api_key
        config.llm.api_base = args.api_base
        config.llm.generation_model = args.model
        config.log_level = args.log_level
        
        # Инициализация пайплайна
        pipeline = GraphRAGPipeline(config)
        
        # Инициализация из директории
        logger.info(f"Инициализация GraphRAG из директории: {args.input_dir}")
        pipeline.init_from_directory(
            input_dir=args.input_dir,
            llm_config={
                "api_key": args.api_key,
                "api_base": args.api_base,
                "model": args.model
            }
        )
        
        # Выполнение запроса
        if args.query:
            logger.info(f"Выполнение запроса: {args.query}")
            
            # Выполнение запроса
            result = pipeline.query(
                question=args.query,
                return_details=True
            )
            
            # Вывод результата
            print("\n" + "="*80)
            print("РЕЗУЛЬТАТ ЗАПРОСА GraphRAG")
            print("="*80)
            print(f"Вопрос: {args.query}")
            print(f"\nОтвет: {result['answer']}")
            
            if result['contexts']:
                print(f"\nНайдено контекстов: {len(result['contexts'])}")
                print("\nКонтексты:")
                for i, ctx in enumerate(result['contexts']):
                    print(f"\nКонтекст #{i+1}:")
                    print(f"{ctx[:300]}..." if len(ctx) > 300 else ctx)
            
            # Вывод статистики
            if 'context_data' in result:
                stats = result['context_data']
                print(f"\nСтатистика контекста:")
                print(f"  Сущностей: {stats.get('num_entities', 0)}")
                print(f"  Связей: {stats.get('num_relationships', 0)}")
                print(f"  Отчетов: {stats.get('num_reports', 0)}")
                print(f"  Источников: {stats.get('num_sources', 0)}")
                print(f"  Время выполнения: {result['elapsed_time']:.2f} сек")
            
            print("="*80)
        
        logger.info("GraphRAG пайплайн успешно завершен")
        
    except Exception as e:
        logger.error(f"Ошибка выполнения GraphRAG пайплайна: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
    