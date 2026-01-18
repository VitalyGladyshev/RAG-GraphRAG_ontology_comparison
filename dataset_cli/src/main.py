import argparse
import logging
from pathlib import Path
import sys

from .axioms_processor import AxiomsProcessor
from .llm_client import UniversalLLMClient
from .dataset_generator import DatasetGenerator
from .dataset_validator import DatasetValidator
from .dataset_visualizer import DatasetVisualizer

def setup_logging(verbose: bool = False):
    """Настройка логирования."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def generate_dataset(args):
    """Команда генерации датасета."""
    # Инициализация компонентов
    axioms_processor = AxiomsProcessor(args.axioms_file)
    llm_client = UniversalLLMClient(
        model=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    dataset_generator = DatasetGenerator(axioms_processor, llm_client)
    
    # Генерация датасета
    test_cases = dataset_generator.create_dataset(
        save_dir=args.output_dir,
        save_prefix=args.prefix,
        object_range=(args.start_id, args.end_id) if not args.auto_detect else None,
        auto_detect_objects=args.auto_detect,
        query_template=args.query_template,
        batch_size=args.batch_size,
        verbose=not args.quiet
    )
    
    print(f"\nГенерация датасета завершена! Создано {len(test_cases)} тестовых случаев.")

def validate_dataset(args):
    """Команда валидации датасета."""
    validator = DatasetValidator()
    visualizer = DatasetVisualizer()
    
    test_cases = visualizer.load_test_cases(args.dataset_file)
    validation_result = validator.validate_test_cases(test_cases)
    
    print(f"\nРезультат валидации: {'УСПЕШНО' if validation_result['is_valid'] else 'НЕУДАЧНО'}")
    print(f"   Валидные случаи: {validation_result['stats']['valid']}/{validation_result['stats']['total']}")
    
    if validation_result['issues']:
        print(f"\nНайдено {len(validation_result['issues'])} проблем:")
        for i, issue in enumerate(validation_result['issues'][:10]):
            print(f"   {i+1}. {issue}")
        if len(validation_result['issues']) > 10:
            print(f"   ... и еще {len(validation_result['issues']) - 10} проблем")

def preview_dataset(args):
    """Команда предварительного просмотра датасета."""
    visualizer = DatasetVisualizer()
    test_cases = visualizer.load_test_cases(args.dataset_file)
    visualizer.preview_test_cases(test_cases, n=args.num_preview)

def test_search(args):
    """Команда тестирования поиска аксиом."""
    axioms_processor = AxiomsProcessor(args.axioms_file)
    
    print("\n" + "=" * 80)
    print("ТЕСТ ПОИСКА АКСИОМ")
    print("=" * 80)
    
    for obj_id in args.object_ids:
        found = axioms_processor.find_axioms_for_object(obj_id)
        print(f"\nОбъект: {obj_id.upper()}")
        print(f"   Найдено аксиом: {len(found)}")
        
        if found:
            print("   Примеры (первые 3):")
            for i, ax in enumerate(found[:3]):
                print(f"      {i+1}. {ax}")
            if len(found) > 3:
                print(f"      ... и еще {len(found) - 3} аксиом")
        else:
            print("   Не найдено аксиом")
    
    print("=" * 80)

def main():
    """Основная точка входа."""
    parser = argparse.ArgumentParser(
        description="CLI инструмент для генерации датасетов из аксиом",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Включить детальное логирование')
    
    subparsers = parser.add_subparsers(dest='command', help='Команды')
    
    # Команда generate
    gen_parser = subparsers.add_parser('generate', help='Генерация датасета для оценки')
    gen_parser.add_argument('--axioms-file', required=True, help='Путь к файлу с аксиомами')
    gen_parser.add_argument('--output-dir', default='datasets', help='Директория для вывода')
    gen_parser.add_argument('--prefix', default='rag_test_cases', help='Префикс для имен файлов')
    gen_parser.add_argument('--auto-detect', action='store_true', default=True,
                           help='Автоопределение объектов из аксиом')
    gen_parser.add_argument('--start-id', type=int, default=1, help='Начальный ID объекта')
    gen_parser.add_argument('--end-id', type=int, default=199, help='Конечный ID объекта')
    gen_parser.add_argument('--query-template', 
                           default='Расскажите всё о {object_id} и его отношениях в Attempto Controlled English.',
                           help='Шаблон запроса с плейсхолдером {object_id}')
    gen_parser.add_argument('--batch-size', type=int, default=10,
                           help='Сохранять промежуточные результаты каждые N случаев')
    gen_parser.add_argument('--quiet', action='store_true', help='Подавить вывод прогресса')
    
    # Аргументы LLM
    gen_parser.add_argument('--model', default='google/gemma-3-27b-it',
                           help='Название модели LLM')
    gen_parser.add_argument('--api-key', required=True, help='API ключ')
    gen_parser.add_argument('--api-base', default='https://api.vsegpt.ru/v1',
                           help='Базовый URL API')
    gen_parser.add_argument('--temperature', type=float, default=0.1,
                           help='Температура генерации')
    gen_parser.add_argument('--max-tokens', type=int, default=2048,
                           help='Максимальное количество токенов для генерации')
    
    # Команда validate
    val_parser = subparsers.add_parser('validate', help='Валидация датасета')
    val_parser.add_argument('--dataset-file', required=True, help='Файл датасета для валидации')
    
    # Команда preview
    prev_parser = subparsers.add_parser('preview', help='Предварительный просмотр датасета')
    prev_parser.add_argument('--dataset-file', required=True, help='Файл датасета для просмотра')
    prev_parser.add_argument('--num-preview', type=int, default=5, help='Количество случаев для просмотра')
    
    # Команда test-search
    test_parser = subparsers.add_parser('test-search', help='Тестирование поиска аксиом')
    test_parser.add_argument('--axioms-file', required=True, help='Путь к файлу с аксиомами')
    test_parser.add_argument('--object-ids', nargs='+', required=True,
                            help='ID объектов для тестирования (например, I1 I17 I101)')
    
    # Разбор аргументов
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Настройка логирования
    setup_logging(args.verbose)
    
    # Выполнение команды
    try:
        if args.command == 'generate':
            generate_dataset(args)
        elif args.command == 'validate':
            validate_dataset(args)
        elif args.command == 'preview':
            preview_dataset(args)
        elif args.command == 'test-search':
            test_search(args)
        else:
            parser.print_help()
            sys.exit(1)
    except Exception as e:
        logging.error(f"Ошибка выполнения команды: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
    