import json
from pathlib import Path
from typing import List, Dict, Any

class DatasetVisualizer:
    """Визуализатор для просмотра и анализа датасетов."""
    
    def __init__(self, output_format: str = "console"):
        """
        Инициализация визуализатора.
        
        Args:
            output_format: Формат вывода (console, html, etc.)
        """
        self.output_format = output_format
    
    def preview_test_cases(self, test_cases: List[Dict[str, Any]], n: int = 5):
        """
        Предварительный просмотр тестовых случаев в консольном формате.
        
        Args:
            test_cases: Список тестовых случаев
            n: Количество случаев для просмотра
        """
        if not test_cases:
            print("Нет тестовых случаев для отображения")
            return
        
        print("\n" + "=" * 80)
        print(f"ПРЕДВАРИТЕЛЬНЫЙ ПРОСМОТР ТЕСТОВЫХ СЛУЧАЕВ ({min(n, len(test_cases))} из {len(test_cases)})")
        print("=" * 80)
        
        for i, tc in enumerate(test_cases[:n]):
            metadata = tc.get("metadata", {})
            object_id = metadata.get("object_id", "Неизвестно")
            num_axioms = metadata.get("num_axioms", 0)
            
            print(f"\nСлучай #{i+1}: {object_id} ({num_axioms} аксиом)")
            print(f"   Запрос: {tc['query']}")
            print(f"   Эталонный ответ ({len(tc['ground_truth'])} символов):")
            print(f"      {tc['ground_truth'][:200]}{'...' if len(tc['ground_truth']) > 200 else ''}")
        
        if len(test_cases) > n:
            print(f"\n... и еще {len(test_cases) - n} случаев")
        
        print("=" * 80)
    
    def load_test_cases(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Загрузка тестовых случаев из файла.
        
        Args:
            filepath: Путь к файлу с тестовыми случаями
        
        Returns:
            Список тестовых случаев
        
        Raises:
            FileNotFoundError: Если файл не найден
            ValueError: Если формат файла не поддерживается
        """
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")
        
        print(f"Загрузка тестовых случаев: {path}")
        
        if path.suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        raise ValueError(f"Неподдерживаемый формат файла: {path.suffix}")
    