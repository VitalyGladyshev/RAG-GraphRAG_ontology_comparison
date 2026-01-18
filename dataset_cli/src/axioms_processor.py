import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

class AxiomsProcessor:
    """Класс для обработки аксиом: загрузка, поиск и извлечение объектов."""
    
    def __init__(self, axioms_filepath: str):
        """
        Инициализация процессора аксиом.
        
        Args:
            axioms_filepath: Путь к файлу с аксиомами
        """
        self.axioms_filepath = Path(axioms_filepath)
        self.axioms = self.load_axioms()
        
    def load_axioms(self) -> List[str]:
        """Загрузка аксиом из текстового файла."""
        if not self.axioms_filepath.exists():
            raise FileNotFoundError(f"Файл не найден: {self.axioms_filepath}")
        
        with open(self.axioms_filepath, 'r', encoding='utf-8') as f:
            axioms = [line.strip() for line in f if line.strip()]
        
        print(f"Загружено {len(axioms)} аксиом из {self.axioms_filepath}")
        return axioms
    
    def find_axioms_for_object(self, object_id: str) -> List[str]:
        """
        Поиск всех аксиом, содержащих указанный объект.
        
        Args:
            object_id: Идентификатор объекта (например, 'i1', 'i17')
        
        Returns:
            Список аксиом, содержащих объект
        """
        object_id = object_id.lower()
        pattern = re.compile(
            rf'\b{re.escape(object_id)}(?=\s|$|_|\.|,)',
            re.IGNORECASE
        )
        
        found_axioms = []
        for axiom in self.axioms:
            if pattern.search(axiom):
                found_axioms.append(axiom)
        
        return found_axioms
    
    def extract_unique_objects(self, max_id: int = 198) -> List[int]:
        """
        Извлечение всех уникальных идентификаторов объектов из аксиом.
        
        Args:
            max_id: Максимальный допустимый ID объекта
        
        Returns:
            Отсортированный список уникальных ID объектов
        """
        object_pattern = re.compile(r'\bi(\d+)(?:_\w+|\b)', re.IGNORECASE)
        object_ids = set()
        
        for axiom in self.axioms:
            matches = object_pattern.finditer(axiom)
            for match in matches:
                obj_id = int(match.group(1))
                if 1 <= obj_id <= max_id:
                    object_ids.add(obj_id)
        
        return sorted(list(object_ids))
    