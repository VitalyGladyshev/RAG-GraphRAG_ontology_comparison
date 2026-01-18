import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd

from .axioms_processor import AxiomsProcessor
from .llm_client import UniversalLLMClient

class DatasetGenerator:
    """Генератор датасетов для оценки RAG-систем на основе аксиом."""
    
    def __init__(self, axioms_processor: AxiomsProcessor, llm_client: UniversalLLMClient):
        """
        Инициализация генератора датасетов.
        
        Args:
            axioms_processor: Процессор аксиом
            llm_client: Клиент для работы с LLM
        """
        self.axioms_processor = axioms_processor
        self.llm_client = llm_client
    
    def generate_ground_truth(self, query: str, axioms: List[str], max_retries: int = 3) -> str:
        """
        Генерация эталонного ответа с использованием LLM на основе запроса и аксиом.
        
        Args:
            query: Запрос пользователя
            axioms: Список релевантных аксиом
            max_retries: Максимальное количество попыток генерации
        
        Returns:
            Сгенерированный эталонный ответ
        """
        system_prompt = """You are an expert in knowledge representation and Attempto Controlled English (ACE).
Your task is to provide accurate, comprehensive answers based on the given axioms.

Instructions:
1. Analyze all provided axioms carefully
2. Describe what the object is (its type/class)
3. List all properties and their values
4. Explain relationships with other objects
5. Use clear, natural English while being precise
6. Be comprehensive but concise"""

        axioms_text = "\n".join(axioms)
        user_prompt = f"""Based on these axioms in Attempto Controlled English format:

{axioms_text}

{query}

Provide a comprehensive answer following the instructions above."""

        for attempt in range(max_retries):
            try:
                ground_truth = self.llm_client.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=0.3,
                    max_tokens=2048
                )
                
                if ground_truth and len(ground_truth.strip()) > 10:
                    return ground_truth.strip()
                else:
                    print(f"Попытка {attempt + 1}: пустой ответ, повторная попытка...")
                    time.sleep(1)
                    
            except Exception as e:
                print(f"Попытка {attempt + 1}: ошибка - {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    print("Все попытки исчерпаны")
                    return ""
        
        return ""
    
    def create_dataset(self,
                      save_dir: str = "datasets",
                      save_prefix: str = "rag_test_cases",
                      object_range: Optional[Tuple[int, int]] = None,
                      auto_detect_objects: bool = True,
                      query_template: str = "Расскажите всё о {object_id} и его отношениях в Attempto Controlled English.",
                      batch_size: int = 10,
                      verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Создание датасета для оценки.
        
        Args:
            save_dir: Директория для сохранения результатов
            save_prefix: Префикс для имен файлов
            object_range: Диапазон ID объектов (начало, конец)
            auto_detect_objects: Автоматическое определение объектов из аксиом
            query_template: Шаблон запроса с плейсхолдером {object_id}
            batch_size: Размер пакета для промежуточного сохранения
            verbose: Вывод подробной информации
        
        Returns:
            Список сгенерированных тестовых случаев
        """
        # Определение объектов для обработки
        if auto_detect_objects:
            print("\nАвтоопределение объектов из аксиом...")
            object_ids_to_process = self.axioms_processor.extract_unique_objects()
            print(f"Найдено {len(object_ids_to_process)} уникальных объектов")
        else:
            if object_range is None:
                object_range = (1, 199)
            start_idx, end_idx = object_range
            object_ids_to_process = list(range(start_idx, end_idx))
            print(f"\nОбработка объектов I{start_idx} - I{end_idx-1}")
        
        print(f"Всего объектов для обработки: {len(object_ids_to_process)}")
        
        # Создание директории для сохранения
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Обработка объектов
        test_cases = []
        skipped_objects = []
        
        print("\n" + "=" * 80)
        print("НАЧАЛО ГЕНЕРАЦИИ ДАТАСЕТА")
        print("=" * 80)
        
        for idx, obj_num in enumerate(object_ids_to_process):
            object_id_lower = f"i{obj_num}"
            object_id_display = f"I{obj_num}"
            
            # Поиск релевантных аксиом
            found_axioms = self.axioms_processor.find_axioms_for_object(object_id_lower)
            
            if not found_axioms:
                skipped_objects.append(object_id_display)
                if verbose:
                    print(f"Объект {object_id_display}: не найдены аксиомы, пропущен")
                continue
            
            # Генерация запроса
            query = query_template.format(object_id=object_id_display)
            
            if verbose:
                print(f"Объект {object_id_display}: генерация эталонного ответа ({len(found_axioms)} аксиом)...")
            
            # Генерация эталонного ответа
            ground_truth = self.generate_ground_truth(query, found_axioms)
            
            if not ground_truth:
                skipped_objects.append(object_id_display)
                print(f"Объект {object_id_display}: ошибка генерации, пропущен")
                continue
            
            # Создание тестового случая
            test_case = {
                "query": query,
                "ground_truth": ground_truth,
                "metadata": {
                    "object_id": object_id_display,
                    "num_axioms": len(found_axioms),
                    "axioms": found_axioms
                }
            }
            
            test_cases.append(test_case)
            
            if verbose:
                print(f"Объект {object_id_display}: успешно ({len(ground_truth)} символов)")
            
            # Промежуточное сохранение
            if (idx + 1) % batch_size == 0 and test_cases:
                self._save_intermediate_results(test_cases, save_path, save_prefix)
            
            # Задержка для ограничения частоты запросов
            time.sleep(0.3)
        
        # Финальная статистика
        print("\n" + "=" * 80)
        print("ГЕНЕРАЦИЯ ДАТАСЕТА ЗАВЕРШЕНА")
        print("=" * 80)
        print(f"Статистика:")
        print(f"   • Создано тестовых случаев: {len(test_cases)}")
        print(f"   • Пропущено объектов: {len(skipped_objects)}")
        print(f"   • Процент успеха: {len(test_cases) / len(object_ids_to_process) * 100:.1f}%")
        
        if skipped_objects and verbose:
            print(f"\nПропущенные объекты ({len(skipped_objects)}):")
            print(f"   {', '.join(skipped_objects[:30])}")
            if len(skipped_objects) > 30:
                print(f"   ... и еще {len(skipped_objects) - 30} объектов")
        
        # Сохранение финальных результатов
        if test_cases:
            self._save_final_results(test_cases, save_path, save_prefix)
        
        return test_cases
    
    def _save_intermediate_results(self, test_cases: List[Dict], save_path: Path, prefix: str):
        """Сохранение промежуточных результатов."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_path = save_path / f"{prefix}_temp_{timestamp}.json"
        
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(test_cases, f, ensure_ascii=False, indent=2)
        
        print(f"Промежуточное сохранение: {temp_path} ({len(test_cases)} случаев)")
    
    def _save_final_results(self, test_cases: List[Dict], save_path: Path, prefix: str):
        """Сохранение финальных результатов в различных форматах."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Подготовка данных для DataFrame
        df_data = []
        for tc in test_cases:
            df_data.append({
                "query": tc["query"],
                "ground_truth": tc["ground_truth"],
                "object_id": tc["metadata"]["object_id"],
                "num_axioms": tc["metadata"]["num_axioms"],
                "axioms": "\n".join(tc["metadata"]["axioms"])
            })
        
        df = pd.DataFrame(df_data)
        
        print("\nСохранение результатов:")
        print("-" * 80)
        
        # 1. CSV
        csv_path = save_path / f"{prefix}_{timestamp}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"CSV: {csv_path}")
        print(f"   Размер: {csv_path.stat().st_size / 1024:.1f} КБ")
        
        # 2. Parquet
        parquet_path = save_path / f"{prefix}_{timestamp}.parquet"
        df.to_parquet(parquet_path, index=False, compression='gzip')
        print(f"Parquet: {parquet_path}")
        print(f"   Размер: {parquet_path.stat().st_size / 1024:.1f} КБ")
        
        # 3. JSON
        json_path = save_path / f"{prefix}_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(test_cases, f, ensure_ascii=False, indent=2)
        print(f"JSON: {json_path}")
        print(f"   Размер: {json_path.stat().st_size / 1024:.1f} КБ")
        
        # 4. Статистика датасета
        print(f"\nСтатистика датасета:")
        print(f"   • Всего случаев: {len(df)}")
        print(f"   • Средняя длина запроса: {df['query'].str.len().mean():.0f} символов")
        print(f"   • Средняя длина эталонного ответа: {df['ground_truth'].str.len().mean():.0f} символов")
        print(f"   • Среднее количество аксиом на случай: {df['num_axioms'].mean():.1f}")
        print(f"   • Мин/макс количество аксиом: {df['num_axioms'].min()}/{df['num_axioms'].max()}")
        
        return csv_path, parquet_path, json_path
    