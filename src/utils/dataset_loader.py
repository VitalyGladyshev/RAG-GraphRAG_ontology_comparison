import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import logging

logger = logging.getLogger(__name__)

def load_test_cases_from_file(
    filepath: str,
    max_cases: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Загружает тестовые кейсы из файла.
    
    Args:
        filepath: Путь к файлу с тестовыми кейсами
        max_cases: Максимальное количество загружаемых кейсов
    
    Returns:
        Список тестовых кейсов
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Файл не найден: {filepath}")
    
    logger.info(f"Загрузка тестовых кейсов: {filepath}")
    
    # JSON - прямая загрузка
    if filepath.suffix == '.json':
        with open(filepath, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
        logger.info(f"Загружено {len(test_cases)} тестовых кейсов из JSON")
        return test_cases[:max_cases] if max_cases else test_cases
    
    # CSV или Parquet - конвертируем из DataFrame
    if filepath.suffix == '.csv':
        df = pd.read_csv(filepath, encoding='utf-8')
    elif filepath.suffix == '.parquet':
        df = pd.read_parquet(filepath)
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {filepath.suffix}")
    
    # Конвертируем в список словарей
    test_cases = []
    for _, row in df.iterrows():
        test_case = {
            "query": row["query"],
            "ground_truth": row["ground_truth"] if "ground_truth" in row else "",
            "metadata": {}
        }
        
        # Добавляем метаданные
        if "object_id" in row:
            test_case["metadata"]["object_id"] = row["object_id"]
        if "num_axioms" in row:
            test_case["metadata"]["num_axioms"] = int(row["num_axioms"])
        if "axioms" in row and pd.notna(row["axioms"]):
            axioms_str = str(row["axioms"])
            test_case["metadata"]["axioms"] = axioms_str.split("\n") if axioms_str else []
        
        test_cases.append(test_case)
    
    logger.info(f"Загружено {len(test_cases)} тестовых кейсов")
    return test_cases[:max_cases] if max_cases else test_cases

def save_evaluation_dataset(
    dataset: pd.DataFrame,
    save_path: str,
    compression: str = "gzip"
) -> Path:
    """
    Сохраняет датасет для оценки.
    
    Args:
        dataset: DataFrame с данными
        save_path: Путь для сохранения
        compression: Метод сжатия (для Parquet)
    
    Returns:
        Путь к сохраненному файлу
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if save_path.suffix == '.parquet':
        dataset.to_parquet(save_path, index=False, compression=compression)
    elif save_path.suffix == '.csv':
        dataset.to_csv(save_path, index=False, encoding='utf-8')
    elif save_path.suffix == '.json':
        dataset.to_json(save_path, orient='records', lines=True, force_ascii=False)
    else:
        # По умолчанию используем Parquet
        save_path = save_path.with_suffix('.parquet')
        dataset.to_parquet(save_path, index=False, compression=compression)
    
    logger.info(f"Датасет сохранен: {save_path}")
    logger.info(f"Размер: {save_path.stat().st_size / 1024:.1f} KB")
    return save_path
