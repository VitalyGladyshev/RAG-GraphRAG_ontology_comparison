from typing import List, Dict, Any

class DatasetValidator:
    """Валидатор для проверки качества тестовых случаев."""
    
    def validate_test_cases(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Валидация тестовых случаев и формирование отчета.
        
        Args:
            test_cases: Список тестовых случаев
        
        Returns:
            Словарь с отчетом о валидации
        """
        issues = []
        stats = {
            "total": len(test_cases),
            "valid": 0,
            "empty_query": 0,
            "empty_ground_truth": 0,
            "short_ground_truth": 0,
            "missing_metadata": 0
        }
        
        for i, tc in enumerate(test_cases):
            # Проверка запроса
            if not tc.get("query") or not tc["query"].strip():
                issues.append(f"Случай #{i}: пустой запрос")
                stats["empty_query"] += 1
                continue
            
            # Проверка эталонного ответа
            if not tc.get("ground_truth") or not tc["ground_truth"].strip():
                issues.append(f"Случай #{i}: пустой эталонный ответ")
                stats["empty_ground_truth"] += 1
                continue
            
            # Проверка длины эталонного ответа
            if len(tc["ground_truth"]) < 50:
                issues.append(f"Случай #{i}: короткий эталонный ответ ({len(tc['ground_truth'])} символов)")
                stats["short_ground_truth"] += 1
            
            # Проверка метаданных
            if not tc.get("metadata"):
                issues.append(f"Случай #{i}: отсутствуют метаданные")
                stats["missing_metadata"] += 1
            
            stats["valid"] += 1
        
        return {
            "stats": stats,
            "issues": issues,
            "is_valid": len(issues) == 0
        }
    
    def get_summary(self, test_cases: List[Dict[str, Any]]) -> str:
        """
        Получение сводной статистики по тестовым случаям.
        
        Args:
            test_cases: Список тестовых случаев
        
        Returns:
            Форматированная строка со статистикой
        """
        if not test_cases:
            return "Нет тестовых случаев"
        
        num_axioms_list = [tc["metadata"]["num_axioms"] for tc in test_cases]
        gt_lengths = [len(tc["ground_truth"]) for tc in test_cases]
        
        summary = "=" * 80 + "\n"
        summary += "СВОДКА ПО ТЕСТОВЫМ СЛУЧАЯМ\n"
        summary += "=" * 80 + "\n"
        summary += f"Всего случаев: {len(test_cases)}\n\n"
        summary += "Аксиом на случай:\n"
        summary += f"  • Среднее: {sum(num_axioms_list) / len(num_axioms_list):.1f}\n"
        summary += f"  • Минимум: {min(num_axioms_list)}\n"
        summary += f"  • Максимум: {max(num_axioms_list)}\n\n"
        summary += "Длина эталонного ответа (символы):\n"
        summary += f"  • Среднее: {sum(gt_lengths) / len(gt_lengths):.0f}\n"
        summary += f"  • Минимум: {min(gt_lengths)}\n"
        summary += f"  • Максимум: {max(gt_lengths)}\n"
        summary += "=" * 80
        
        return summary
    