import unittest
from unittest.mock import mock_open, patch
from src.axioms_processor import AxiomsProcessor

class TestAxiomsProcessor(unittest.TestCase):
    """Тесты для класса AxiomsProcessor."""
    
    def setUp(self):
        """Подготовка тестовых данных."""
        self.test_axioms = [
            "i1 is a class.",
            "i1 has property i2.",
            "i3 relates to i1 through i4.",
            "i10 has value 'test'.",
            "i101 is connected to i200."
        ]
        self.mock_file_content = "\n".join(self.test_axioms)
    
    @patch('pathlib.Path.exists')
    @patch('builtins.open')
    def test_load_axioms_success(self, mock_open_file, mock_exists):
        """Тест успешной загрузки аксиом."""
        mock_exists.return_value = True
        mock_open_file.return_value.read.return_value = self.mock_file_content
        
        processor = AxiomsProcessor("test.txt")
        
        self.assertEqual(len(processor.axioms), 5)
        self.assertEqual(processor.axioms, self.test_axioms)
    
    @patch('pathlib.Path.exists')
    def test_load_axioms_file_not_found(self, mock_exists):
        """Тест обработки отсутствующего файла."""
        mock_exists.return_value = False
        
        with self.assertRaises(FileNotFoundError):
            AxiomsProcessor("nonexistent.txt")
    
    def test_find_axioms_for_object(self):
        """Тест поиска аксиом для объекта."""
        processor = AxiomsProcessor.__new__(AxiomsProcessor)
        processor.axioms = self.test_axioms
        
        # Тест для объекта i1
        result_i1 = processor.find_axioms_for_object("i1")
        self.assertEqual(len(result_i1), 3)
        self.assertIn("i1 is a class.", result_i1)
        self.assertIn("i1 has property i2.", result_i1)
        self.assertIn("i3 relates to i1 through i4.", result_i1)
        
        # Тест для объекта i10
        result_i10 = processor.find_axioms_for_object("i10")
        self.assertEqual(len(result_i10), 1)
        self.assertIn("i10 has value 'test'.", result_i10)
        
        # Тест для объекта i101
        result_i101 = processor.find_axioms_for_object("i101")
        self.assertEqual(len(result_i101), 1)
        self.assertIn("i101 is connected to i200.", result_i101)
        
        # Тест для несуществующего объекта
        result_none = processor.find_axioms_for_object("i999")
        self.assertEqual(len(result_none), 0)
    
    def test_extract_unique_objects(self):
        """Тест извлечения уникальных объектов."""
        processor = AxiomsProcessor.__new__(AxiomsProcessor)
        processor.axioms = self.test_axioms
        
        result = processor.extract_unique_objects(max_id=200)
        
        self.assertEqual(len(result), 5)
        self.assertIn(1, result)   # i1
        self.assertIn(2, result)   # i2
        self.assertIn(3, result)   # i3
        self.assertIn(4, result)   # i4
        self.assertIn(10, result)  # i10
        self.assertIn(101, result) # i101
        self.assertIn(200, result) # i200
        self.assertNotIn(999, result)  # отсутствует в аксиомах

if __name__ == '__main__':
    unittest.main()
