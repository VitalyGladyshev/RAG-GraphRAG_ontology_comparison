# Онтологический подход к оценке GraphRAG в PLM-домене

## Сравнение RAG и GraphRAG оценка с использованием RAGAS

Набор консольных утилит для оценки и сравнения RAG (Retrieval-Augmented Generation) и GraphRAG систем с использованием метрик RAGAS.

## Основные возможности

- **RAG пайплайн**: Поиск по векторной базе и генерация ответов
- **GraphRAG пайплайн**: Работа с графовой структурой данных (сущности, связи, отчеты сообществ)
- **RAGAS оценка**: Оценка качества с использованием 7 метрик:
  - `faithfulness` - Соответствие ответа контексту
  - `answer_relevancy` - Релевантность ответа запросу
  - `context_precision` - Точность контекста
  - `context_recall` - Полнота контекста
  - `context_relevance` - Релевантность контекста
  - `response_groundedness` - Обоснованность ответа
  - `answer_accuracy` - Точность ответа
- **Сравнение результатов**: Детальный анализ и визуализация сравнения RAG и GraphRAG
- **Генерация оценочного датасета**: Создание оценочных наборов данных на основе файлов axiom с использованием LLM

## Установка

### Требования

- Python 3.8+
- GPU с поддержкой CUDA (рекомендуется для работы с эмбеддингами)

### Установка зависимостей

```bash
# Клонирование репозитория
git clone https://github.com/VitalyGladyshev/RAG-GraphRAG_ontology_comparison.git
cd RAG-GraphRAG_ontology_comparison

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate    # Windows

# Установка зависимостей
pip install -r requirements.txt
```

## Этапы оценки

### 1. Подготовка данных для RAG

```bash
# Размещение текстовых файлов в директории data/chunks/
mkdir -p data/chunks
cp your_text_files/*.txt data/chunks/

# Построение векторного индекса
python scripts/run_rag.py \
  --api-key "your-api-key" \
  --rebuild-index \
  --max-files 100
```

### 2. Запуск RAG запроса

```bash
python scripts/run_rag.py \
  --api-key "your-api-key" \
  --query "Расскажи об объекте I101 и его связях"
```

### 3. Запуск GraphRAG запроса

```bash
python scripts/run_graphrag.py \
  --api-key "your-api-key" \
  --input-dir "path/to/graphrag/output" \
  --query "Расскажи об объекте I101 и его связях"
```

### 4. Оценка RAG системы

```bash
python scripts/evaluate.py \
  --mode rag \
  --api-key "your-api-key" \
  --dataset "datasets/rag_test_cases_20260109_222314.json" \
  --output-dir "results/rag_evaluation"
```

### 5. Оценка GraphRAG системы

```bash
python scripts/evaluate.py \
  --mode graphrag \
  --api-key "your-api-key" \
  --dataset "datasets/rag_test_cases_20260109_222314.json" \
  --graphrag-dir "path/to/graphrag/output" \
  --output-dir "results/graphrag_evaluation"
```

### 6. Сравнение RAG и GraphRAG

```bash
python scripts/evaluate.py \
  --mode compare \
  --api-key "your-api-key" \
  --dataset "datasets/rag_test_cases_20260109_222314.json" \
  --graphrag-dir "path/to/graphrag/output" \
  --output-dir "results/comparison"
```

### 7. Сравнение результатов из файлов

```bash
python scripts/compare.py \
  --rag-result "results/rag_evaluation/rag_evaluation_20260110_123456.json" \
  --graphrag-result "results/graphrag_evaluation/graphrag_evaluation_20260110_123456.json" \
  --output-dir "results/final_comparison"
```

## Подробное описание утилит

### run_rag.py

Запускает RAG пайплайн для поиска и генерации ответов.

**Основные аргументы:**

- `--data-dir`: Директория с текстовыми файлами
- `--db-path`: Путь к векторной базе данных
- `--api-key`: API ключ для LLM
- `--query`: Запрос для поиска и генерации ответа
- `--rebuild-index`: Перестроить индекс заново
- `--top-k`: Количество возвращаемых результатов (по умолчанию: 5)
- `--threshold`: Порог сходства для фильтрации (по умолчанию: 0.7)

### run_graphrag.py

Запускает GraphRAG пайплайн для работы с графовой структурой данных.

**Основные аргументы:**

- `--input-dir`: Директория с обработанными данными GraphRAG
- `--api-key`: API ключ для LLM
- `--query`: Запрос для выполнения
- `--model`: Модель для генерации (по умолчанию: google/gemma-3-27b-it)

### evaluate.py

Выполняет оценку RAG или GraphRAG систем с использованием RAGAS.

**Основные аргументы:**

- `--mode`: Режим работы (`rag`, `graphrag`, `compare`)
- `--dataset`: Путь к файлу с тестовыми кейсами
- `--api-key`: API ключ для LLM
- `--output-dir`: Директория для сохранения результатов
- `--max-cases`: Максимальное количество тестовых кейсов
- `--graphrag-dir`: Директория с обработанными данными GraphRAG (для режимов `graphrag` и `compare`)
- `--top-k`: Количество контекстов для поиска в RAG (по умолчанию: 5)

### compare.py

Сравнивает результаты оценки RAG и GraphRAG систем из файлов.

**Основные аргументы:**

- `--rag-result`: Путь к файлу с результатами RAG оценки
- `--graphrag-result`: Путь к файлу с результатами GraphRAG оценки
- `--output-dir`: Директория для сохранения результатов сравнения

## Структура проекта

```
src/
├── config/           # Конфигурации
├── rag/              # RAG пайплайн
├── graphrag/         # GraphRAG пайплайн
├── evaluation/       # Оценка с использованием RAGAS
├── comparison/       # Сравнение результатов
├── utils/            # Вспомогательные утилиты
scripts/              # Скрипты запуска конвейеров
dataset_cli/          # Генератор оценочного датасета
images/               # Изображения для оформления
results/              # Результаты оценок и сравнений
data/                 # Исходные данные
requirements.txt      # Зависимости
```

## Подготовка тестовых данных

Для оценки систем необходимы тестовые кейсы в формате JSON. Пример структуры файла:

```json
[
  {
    "query": "Расскажите об объекте I101 и его связях в Attempto Controlled English",
    "ground_truth": "i101 - это length_measure_with_unit, который представляет измерение длины с единицей измерения. i101 имеет два компонента: i101_value_component (численное значение) и i17 (единица измерения). i101 также служит коэффициентом преобразования для i103.",
    "metadata": {
      "object_id": "I101",
      "num_axioms": 5,
      "axioms": [
        "i101 measure_with_unit_has_unit_component i17.",
        "i101 measure_with_unit_has_value_component i101_value_component.",
        "i103 conversion_based_unit_has_conversion_factor i101."
      ]
    }
  }
]
```

Для подготовки тестовых наборов из аксиом используется утилита dataset_cli.

## Интерпретация результатов

### Метрики RAGAS

- **faithfulness** (0-1): Насколько ответ соответствует предоставленному контексту
- **answer_relevancy** (0-1): Насколько ответ релевантен запросу
- **context_precision** (0-1): Точность выбранных контекстов
- **context_recall** (0-1): Полнота выбранных контекстов относительно ground truth
- **context_relevance** (0-1): Релевантность контекстов запросу
- **response_groundedness** (0-1): Насколько ответ обоснован контекстом
- **answer_accuracy** (0-1): Точность ответа относительно ground truth

### Сравнение RAG и GraphRAG

Результаты сравнения включают:
- Сводную статистику по победам
- Детальное сравнение по каждой метрике
- Процент улучшения GraphRAG относительно RAG
- Время выполнения запросов

## Требования к данным GraphRAG

Для работы GraphRAG необходимы обработанные данные в следующем формате:
- `entities.parquet` - сущности
- `communities.parquet` - сообщества
- `community_reports.parquet` - отчеты сообществ
- `text_units.parquet` - текстовые единицы
- `relationships.parquet` - связи
- `lancedb/` - векторная база данных для эмбеддингов сущностей

Эти файлы должны быть сгенерированы с помощью Microsoft GraphRAG pipeline.

## Настройка конфигурации

Для настройки параметров системы можно создать файл конфигурации `config.json`:

```json
{
  "data": {
    "data_dir": "data/chunks",
    "db_path": "vector_db",
    "results_dir": "results"
  },
  "embeddings": {
    "model_name": "intfloat/multilingual-e5-large",
    "use_fp16": true
  },
  "vector_store": {
    "top_k": 5,
    "similarity_threshold": 0.7
  },
  "llm": {
    "api_base": "https://api.vsegpt.ru/v1",
    "generation_model": "google/gemma-3-27b-it"
  },
  "ragas": {
    "metrics": [
      "faithfulness",
      "answer_relevancy",
      "context_precision",
      "context_recall",
      "context_relevance",
      "response_groundedness",
      "answer_accuracy"
    ]
  }
}
```

## Пример сквозного сравнения

```bash
# 1. Подготовка данных для RAG (если еще не сделано)
python scripts/run_rag.py \
  --api-key "sk-or-vv-your-api-key-here" \
  --rebuild-index \
  --max-files 50

# 2. Оценка RAG системы
python scripts/evaluate.py \
  --mode rag \
  --api-key "sk-or-vv-your-api-key-here" \
  --dataset "datasets/rag_test_cases_20260109_222314.json" \
  --output-dir "results/rag_evaluation"

# 3. Оценка GraphRAG системы
python scripts/evaluate.py \
  --mode graphrag \
  --api-key "sk-or-vv-your-api-key-here" \
  --dataset "datasets/rag_test_cases_20260109_222314.json" \
  --graphrag-dir "graph_local3/output" \
  --output-dir "results/graphrag_evaluation"

# 4. Сравнение результатов
python scripts/evaluate.py \
  --mode compare \
  --api-key "sk-or-vv-your-api-key-here" \
  --dataset "datasets/rag_test_cases_20260109_222314.json" \
  --graphrag-dir "graph_local3/output" \
  --output-dir "results/final_comparison"

# ИЛИ (если результаты уже сохранены)
python scripts/compare.py \
  --rag-result "results/rag_evaluation/rag_evaluation_20260110_123456.json" \
  --graphrag-result "results/graphrag_evaluation/graphrag_evaluation_20260110_123456.json" \
  --output-dir "results/final_comparison"
```
