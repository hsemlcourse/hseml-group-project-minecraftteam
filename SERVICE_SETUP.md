# Настройка сервиса

Теперь в проекте есть:

- backend на `FastAPI` в `src/api.py`
- интерфейс на `Streamlit` в `src/streamlit_app.py`
- Docker-файлы для обоих сервисов

## 1. Локальный запуск (без Docker)

Установите зависимости:

```bash
pip install -r requirements.txt
```

Запустите API:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Запустите Streamlit (в новом терминале):

```bash
streamlit run src/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

Откройте:

- документацию API: `http://localhost:8000/docs`
- UI: `http://localhost:8501`

## 2. Запуск в Docker

```bash
docker compose up --build
```

Откройте:

- документацию API: `http://localhost:8000/docs`
- UI: `http://localhost:8501`

## 3. Эндпоинты API

- `GET /health` - состояние сервиса и статус загрузки модели
- `GET /model/info` - признаки и метаданные модели
- `POST /predict/features` - предсказание по строкам JSON
- `POST /predict/csv` - пакетное предсказание по загруженному CSV (ответ `json` или `csv`)
- `POST /reload-model` - перезагрузка модели с диска

## 4. Ожидаемые файлы модели

По умолчанию API использует:

- `models/repair_cost_multimodal.joblib`
- `models/repair_cost_multimodal_metrics.json`

Пути можно переопределить переменными окружения:

- `REPAIR_MODEL_PATH`
- `REPAIR_METRICS_PATH`
