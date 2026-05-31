# Предсказание стоимости ремонта по фото комнаты

Студенты: Рогачев Глеб Игоревич, Соснин Артем Олегович  
Группа: БИВ234

Проект оценивает стоимость ремонта по признакам, извлеченным из фотографий комнат. Основной пайплайн работает с
`data/room_dataset.csv`, обучает регрессионные модели и сохраняет предсказания в `data/repair_verdicts.csv`.

## Данные

Источник данных - локальный набор фотографий комнат в `data/photo/`, разложенный по типам помещений:
`bathroom`, `kitchen`, `bedroom`, `livingRoom`, `frontyard`, `backyard`. Его выбрали, потому что он содержит
разные типы комнат, освещения, мебели и визуальной сложности, то есть признаки, связанные с объемом ремонта.

Самостоятельный парсинг реализован в `src/make_csv_metrics_from_protos.py`: скрипт проходит по изображениям,
извлекает базовые CV-признаки, категорию сцены через Places365 и количество объектов через Faster R-CNN.

Итоговый датасет:

- `data/room_dataset.csv`: 5859 строк, 17 исходных столбцов.
- `data/processed/room_dataset_clean.csv`: 5859 строк, 24 столбца после feature engineering.
- Пропуски: 0 до и после очистки.
- Дубли: 0.
- Выбросы: 4431 значения обнаружены IQR-правилом; clipping применяется внутри sklearn pipeline только на train.
- Целевой столбец: `<synthetic_repair_cost>`, детерминированный proxy target, потому что в image dataset нет явной цены.

Основные признаки: яркость, контраст, blur score, edge density, число объектов, доля зеленого цвета, entropy,
равномерность света, окна, defect score, количество и плотность мебели, aesthetic score, тип комнаты и scene category.

Добавленные признаки:

- `low_light_defect_interaction` - взаимодействие плохого освещения и дефектов.
- `visual_complexity` - edge density * color entropy.
- `clutter_score` - сумма объектов и мебели.
- `log_furniture_density` - логарифм плотности мебели.
- `is_wet_room`, `is_outdoor` - бинарные признаки типа помещения.

## Метрики

Главная метрика - MAE в рублях: она прямо показывает среднюю ошибку сметы и менее чувствительна к редким дорогим
ремонтам. RMSE используется как дополнительная метрика для крупных промахов, R2 - для объясненной дисперсии,
MAPE - для относительной ошибки. При выборе финальной модели приоритет отдается validation MAE.

## Эксперименты

Сплит фиксированный и воспроизводимый: 4101 train, 879 validation, 879 test, `random_state=42`, стратификация по
`location`. Data leakage избегается так: дубли удаляются до сплита, целевой столбец и предсказания не используются
как признаки, а imputer/scaler/IQR clipping/encoder обучаются внутри pipeline только на train.

| Модель | Feature set | Dim reduction | Параметры | Val MAE | Test MAE | Test RMSE | Test R2 | Test MAPE |
|---|---|---|---|---:|---:|---:|---:|---:|
| HistGradientBoosting | engineered | none | learning_rate=0.05, max_iter=180, max_leaf_nodes=31 | 7972.65 | 7702.73 | 9614.22 | 0.9455 | 6.21% |
| Voting ensemble | engineered | none | RF + ExtraTrees + HGB | 7995.42 | 7869.50 | 9964.28 | 0.9415 | 6.35% |
| ExtraTrees | engineered | none | n_estimators=80, min_samples_leaf=4 | 8105.45 | 8016.45 | 10205.55 | 0.9386 | 6.45% |
| Ridge | engineered | none | alpha=30 | 8205.31 | 7805.44 | 9831.82 | 0.9430 | 6.23% |
| RandomForest | engineered | none | n_estimators=80, max_depth=16 | 8728.84 | 8644.62 | 11152.02 | 0.9267 | 6.93% |
| LinearRegression baseline | no FE | none | out of the box | 9114.52 | 8756.30 | 10954.91 | 0.9293 | 6.94% |
| KNN | engineered | none | n_neighbors=15, weights=distance | 9529.73 | 9265.65 | 11803.28 | 0.9179 | 7.30% |
| Ridge + PCA | engineered | PCA(10) | alpha=10 | 9549.86 | 9530.71 | 12027.82 | 0.9147 | 7.53% |
| Dummy baseline | no FE | none | mean target | 33724.12 | 34009.10 | 41206.82 | -0.0007 | 28.08% |

Финальная модель по validation MAE - `HistGradientBoosting_engineered`. PCA-вариант оказался хуже, поэтому
уменьшение размерности оставлено как диагностический эксперимент, а не как финальный пайплайн.

Основной CLI-пайплайн `src/repair_cost_cli.py train` после обновления признаков дает:

- MAE: 8067.02
- RMSE: 9971.34
- R2: 0.9432
- MAPE: 6.79%

## Визуализации и артефакты

- `report/images/target_distribution.png` - распределение target.
- `report/images/feature_correlation_heatmap.png` - корреляции числовых признаков.
- `report/images/pca_projection.png` - 2D PCA-проекция признаков.
- `report/images/model_comparison.png` - сравнение моделей по validation MAE.
- `report/images/feature_importance.png` - permutation importance финальной модели.
- `models/experiment_results.csv` - полная таблица экспериментов.
- `models/feature_importance.csv` - важность признаков.
- `models/best_experiment_metrics.json` - split, лучшая модель и итоговые метрики.

Топ признаков по permutation importance: `furniture_density`, `clutter_score`, `is_wet_room`, `num_windows`,
`low_light_defect_interaction`, `location`.

## Структура

```text
.
├── data/
│   ├── photo/                         # исходные изображения
│   ├── processed/                     # очищенный датасет и data quality summary
│   ├── room_dataset.csv               # CSV с image-derived признаками
│   └── repair_verdicts.csv            # предсказания CLI
├── docs/                              # служебные материалы и исходный фидбек
├── models/                            # метрики, таблицы экспериментов, feature importance
├── report/
│   ├── images/                        # графики для отчета
│   └── report.md                      # финальный отчет
├── src/
│   ├── make_csv_metrics_from_protos.py
│   ├── repair_cost_cli.py
│   ├── repair_cost_experiments.py
│   └── repair_cost_model.py
├── tests/
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── requirements.txt
```

## Запуск

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Команды:

```bash
python src/repair_cost_experiments.py
python src/repair_cost_cli.py train
python src/repair_cost_cli.py predict
python -m ruff check src tests
python -m flake8 src tests
python -m pytest -q
```

Через Makefile:

```bash
make lint
make precommit
make test
make experiments
make train
make predict
```

Для автоматической проверки перед коммитом:

```bash
pre-commit install
pre-commit run --all-files
```

Через Docker:

```bash
docker compose up --build
```
