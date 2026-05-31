# Отчет по проекту

## 1. Постановка задачи

Цель проекта - оценить стоимость ремонта по фотографии помещения. Задача формулируется как регрессия: по
визуальным и категориальным признакам комнаты предсказать стоимость ремонта в рублях

Практический смысл: пользователь загружает фото комнаты, система извлекает признаки состояния помещения и дает
приблизительную смету: без ремонта, косметический ремонт или капитальный ремонт

## 2. Данные и источник

Данные находятся в `data/photo/` и `data/room_dataset.csv`. Фото разложены по типам помещений:
`bathroom`, `kitchen`, `bedroom`, `livingRoom`, `frontyard`, `backyard`. Такой набор выбран из-за разнообразия
сцен: внутренние комнаты, влажные зоны и outdoor-зоны имеют разную стоимость ремонта

Самостоятельный парсинг выполнен в `src/make_csv_metrics_from_protos.py`. Скрипт:

- проходит по подпапкам с изображениями
- считает brightness, contrast, blur score, edge density, green ratio, color entropy
- определяет scene category через Places365
- считает объекты и мебель через Faster R-CNN
- добавляет эвристические признаки окон, дефектов, равномерности света и плотности мебели

`data/room_dataset.csv` содержит 5859 строк и 17 исходных столбцов После feature engineering в
`data/processed/room_dataset_clean.csv` используется 24 столбца

## 3. Очистка и подготовка данных

Проверки качества данных сохранены в `data/processed/data_quality_summary.json`

| Проверка | Результат |
|---|---:|
| Строк до очистки | 5859 |
| Строк после очистки | 5859 |
| Исходных столбцов | 17 |
| Столбцов после feature engineering | 24 |
| Пропуски до очистки | 0 |
| Пропуски после очистки | 0 |
| Дубли | 0 |
| IQR-выбросы | 4431 значений |

Стратегия очистки:

- типы числовых признаков приводятся через `pd.to_numeric`
- категориальные значения нормализуются как строки, пустые значения заменяются на `unknown`
- бесконечности заменяются на `NaN`
- дубли удаляются до сплита
- пропуски и выбросы обрабатываются внутри sklearn pipeline: median imputer и `IQRClipper`
- `IQRClipper` обучается только на train, поэтому статистики validation/test не попадают в обучение

## 4. Feature engineering

Исходно было 14 числовых признаков, 2 категориальных и идентификатор файла. Добавлено 6 новых признаков:

- `low_light_defect_interaction` - сильнее штрафует комнаты с дефектами при плохом освещении
- `visual_complexity` - отражает сложность текстур и границ
- `clutter_score` - суммарная заставленность объектами и мебелью
- `log_furniture_density` - сглаженная плотность мебели
- `is_wet_room` - кухня/ванная, где ремонт обычно дороже
- `is_outdoor` - двор/фасад, где другая структура работ

Permutation importance финальной модели подтверждает значимость новых признаков:

| Признак | Увеличение MAE при перестановке |
|---|---:|
| furniture_density | 9990.38 |
| clutter_score | 9643.34 |
| is_wet_room | 5742.88 |
| num_windows | 4144.36 |
| low_light_defect_interaction | 3236.83 |
| location | 2284.65 |

## 5. Метрики и split

Основная метрика - MAE, потому что она измеряется в рублях и легко интерпретируется как средняя ошибка сметы
RMSE добавлена для контроля крупных промахов, R2 - для общей объясненной дисперсии, MAPE - для относительной ошибки
При выборе модели приоритет отдан validation MAE

Сплит:

- train: 4101 строк
- validation: 879 строк
- test: 879 строк
- `random_state=42`
- стратификация по `location`

Data leakage предотвращался так: дубли удалены до split, target/predicted columns не попадают в признаки,
все обучаемые preprocessing-операции находятся внутри pipeline и fit выполняется только на train

## 6. Baseline

Baseline без feature engineering:

| Модель | Val MAE | Test MAE | Test RMSE | Test R2 | Test MAPE |
|---|---:|---:|---:|---:|---:|
| Dummy mean | 33724.12 | 34009.10 | 41206.82 | -0.0007 | 28.08% |
| LinearRegression | 9114.52 | 8756.30 | 10954.91 | 0.9293 | 6.94% |

LinearRegression baseline уже сильно лучше среднего target, поэтому дальнейшие эксперименты сравнивались с ним

## 7. Эксперименты

Полная таблица сохранена в `models/experiment_results.csv`

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

Проверено минимум 5 семейств моделей: linear baseline, Ridge, KNN, RandomForest, ExtraTrees,
HistGradientBoosting и Voting ensemble. Для Ridge, KNN, RandomForest, ExtraTrees и HistGradientBoosting выполнен
перебор гиперпараметров

## 8. Уменьшение размерности

Из-за one-hot кодирования `location` и `scene_category` пространство признаков расширяется, поэтому добавлен
эксперимент `Ridge + PCA(10)`. Результат: test MAE 9530.71, test R2 0.9147. Это хуже финальной модели, но PCA
полезна для визуальной диагностики кластеров комнат

График: `report/images/pca_projection.png`

## 9. Визуализации

- `report/images/target_distribution.png` - распределение target
- `report/images/feature_correlation_heatmap.png` - корреляции числовых признаков
- `report/images/model_comparison.png` - сравнение validation MAE
- `report/images/feature_importance.png` - важность признаков
- `report/images/pca_projection.png` - PCA-проекция

## 10. Финальная модель

Финальная модель - `HistGradientBoostingRegressor` с feature engineering:

- `learning_rate=0.05`
- `max_iter=180`
- `max_leaf_nodes=31`
- validation MAE: 7972.65
- test MAE: 7702.73
- test RMSE: 9614.22
- test R2: 0.9455
- test MAPE: 6.21%

Модель выбрана потому, что дает лучший validation MAE, хорошо работает с нелинейными зависимостями и не требует
PCA. Ансамбль Voting близок по качеству, но немного хуже и сложнее в поддержке

## 11. Воспроизводимость

Сделано:

- фиксированный seed `42` во всех экспериментах
- зависимости закреплены в `requirements.txt`
- линтеры настроены через `ruff`, `flake8`, `pyproject.toml` и `.flake8`
- добавлен `pre-commit` и конфигурация `.pre-commit-config.yaml`
- добавлен `Makefile` с командами `precommit`, `lint`, `test`, `experiments`, `train`, `predict`
- добавлены `Dockerfile` и `docker-compose.yml`
- CI запускает линтеры и pytest
- структура проекта описана в README
- служебные файлы из корня перенесены в `docs/`

Команды проверки:

```bash
python -m ruff check src tests
python -m flake8 src tests
python -m pytest -q
python src/repair_cost_experiments.py
python src/repair_cost_cli.py train
python src/repair_cost_cli.py predict
```

## 12. Выводы

Feature engineering улучшил качество относительно LinearRegression baseline: test MAE снизился с 8756.30 до
7702.73. Самыми важными оказались признаки плотности мебели, заставленности, влажной зоны, количества окон и
взаимодействия света с дефектами. PCA ухудшила результат, поэтому финальная модель использует исходное
engineered-пространство. Главный риск проекта - отсутствие реальной цены в image dataset; поэтому target является
proxy estimate. Для следующей версии лучше собрать реальные объявления с фотографиями и фактическими сметами
