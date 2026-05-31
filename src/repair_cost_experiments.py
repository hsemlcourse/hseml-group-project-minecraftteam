from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from repair_cost_model import (
    BASE_NUMERIC_FEATURES,
    DEFAULT_CATEGORICAL_FEATURES,
    DEFAULT_NUMERIC_FEATURES,
    IQRClipper,
    prepare_training_data,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RANDOM_STATE = 42
TARGET_OUTPUT_COLUMN = "repair_cost_target"


@dataclass
class DataQualitySummary:
    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int
    duplicates_removed: int
    missing_values_before: int
    missing_values_after: int
    outlier_values_detected: int
    target_name: str
    target_is_synthetic: bool
    text_source_column: str
    numeric_features: List[str]
    categorical_features: List[str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _mape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    safe_true = np.maximum(np.abs(y_true_array), 1e-6)
    return float(np.mean(np.abs((y_true_array - y_pred_array) / safe_true)) * 100.0)


def _metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true_array, y_pred_array)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_array, y_pred_array))),
        "r2": float(r2_score(y_true_array, y_pred_array)),
        "mape_percent": _mape(y_true_array, y_pred_array),
    }


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def clean_source_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[column] = cleaned[column].fillna("unknown").astype(str).str.strip()
        cleaned[column] = cleaned[column].mask(cleaned[column] == "", "unknown")

    return cleaned


def count_iqr_outliers(df: pd.DataFrame, numeric_features: Sequence[str]) -> int:
    outliers = 0
    for feature in numeric_features:
        if feature not in df.columns:
            continue
        values = pd.to_numeric(df[feature], errors="coerce")
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers += int(((values < lower) | (values > upper)).sum())
    return outliers


def build_modeling_frame(
    raw_df: pd.DataFrame,
    target_column: Optional[str] = None,
    text_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series, DataQualitySummary]:
    missing_before = int(raw_df.isna().sum().sum())
    duplicates_before = int(raw_df.duplicated().sum())
    outliers_before = count_iqr_outliers(raw_df, BASE_NUMERIC_FEATURES)

    cleaned_source = clean_source_dataframe(raw_df)
    bundle = prepare_training_data(cleaned_source, target_column=target_column, text_column=text_column)

    modeling_frame = bundle.features.copy()
    modeling_frame[TARGET_OUTPUT_COLUMN] = bundle.target.to_numpy(dtype=float)

    summary = DataQualitySummary(
        rows_before=len(raw_df),
        rows_after=len(modeling_frame),
        columns_before=len(raw_df.columns),
        columns_after=len(modeling_frame.columns),
        duplicates_removed=duplicates_before,
        missing_values_before=missing_before,
        missing_values_after=int(modeling_frame.isna().sum().sum()),
        outlier_values_detected=outliers_before,
        target_name=bundle.target_name,
        target_is_synthetic=bundle.target_is_synthetic,
        text_source_column=bundle.text_source_column,
        numeric_features=bundle.numeric_features,
        categorical_features=bundle.categorical_features,
    )
    return modeling_frame, bundle.target, summary


def train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    stratify = X["location"] if "location" in X.columns else None
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=random_state,
        stratify=stratify,
    )

    valid_stratify = X_train_valid["location"] if "location" in X_train_valid.columns else None
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid,
        y_train_valid,
        test_size=0.1765,
        random_state=random_state,
        stratify=valid_stratify,
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def build_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    numeric_steps: List[Tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
        ("iqr_clipper", IQRClipper(multiplier=1.5)),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), list(numeric_features)),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", _one_hot_encoder()),
                    ]
                ),
                list(categorical_features),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(
    estimator: Any,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    scale_numeric: bool,
    pca_components: Optional[int] = None,
) -> Pipeline:
    steps: List[Tuple[str, Any]] = [
        ("preprocessor", build_preprocessor(numeric_features, categorical_features, scale_numeric))
    ]
    if pca_components is not None:
        steps.append(("pca", PCA(n_components=pca_components, random_state=RANDOM_STATE)))
    steps.append(("model", estimator))
    return Pipeline(steps)


def _experiment_specs() -> List[Dict[str, Any]]:
    voting_regressor = VotingRegressor(
        estimators=[
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=80,
                    max_depth=14,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
            (
                "extra",
                ExtraTreesRegressor(
                    n_estimators=80,
                    max_depth=None,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
            (
                "hgb",
                HistGradientBoostingRegressor(
                    learning_rate=0.06,
                    max_iter=180,
                    max_leaf_nodes=31,
                    random_state=RANDOM_STATE,
                ),
            ),
        ],
        n_jobs=-1,
    )

    return [
        {
            "name": "DummyRegressor_mean",
            "estimator": DummyRegressor(strategy="mean"),
            "param_grid": [{}],
            "feature_set": "baseline_no_feature_engineering",
            "numeric_features": BASE_NUMERIC_FEATURES,
            "scale_numeric": False,
            "pca_components": None,
        },
        {
            "name": "LinearRegression_baseline",
            "estimator": LinearRegression(),
            "param_grid": [{}],
            "feature_set": "baseline_no_feature_engineering",
            "numeric_features": BASE_NUMERIC_FEATURES,
            "scale_numeric": True,
            "pca_components": None,
        },
        {
            "name": "Ridge_engineered",
            "estimator": Ridge(random_state=RANDOM_STATE),
            "param_grid": [{"alpha": [1.0, 10.0, 30.0]}],
            "feature_set": "engineered_features",
            "numeric_features": DEFAULT_NUMERIC_FEATURES,
            "scale_numeric": True,
            "pca_components": None,
        },
        {
            "name": "KNN_engineered",
            "estimator": KNeighborsRegressor(),
            "param_grid": [{"n_neighbors": [7, 15], "weights": ["distance"]}],
            "feature_set": "engineered_features",
            "numeric_features": DEFAULT_NUMERIC_FEATURES,
            "scale_numeric": True,
            "pca_components": None,
        },
        {
            "name": "RandomForest_engineered",
            "estimator": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            "param_grid": [
                {"n_estimators": [80], "max_depth": [10, 16], "min_samples_leaf": [2]},
            ],
            "feature_set": "engineered_features",
            "numeric_features": DEFAULT_NUMERIC_FEATURES,
            "scale_numeric": False,
            "pca_components": None,
        },
        {
            "name": "ExtraTrees_engineered",
            "estimator": ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            "param_grid": [{"n_estimators": [80], "max_depth": [None], "min_samples_leaf": [2, 4]}],
            "feature_set": "engineered_features",
            "numeric_features": DEFAULT_NUMERIC_FEATURES,
            "scale_numeric": False,
            "pca_components": None,
        },
        {
            "name": "HistGradientBoosting_engineered",
            "estimator": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
            "param_grid": [
                {"learning_rate": [0.05, 0.08], "max_iter": [180], "max_leaf_nodes": [31]},
            ],
            "feature_set": "engineered_features",
            "numeric_features": DEFAULT_NUMERIC_FEATURES,
            "scale_numeric": False,
            "pca_components": None,
        },
        {
            "name": "Ridge_PCA_engineered",
            "estimator": Ridge(alpha=10.0, random_state=RANDOM_STATE),
            "param_grid": [{}],
            "feature_set": "engineered_features",
            "numeric_features": DEFAULT_NUMERIC_FEATURES,
            "scale_numeric": True,
            "pca_components": 10,
        },
        {
            "name": "VotingEnsemble_engineered",
            "estimator": voting_regressor,
            "param_grid": [{}],
            "feature_set": "engineered_features",
            "numeric_features": DEFAULT_NUMERIC_FEATURES,
            "scale_numeric": False,
            "pca_components": None,
        },
    ]


def run_model_experiments(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Pipeline]:
    rows: List[Dict[str, Any]] = []
    best_pipeline: Optional[Pipeline] = None
    best_val_mae = np.inf

    for spec in _experiment_specs():
        numeric_features = [feature for feature in spec["numeric_features"] if feature in X_train.columns]
        categorical_features = [feature for feature in DEFAULT_CATEGORICAL_FEATURES if feature in X_train.columns]

        for params in ParameterGrid(spec["param_grid"]):
            estimator = clone(spec["estimator"])
            estimator.set_params(**params)
            pipeline = build_pipeline(
                estimator=estimator,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                scale_numeric=spec["scale_numeric"],
                pca_components=spec["pca_components"],
            )
            pipeline.fit(X_train, y_train)

            valid_pred = pipeline.predict(X_valid)
            test_pred = pipeline.predict(X_test)
            valid_metrics = _metrics(y_valid, valid_pred)
            test_metrics = _metrics(y_test, test_pred)

            row = {
                "model": spec["name"],
                "feature_set": spec["feature_set"],
                "dimensionality_reduction": f"PCA({spec['pca_components']})"
                if spec["pca_components"] is not None
                else "none",
                "params": _json_dumps(params),
                **{f"val_{key}": value for key, value in valid_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
            rows.append(row)

            if valid_metrics["mae"] < best_val_mae:
                best_val_mae = valid_metrics["mae"]
                best_pipeline = pipeline

    if best_pipeline is None:
        raise RuntimeError("No experiments were executed.")

    best_pipeline.fit(pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]))
    return pd.DataFrame(rows).sort_values("val_mae").reset_index(drop=True), best_pipeline


def save_feature_importance(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_csv: Path,
    output_png: Path,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "mae_increase_mean": result.importances_mean,
            "mae_increase_std": result.importances_std,
        }
    ).sort_values("mae_increase_mean", ascending=False)
    importance.to_csv(output_csv, index=False)

    top = importance.head(12).iloc[::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(top["feature"], top["mae_increase_mean"], color="#31688e")
    plt.xlabel("MAE increase after permutation")
    plt.title("Top feature importance")
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=160)
    plt.close()
    return importance


def save_visualizations(
    modeling_frame: pd.DataFrame,
    X_train: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "target_distribution": output_dir / "target_distribution.png",
        "correlation_heatmap": output_dir / "feature_correlation_heatmap.png",
        "pca_projection": output_dir / "pca_projection.png",
    }

    plt.figure(figsize=(8, 4.5))
    plt.hist(modeling_frame[TARGET_OUTPUT_COLUMN], bins=36, color="#3b528b", edgecolor="white")
    plt.xlabel("Repair cost target, RUB")
    plt.ylabel("Rows")
    plt.title("Target distribution")
    plt.tight_layout()
    plt.savefig(paths["target_distribution"], dpi=160)
    plt.close()

    corr_columns = [feature for feature in DEFAULT_NUMERIC_FEATURES if feature in modeling_frame.columns]
    corr = modeling_frame[corr_columns + [TARGET_OUTPUT_COLUMN]].corr(numeric_only=True)
    plt.figure(figsize=(9, 7))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=7)
    plt.title("Numeric feature correlations")
    plt.tight_layout()
    plt.savefig(paths["correlation_heatmap"], dpi=170)
    plt.close()

    numeric_features = [feature for feature in DEFAULT_NUMERIC_FEATURES if feature in X_train.columns]
    categorical_features = [feature for feature in DEFAULT_CATEGORICAL_FEATURES if feature in X_train.columns]
    pca_pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor(numeric_features, categorical_features, scale_numeric=True)),
            ("pca", PCA(n_components=2, random_state=RANDOM_STATE)),
        ]
    )
    features = modeling_frame.drop(columns=[TARGET_OUTPUT_COLUMN])
    projection = pca_pipeline.fit_transform(features)
    sample_size = min(2500, len(projection))
    sample_idx = np.random.default_rng(RANDOM_STATE).choice(len(projection), size=sample_size, replace=False)

    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(
        projection[sample_idx, 0],
        projection[sample_idx, 1],
        c=modeling_frame[TARGET_OUTPUT_COLUMN].to_numpy()[sample_idx],
        cmap="viridis",
        s=10,
        alpha=0.7,
    )
    plt.colorbar(scatter, label="Repair cost target, RUB")
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.title("PCA projection of engineered features")
    plt.tight_layout()
    plt.savefig(paths["pca_projection"], dpi=160)
    plt.close()

    return {name: str(path) for name, path in paths.items()}


def save_model_comparison(results: pd.DataFrame, output_png: Path) -> None:
    best_by_model = results.sort_values("val_mae").drop_duplicates("model").sort_values("val_mae")
    plt.figure(figsize=(9, 4.8))
    plt.barh(best_by_model["model"].iloc[::-1], best_by_model["val_mae"].iloc[::-1], color="#35b779")
    plt.xlabel("Validation MAE")
    plt.title("Best validation MAE by model")
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=160)
    plt.close()


def run_experiments(
    dataset_path: Path,
    processed_dir: Path,
    models_dir: Path,
    report_images_dir: Path,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    raw_df = pd.read_csv(dataset_path)
    modeling_frame, target, quality_summary = build_modeling_frame(raw_df)
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    clean_csv = processed_dir / "room_dataset_clean.csv"
    quality_json = processed_dir / "data_quality_summary.json"
    modeling_frame.to_csv(clean_csv, index=False)
    quality_json.write_text(json.dumps(asdict(quality_summary), ensure_ascii=False, indent=2), encoding="utf-8")

    X = modeling_frame.drop(columns=[TARGET_OUTPUT_COLUMN])
    y = target.reset_index(drop=True)
    X_train, X_valid, X_test, y_train, y_valid, y_test = train_val_test_split(X, y, random_state=random_state)

    results, best_model = run_model_experiments(X_train, X_valid, X_test, y_train, y_valid, y_test)
    experiment_csv = models_dir / "experiment_results.csv"
    results.to_csv(experiment_csv, index=False)

    best_row = results.iloc[0].to_dict()
    test_metrics = {key.removeprefix("test_"): best_row[key] for key in best_row if key.startswith("test_")}
    best_model_path = models_dir / "best_experiment_model.joblib"
    best_metrics_path = models_dir / "best_experiment_metrics.json"
    joblib.dump(best_model, best_model_path)

    feature_importance_csv = models_dir / "feature_importance.csv"
    feature_importance_png = report_images_dir / "feature_importance.png"
    importance = save_feature_importance(best_model, X_test, y_test, feature_importance_csv, feature_importance_png)

    image_paths = save_visualizations(modeling_frame, X_train, report_images_dir)
    comparison_png = report_images_dir / "model_comparison.png"
    save_model_comparison(results, comparison_png)
    image_paths["feature_importance"] = str(feature_importance_png)
    image_paths["model_comparison"] = str(comparison_png)

    pca_row = results[results["dimensionality_reduction"] != "none"].head(1)
    pca_csv = models_dir / "pca_experiment.csv"
    pca_row.to_csv(pca_csv, index=False)

    payload = {
        "dataset": str(dataset_path),
        "clean_dataset": str(clean_csv),
        "data_quality_summary": asdict(quality_summary),
        "split": {
            "train_rows": len(X_train),
            "validation_rows": len(X_valid),
            "test_rows": len(X_test),
            "random_state": random_state,
            "stratified_by": "location",
        },
        "best_experiment": best_row,
        "best_test_metrics": test_metrics,
        "top_features": importance.head(10).to_dict(orient="records"),
        "artifacts": {
            "experiment_results": str(experiment_csv),
            "pca_experiment": str(pca_csv),
            "best_model": str(best_model_path),
            "feature_importance": str(feature_importance_csv),
            "images": image_paths,
        },
    }
    best_metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Run repair cost data checks and model experiments.")
    parser.add_argument("--dataset", type=Path, default=root / "data" / "room_dataset.csv")
    parser.add_argument("--processed-dir", type=Path, default=root / "data" / "processed")
    parser.add_argument("--models-dir", type=Path, default=root / "models")
    parser.add_argument("--report-images-dir", type=Path, default=root / "report" / "images")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    payload = run_experiments(
        dataset_path=args.dataset,
        processed_dir=args.processed_dir,
        models_dir=args.models_dir,
        report_images_dir=args.report_images_dir,
        random_state=args.random_state,
    )
    best = payload["best_experiment"]
    print("Experiments finished.")
    print(f"Best model: {best['model']}")
    print(f"Validation MAE: {best['val_mae']:.2f}")
    print(f"Test MAE: {best['test_mae']:.2f}")
    print(f"Results: {payload['artifacts']['experiment_results']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
