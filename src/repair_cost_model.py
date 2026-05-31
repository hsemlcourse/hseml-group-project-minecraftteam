from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

BASE_NUMERIC_FEATURES: List[str] = [
    "brightness",
    "contrast",
    "blur_score",
    "edge_density",
    "num_objects",
    "green_ratio",
    "color_entropy",
    "wall_floor_ratio",
    "light_uniformity",
    "num_windows",
    "defect_score",
    "furniture_count",
    "furniture_density",
    "aesthetic_score",
]

ENGINEERED_NUMERIC_FEATURES: List[str] = [
    "low_light_defect_interaction",
    "visual_complexity",
    "clutter_score",
    "log_furniture_density",
    "is_wet_room",
    "is_outdoor",
]

DEFAULT_NUMERIC_FEATURES: List[str] = BASE_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
DEFAULT_CATEGORICAL_FEATURES: List[str] = ["location", "scene_category"]
TEXT_CANDIDATE_COLUMNS: List[str] = ["description", "ad_text", "title", "caption", "comment"]
TARGET_CANDIDATE_COLUMNS: List[str] = [
    "repair_cost",
    "price",
    "cost",
    "final_price",
    "total_price",
]

DESCRIPTION_COLUMN = "__description_text__"
TARGET_COLUMN = "__repair_cost_target__"


class IQRClipper(BaseEstimator, TransformerMixin):
    def __init__(self, multiplier: float = 1.5) -> None:
        self.multiplier = multiplier

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "IQRClipper":
        values = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(values, 25, axis=0)
        q3 = np.nanpercentile(values, 75, axis=0)
        iqr = q3 - q1
        self.lower_bounds_ = q1 - self.multiplier * iqr
        self.upper_bounds_ = q3 + self.multiplier * iqr
        self.lower_bounds_ = np.where(np.isfinite(self.lower_bounds_), self.lower_bounds_, -np.inf)
        self.upper_bounds_ = np.where(np.isfinite(self.upper_bounds_), self.upper_bounds_, np.inf)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)


@dataclass
class DataBundle:
    features: pd.DataFrame
    target: pd.Series
    target_name: str
    target_is_synthetic: bool
    text_source_column: str
    numeric_features: List[str]
    categorical_features: List[str]


def _stable_noise_from_filename(filename: str, sigma: float = 9000.0) -> float:
    digest = hashlib.sha256(str(filename).encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    rng = np.random.default_rng(seed)
    return float(rng.normal(0.0, sigma))


def _as_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = series.astype(str).str.replace(r"[^\d,\.\-]", "", regex=True).str.replace(",", ".", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def _resolve_target_column(df: pd.DataFrame, explicit_target: Optional[str]) -> Optional[str]:
    if explicit_target:
        return explicit_target if explicit_target in df.columns else None

    for column in TARGET_CANDIDATE_COLUMNS:
        if column in df.columns:
            numeric = _as_numeric(df[column])
            non_na_ratio = numeric.notna().mean()
            if non_na_ratio >= 0.65:
                return column
    return None


def _resolve_text_column(df: pd.DataFrame, explicit_text: Optional[str]) -> Optional[str]:
    if explicit_text:
        return explicit_text if explicit_text in df.columns else None

    for column in TEXT_CANDIDATE_COLUMNS:
        if column in df.columns:
            values = df[column].astype(str).str.strip()
            if (values != "").mean() >= 0.1:
                return column
    return None


def _safe_numeric_column(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    numeric = _as_numeric(df[column]).fillna(default).astype(float)
    return numeric


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    local_df = df.copy()

    for feature in BASE_NUMERIC_FEATURES:
        local_df[feature] = _safe_numeric_column(local_df, feature)

    location = local_df.get("location", pd.Series("unknown", index=local_df.index)).astype(str).str.lower()
    low_light = 1.0 - local_df["light_uniformity"].clip(0.0, 1.0)

    local_df["low_light_defect_interaction"] = low_light * local_df["defect_score"].clip(0.0, 1.0)
    local_df["visual_complexity"] = local_df["edge_density"].clip(lower=0.0) * local_df["color_entropy"].clip(lower=0.0)
    local_df["clutter_score"] = local_df["num_objects"].clip(lower=0.0) + local_df["furniture_count"].clip(lower=0.0)
    local_df["log_furniture_density"] = np.log1p(local_df["furniture_density"].clip(lower=0.0))
    local_df["is_wet_room"] = location.isin(["bathroom", "kitchen"]).astype(float)
    local_df["is_outdoor"] = location.isin(["frontyard", "backyard"]).astype(float)

    return local_df


def _synthesize_description(df: pd.DataFrame) -> pd.Series:
    location = df.get("location", pd.Series("unknown", index=df.index)).astype(str)
    scene = df.get("scene_category", pd.Series("room", index=df.index)).astype(str)

    defect = _safe_numeric_column(df, "defect_score")
    light = _safe_numeric_column(df, "light_uniformity", 0.5)
    clutter = _safe_numeric_column(df, "num_objects")
    furniture = _safe_numeric_column(df, "furniture_count")

    defect_level = np.select(
        [defect < 0.04, defect < 0.08],
        ["low defects", "medium defects"],
        default="high defects",
    )
    light_level = np.select(
        [light >= 0.7, light >= 0.45],
        ["good lighting", "acceptable lighting"],
        default="poor lighting",
    )
    clutter_level = np.select(
        [clutter < 3, clutter < 7],
        ["low clutter", "moderate clutter"],
        default="high clutter",
    )
    furniture_level = np.select(
        [furniture < 2, furniture < 6],
        ["minimal furniture", "average furniture"],
        default="furniture packed",
    )

    return (
        "room_type="
        + location.str.lower()
        + "; scene="
        + scene.str.lower()
        + "; condition="
        + pd.Series(defect_level, index=df.index)
        + "; lighting="
        + pd.Series(light_level, index=df.index)
        + "; clutter="
        + pd.Series(clutter_level, index=df.index)
        + "; furnishing="
        + pd.Series(furniture_level, index=df.index)
    )


def _generate_proxy_cost(df: pd.DataFrame) -> pd.Series:
    defect = _safe_numeric_column(df, "defect_score")
    edge = _safe_numeric_column(df, "edge_density")
    light = _safe_numeric_column(df, "light_uniformity", 0.5)
    furn_density = _safe_numeric_column(df, "furniture_density")
    furn_count = _safe_numeric_column(df, "furniture_count")
    windows = _safe_numeric_column(df, "num_windows")
    aesthetic = _safe_numeric_column(df, "aesthetic_score", 0.5)
    objects = _safe_numeric_column(df, "num_objects")

    room_mult_map = {
        "bathroom": 1.22,
        "kitchen": 1.18,
        "livingroom": 1.08,
        "living_room": 1.08,
        "bedroom": 0.96,
        "frontyard": 0.86,
        "backyard": 0.82,
    }
    location_key = (
        df.get("location", pd.Series("unknown", index=df.index))
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", "", regex=True)
    )
    room_multiplier = location_key.map(room_mult_map).fillna(1.0)

    base = 55_000.0
    defect_term = defect * 195_000.0
    edge_term = edge * 62_000.0
    lighting_term = (1.0 - light.clip(0.0, 1.0)) * 45_000.0
    density_term = furn_density.clip(0.0, 12.0) / 12.0 * 36_000.0
    furniture_term = furn_count.clip(0.0, 12.0) / 12.0 * 24_000.0
    windows_term = windows.clip(0.0, 6.0) * 7_000.0
    objects_term = objects.clip(0.0, 20.0) / 20.0 * 21_000.0
    style_term = (1.0 - aesthetic.clip(0.0, 1.0)) * 22_000.0

    deterministic_cost = (
        base
        + defect_term
        + edge_term
        + lighting_term
        + density_term
        + furniture_term
        + windows_term
        + objects_term
        + style_term
    ) * room_multiplier

    file_series = df.get("filename", pd.Series(np.arange(len(df)), index=df.index))
    noise = file_series.astype(str).map(_stable_noise_from_filename)
    final_cost = (deterministic_cost + noise).clip(lower=25_000.0, upper=900_000.0)
    return final_cost.round(2)


def prepare_training_data(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    text_column: Optional[str] = None,
    numeric_features: Optional[Sequence[str]] = None,
    categorical_features: Optional[Sequence[str]] = None,
) -> DataBundle:
    numeric_features = list(numeric_features or DEFAULT_NUMERIC_FEATURES)
    categorical_features = list(categorical_features or DEFAULT_CATEGORICAL_FEATURES)
    local_df = add_engineered_features(df)

    for feature in numeric_features:
        local_df[feature] = _safe_numeric_column(local_df, feature)

    for feature in categorical_features:
        if feature not in local_df.columns:
            local_df[feature] = "unknown"
        local_df[feature] = local_df[feature].astype(str).fillna("unknown")

    resolved_text_column = _resolve_text_column(local_df, text_column)
    if resolved_text_column is None:
        local_df[DESCRIPTION_COLUMN] = _synthesize_description(local_df)
        text_source_column = "<synthetic_description>"
    else:
        text_values = local_df[resolved_text_column].fillna("").astype(str).str.strip()
        local_df[DESCRIPTION_COLUMN] = text_values.mask(text_values == "", _synthesize_description(local_df))
        text_source_column = resolved_text_column

    resolved_target_column = _resolve_target_column(local_df, target_column)
    if resolved_target_column is None:
        local_df[TARGET_COLUMN] = _generate_proxy_cost(local_df)
        target_name = "<synthetic_repair_cost>"
        target_is_synthetic = True
    else:
        target_numeric = _as_numeric(local_df[resolved_target_column])
        if target_numeric.notna().sum() < max(64, int(0.4 * len(local_df))):
            local_df[TARGET_COLUMN] = _generate_proxy_cost(local_df)
            target_name = "<synthetic_repair_cost>"
            target_is_synthetic = True
        else:
            filled = target_numeric.fillna(target_numeric.median())
            local_df[TARGET_COLUMN] = filled.astype(float)
            target_name = resolved_target_column
            target_is_synthetic = False

    feature_columns = numeric_features + categorical_features + [DESCRIPTION_COLUMN]
    features = local_df[feature_columns].copy()
    target = local_df[TARGET_COLUMN].astype(float).copy()

    return DataBundle(
        features=features,
        target=target,
        target_name=target_name,
        target_is_synthetic=target_is_synthetic,
        text_source_column=text_source_column,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )


def prepare_inference_data(
    df: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    text_column: Optional[str] = None,
) -> pd.DataFrame:
    local_df = add_engineered_features(df)

    for feature in numeric_features:
        local_df[feature] = _safe_numeric_column(local_df, feature)

    for feature in categorical_features:
        if feature not in local_df.columns:
            local_df[feature] = "unknown"
        local_df[feature] = local_df[feature].fillna("unknown").astype(str)

    resolved_text_column = _resolve_text_column(local_df, text_column)
    if resolved_text_column is None:
        local_df[DESCRIPTION_COLUMN] = _synthesize_description(local_df)
    else:
        text_values = local_df[resolved_text_column].fillna("").astype(str).str.strip()
        local_df[DESCRIPTION_COLUMN] = text_values.mask(text_values == "", _synthesize_description(local_df))

    return local_df[list(numeric_features) + list(categorical_features) + [DESCRIPTION_COLUMN]].copy()


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    safe_true = np.maximum(np.abs(y_true), 1e-6)
    return float(np.mean(np.abs((y_true - y_pred) / safe_true)) * 100.0)


class RepairCostEnsembleModel:
    def __init__(
        self,
        numeric_features: Sequence[str],
        categorical_features: Sequence[str],
        random_state: int = 42,
    ) -> None:
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features)
        self.random_state = int(random_state)

        self.tabular_model: Optional[Pipeline] = None
        self.text_model: Optional[Pipeline] = None
        self.text_weight: float = 0.0
        self.use_text_model: bool = False

        self.target_name: str = "<unknown>"
        self.target_is_synthetic: bool = False
        self.text_source_column: str = "<unknown>"

    def _build_tabular_pipeline(self) -> Pipeline:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("iqr_clipper", IQRClipper(multiplier=1.5)),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, self.numeric_features),
                ("cat", categorical_pipeline, self.categorical_features),
            ]
        )

        regressor = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.06,
            max_depth=6,
            max_iter=250,
            min_samples_leaf=24,
            l2_regularization=0.2,
            random_state=self.random_state,
        )

        return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])

    def _build_text_pipeline(self) -> Pipeline:
        return Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=4000,
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.95,
                        token_pattern=r"(?u)\b\w\w+\b",
                    ),
                ),
                (
                    "regressor",
                    SGDRegressor(
                        loss="huber",
                        alpha=0.0002,
                        penalty="elasticnet",
                        l1_ratio=0.2,
                        max_iter=5000,
                        tol=1e-3,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    def _fit_text_weight(
        self,
        y_valid: np.ndarray,
        tabular_pred: np.ndarray,
        text_pred: np.ndarray,
    ) -> float:
        best_weight = 0.0
        best_mae = mean_absolute_error(y_valid, tabular_pred)
        for weight in np.linspace(0.0, 0.6, 13):
            combined = (1.0 - weight) * tabular_pred + weight * text_pred
            mae = mean_absolute_error(y_valid, combined)
            if mae < best_mae:
                best_mae = mae
                best_weight = float(weight)
        return best_weight

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RepairCostEnsembleModel":
        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=0.15,
            random_state=self.random_state,
        )

        self.tabular_model = self._build_tabular_pipeline()
        self.tabular_model.fit(X_train, y_train)

        train_text = X_train[DESCRIPTION_COLUMN].astype(str)
        valid_text = X_valid[DESCRIPTION_COLUMN].astype(str)
        use_text = (train_text.str.len() > 0).sum() >= 64 and train_text.nunique() >= 40

        if use_text:
            self.text_model = self._build_text_pipeline()
            self.text_model.fit(train_text, y_train)
            tab_valid_pred = self.tabular_model.predict(X_valid)
            text_valid_pred = self.text_model.predict(valid_text)
            self.text_weight = self._fit_text_weight(y_valid.to_numpy(), tab_valid_pred, text_valid_pred)
            self.use_text_model = self.text_weight > 0.0
        else:
            self.text_model = None
            self.text_weight = 0.0
            self.use_text_model = False

        self.tabular_model.fit(X, y)
        if self.use_text_model and self.text_model is not None:
            self.text_model.fit(X[DESCRIPTION_COLUMN].astype(str), y)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.tabular_model is None:
            raise RuntimeError("Model is not fitted yet.")

        tabular_pred = self.tabular_model.predict(X)
        if not self.use_text_model or self.text_model is None:
            return np.asarray(tabular_pred, dtype=float)

        text_pred = self.text_model.predict(X[DESCRIPTION_COLUMN].astype(str))
        combined = (1.0 - self.text_weight) * tabular_pred + self.text_weight * text_pred
        return np.asarray(combined, dtype=float)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        pred = self.predict(X)
        y_true = y.to_numpy(dtype=float)
        metrics = {
            "mae": float(mean_absolute_error(y_true, pred)),
            "rmse": _rmse(y_true, pred),
            "r2": float(r2_score(y_true, pred)),
            "mape_percent": _mape(y_true, pred),
        }
        return metrics

    def save(self, model_path: Path) -> None:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, model_path)

    @staticmethod
    def load(model_path: Path) -> "RepairCostEnsembleModel":
        return joblib.load(model_path)


def train_model_from_csv(
    csv_path: Path,
    model_path: Path,
    metrics_path: Optional[Path] = None,
    target_column: Optional[str] = None,
    text_column: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, float]:
    raw_df = pd.read_csv(csv_path)
    bundle = prepare_training_data(
        raw_df,
        target_column=target_column,
        text_column=text_column,
        numeric_features=DEFAULT_NUMERIC_FEATURES,
        categorical_features=DEFAULT_CATEGORICAL_FEATURES,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        bundle.features,
        bundle.target,
        test_size=test_size,
        random_state=random_state,
    )

    model = RepairCostEnsembleModel(
        numeric_features=bundle.numeric_features,
        categorical_features=bundle.categorical_features,
        random_state=random_state,
    )
    model.target_name = bundle.target_name
    model.target_is_synthetic = bundle.target_is_synthetic
    model.text_source_column = bundle.text_source_column
    model.fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)

    model.save(model_path)
    summary = {
        **metrics,
        "rows": float(len(raw_df)),
        "target_is_synthetic": float(model.target_is_synthetic),
        "text_weight": float(model.text_weight),
    }

    if metrics_path is not None:
        metrics_path = Path(metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": str(Path(csv_path).resolve()),
            "model_path": str(Path(model_path).resolve()),
            "target_name": model.target_name,
            "target_is_synthetic": model.target_is_synthetic,
            "text_source_column": model.text_source_column,
            "text_weight": model.text_weight,
            "metrics": metrics,
            "n_rows": len(raw_df),
            "test_size": test_size,
            "random_state": random_state,
            "numeric_features": bundle.numeric_features,
            "categorical_features": bundle.categorical_features,
        }
        metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary


def predict_costs_from_csv(
    input_csv: Path,
    model_path: Path,
    output_csv: Optional[Path] = None,
    text_column: Optional[str] = None,
) -> pd.DataFrame:
    model = RepairCostEnsembleModel.load(model_path)
    raw_df = pd.read_csv(input_csv)
    inference_df = prepare_inference_data(
        raw_df,
        numeric_features=model.numeric_features,
        categorical_features=model.categorical_features,
        text_column=text_column,
    )
    pred = model.predict(inference_df)

    result = raw_df.copy()
    result["predicted_repair_cost"] = np.round(pred, 2)
    result["predicted_repair_cost_label"] = result["predicted_repair_cost"].map(
        lambda value: f"{int(round(value)):,} RUB".replace(",", " ")
    )
    result["predicted_repair"] = np.select(
        [
            result["predicted_repair_cost"] < 90_000,
            result["predicted_repair_cost"] < 170_000,
        ],
        ["Без ремонта", "Косметический"],
        default="Капитальный",
    )

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_csv, index=False)

    return result
