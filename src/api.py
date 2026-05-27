from __future__ import annotations

import io
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

try:
    from .repair_cost_model import RepairCostEnsembleModel, prepare_inference_data
except ImportError:  # pragma: no cover - fallback for direct script-style imports
    from repair_cost_model import RepairCostEnsembleModel, prepare_inference_data


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "repair_cost_multimodal.joblib"
DEFAULT_METRICS_PATH = REPO_ROOT / "models" / "repair_cost_multimodal_metrics.json"


class PredictFeaturesRequest(BaseModel):
    rows: List[Dict[str, Any]] = Field(..., min_length=1, description="List of rows with model features.")
    text_column: Optional[str] = Field(
        default=None,
        description="Optional column name with free-text description; if omitted, synthetic text is used.",
    )


class PredictionItem(BaseModel):
    row_index: int
    predicted_repair_cost: float
    predicted_repair_cost_label: str
    predicted_repair: str


class PredictFeaturesResponse(BaseModel):
    count: int
    predictions: List[PredictionItem]


class ModelInfoResponse(BaseModel):
    model_path: str
    metrics_path: str
    numeric_features: List[str]
    categorical_features: List[str]
    target_name: str
    target_is_synthetic: bool
    text_source_column: str
    text_weight: float
    metrics: Optional[Dict[str, float]] = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_service_state()
    yield


app = FastAPI(
    title="Repair Cost API",
    description="Multimodal service for estimating apartment repair cost from structured features and optional text.",
    version="1.0.0",
    lifespan=lifespan,
)


def _resolve_path(env_name: str, default_path: Path) -> Path:
    raw_value = os.getenv(env_name)
    if raw_value is None or raw_value.strip() == "":
        return default_path
    return Path(raw_value).expanduser().resolve()


def _load_metrics(path: Path) -> Optional[Dict[str, float]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        return None

    parsed_metrics: Dict[str, float] = {}
    for key, value in metrics.items():
        try:
            parsed_metrics[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return parsed_metrics or None


def _repair_label(cost: float) -> str:
    if cost < 90_000:
        return "Без ремонта"
    if cost < 170_000:
        return "Косметический"
    return "Капитальный"


def _repair_cost_human_label(cost: float) -> str:
    return f"{int(round(cost)):,} RUB".replace(",", " ")


def _build_prediction_table(source_df: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    result = source_df.copy()
    result["predicted_repair_cost"] = np.round(prediction.astype(float), 2)
    result["predicted_repair_cost_label"] = result["predicted_repair_cost"].map(_repair_cost_human_label)
    result["predicted_repair"] = result["predicted_repair_cost"].map(_repair_label)
    return result


def _load_service_state() -> None:
    model_path = _resolve_path("REPAIR_MODEL_PATH", DEFAULT_MODEL_PATH)
    metrics_path = _resolve_path("REPAIR_METRICS_PATH", DEFAULT_METRICS_PATH)

    app.state.model_path = model_path
    app.state.metrics_path = metrics_path
    app.state.metrics = _load_metrics(metrics_path)
    app.state.model = None
    app.state.model_error = None

    try:
        src_dir = Path(__file__).resolve().parent
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        __import__("repair_cost_model")
        app.state.model = RepairCostEnsembleModel.load(model_path)
    except Exception as exc:  # pragma: no cover - runtime safeguard
        app.state.model_error = f"{type(exc).__name__}: {exc}"


def _require_model() -> RepairCostEnsembleModel:
    model = getattr(app.state, "model", None)
    if model is not None:
        return model

    model_error = getattr(app.state, "model_error", None)
    model_path = getattr(app.state, "model_path", DEFAULT_MODEL_PATH)
    detail = (
        f"Model is not loaded. model_path={model_path}. "
        f"error={model_error if model_error else 'unknown'}"
    )
    raise HTTPException(status_code=503, detail=detail)


def _predict_dataframe(raw_df: pd.DataFrame, text_column: Optional[str]) -> pd.DataFrame:
    if raw_df.empty:
        raise HTTPException(status_code=400, detail="Input dataset is empty.")

    model = _require_model()
    prepared = prepare_inference_data(
        raw_df,
        numeric_features=model.numeric_features,
        categorical_features=model.categorical_features,
        text_column=text_column,
    )
    prediction = model.predict(prepared)
    return _build_prediction_table(raw_df, prediction)


@app.get("/")
def root() -> Dict[str, str]:
    return {
        "service": "repair-cost-api",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    model_error = getattr(app.state, "model_error", None)
    model_loaded = getattr(app.state, "model", None) is not None
    status = "ok" if model_loaded else "degraded"
    return {
        "status": status,
        "model_loaded": model_loaded,
        "model_path": str(getattr(app.state, "model_path", DEFAULT_MODEL_PATH)),
        "metrics_path": str(getattr(app.state, "metrics_path", DEFAULT_METRICS_PATH)),
        "model_error": model_error,
    }


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    model = _require_model()
    metrics = getattr(app.state, "metrics", None)
    return ModelInfoResponse(
        model_path=str(getattr(app.state, "model_path", DEFAULT_MODEL_PATH)),
        metrics_path=str(getattr(app.state, "metrics_path", DEFAULT_METRICS_PATH)),
        numeric_features=list(model.numeric_features),
        categorical_features=list(model.categorical_features),
        target_name=str(getattr(model, "target_name", "<unknown>")),
        target_is_synthetic=bool(getattr(model, "target_is_synthetic", False)),
        text_source_column=str(getattr(model, "text_source_column", "<unknown>")),
        text_weight=float(getattr(model, "text_weight", 0.0)),
        metrics=metrics,
    )


@app.post("/reload-model")
def reload_model() -> Dict[str, Any]:
    _load_service_state()
    model_loaded = getattr(app.state, "model", None) is not None
    return {
        "model_loaded": model_loaded,
        "model_path": str(getattr(app.state, "model_path", DEFAULT_MODEL_PATH)),
        "model_error": getattr(app.state, "model_error", None),
    }


@app.post("/predict/features", response_model=PredictFeaturesResponse)
def predict_features(request: PredictFeaturesRequest) -> PredictFeaturesResponse:
    raw_df = pd.DataFrame(request.rows)
    predicted = _predict_dataframe(raw_df, text_column=request.text_column)

    rows: List[PredictionItem] = []
    for idx, row in predicted.reset_index(drop=True).iterrows():
        rows.append(
            PredictionItem(
                row_index=int(idx),
                predicted_repair_cost=float(row["predicted_repair_cost"]),
                predicted_repair_cost_label=str(row["predicted_repair_cost_label"]),
                predicted_repair=str(row["predicted_repair"]),
            )
        )

    return PredictFeaturesResponse(count=len(rows), predictions=rows)


@app.post("/predict/csv")
async def predict_csv(
    file: UploadFile = File(...),
    text_column: Optional[str] = Form(default=None),
    response_format: Literal["json", "csv"] = Form(default="json"),
) -> Any:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        raw_df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {exc}") from exc

    predicted = _predict_dataframe(raw_df, text_column=text_column)

    if response_format == "csv":
        output_name = f"predicted_{Path(file.filename or 'input').stem}.csv"
        headers = {"Content-Disposition": f'attachment; filename="{output_name}"'}
        csv_payload = predicted.to_csv(index=False)
        return Response(content=csv_payload, media_type="text/csv; charset=utf-8", headers=headers)

    return {
        "count": int(len(predicted)),
        "columns": list(predicted.columns),
        "predictions": predicted.to_dict(orient="records"),
    }
