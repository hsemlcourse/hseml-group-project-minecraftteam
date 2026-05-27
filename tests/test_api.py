from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import app


MODEL_PATH = Path("models/repair_cost_multimodal.joblib")


@pytest.fixture(scope="module")
def client() -> TestClient:
    if not MODEL_PATH.exists():
        pytest.skip(f"Model file not found: {MODEL_PATH}")
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["model_loaded"] is True
    assert payload["status"] == "ok"


def test_predict_features_endpoint(client: TestClient) -> None:
    request_payload = {
        "rows": [
            {
                "location": "bedroom",
                "scene_category": "bedroom",
                "brightness": 0.52,
                "contrast": 0.31,
                "blur_score": 0.25,
                "edge_density": 0.09,
                "num_objects": 4,
                "green_ratio": 0.05,
                "color_entropy": 0.41,
                "wall_floor_ratio": 2.3,
                "light_uniformity": 0.62,
                "num_windows": 1,
                "defect_score": 0.12,
                "furniture_count": 3,
                "furniture_density": 1.1,
                "aesthetic_score": 0.53,
                "description": "Room with light cosmetic wear and old paint.",
            }
        ],
        "text_column": "description",
    }

    response = client.post("/predict/features", json=request_payload)
    assert response.status_code == 200

    payload = response.json()
    assert payload["count"] == 1
    assert payload["predictions"][0]["predicted_repair_cost"] > 0
