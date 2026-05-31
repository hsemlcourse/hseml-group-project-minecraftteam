from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repair_cost_experiments import build_modeling_frame  # noqa: E402
from repair_cost_model import (  # noqa: E402
    DEFAULT_NUMERIC_FEATURES,
    DESCRIPTION_COLUMN,
    IQRClipper,
    prepare_inference_data,
    prepare_training_data,
)


def _sample_room_rows() -> pd.DataFrame:
    rows = []
    locations = ["bathroom", "kitchen", "bedroom", "frontyard"]
    for index in range(40):
        rows.append(
            {
                "filename": f"room_{index}.jpg",
                "location": locations[index % len(locations)],
                "scene_category": "interior",
                "brightness": 0.35 + index * 0.002,
                "contrast": 0.18,
                "blur_score": 0.5,
                "edge_density": 0.07,
                "num_objects": index % 8,
                "green_ratio": 0.01,
                "color_entropy": 0.18,
                "wall_floor_ratio": 2.31,
                "light_uniformity": 0.6,
                "num_windows": index % 3,
                "defect_score": 0.03 + index * 0.0005,
                "furniture_count": index % 5,
                "furniture_density": float(index % 5) * 4.0,
                "aesthetic_score": 0.45,
            }
        )
    return pd.DataFrame(rows)


def test_prepare_training_data_adds_engineered_features() -> None:
    df = _sample_room_rows()
    bundle = prepare_training_data(df)

    assert "low_light_defect_interaction" in bundle.features.columns
    assert "clutter_score" in bundle.features.columns
    assert DESCRIPTION_COLUMN in bundle.features.columns
    assert set(DEFAULT_NUMERIC_FEATURES).issubset(bundle.features.columns)
    assert bundle.target_is_synthetic is True
    assert bundle.target.notna().all()


def test_prepare_inference_data_matches_training_columns() -> None:
    df = _sample_room_rows().drop(columns=["scene_category"])
    inference = prepare_inference_data(
        df,
        numeric_features=DEFAULT_NUMERIC_FEATURES,
        categorical_features=["location", "scene_category"],
    )

    assert "scene_category" in inference.columns
    assert "visual_complexity" in inference.columns
    assert inference.isna().sum().sum() == 0


def test_build_modeling_frame_removes_duplicates_and_reports_quality() -> None:
    df = _sample_room_rows()
    duplicated = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    duplicated.loc[1, "location"] = None

    modeling_frame, target, summary = build_modeling_frame(duplicated)

    assert summary.rows_before == 41
    assert summary.rows_after == 40
    assert summary.duplicates_removed == 1
    assert modeling_frame.shape[0] == len(target)
    assert modeling_frame.isna().sum().sum() == 0


def test_iqr_clipper_clips_train_outlier() -> None:
    values = np.array([[1.0], [1.0], [1.0], [1.0], [1000.0]])
    clipper = IQRClipper().fit(values)
    transformed = clipper.transform(values)

    assert transformed.max() == 1.0
