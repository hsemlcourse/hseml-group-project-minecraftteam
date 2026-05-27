from __future__ import annotations

import io
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st


def _default_numeric_value(feature: str) -> float:
    defaults = {
        "brightness": 0.5,
        "contrast": 0.3,
        "blur_score": 0.4,
        "edge_density": 0.08,
        "num_objects": 5.0,
        "green_ratio": 0.1,
        "color_entropy": 0.35,
        "wall_floor_ratio": 2.3,
        "light_uniformity": 0.55,
        "num_windows": 1.0,
        "defect_score": 0.15,
        "furniture_count": 4.0,
        "furniture_density": 1.2,
        "aesthetic_score": 0.45,
    }
    return defaults.get(feature, 0.0)


def _api_url(path: str) -> str:
    base_url = st.session_state.api_base_url.rstrip("/")
    return f"{base_url}{path}"


def _get_json(path: str, timeout_sec: int = 15) -> Dict[str, Any]:
    response = requests.get(_api_url(path), timeout=timeout_sec)
    response.raise_for_status()
    return response.json()


def _post_json(path: str, payload: Dict[str, Any], timeout_sec: int = 30) -> Dict[str, Any]:
    response = requests.post(_api_url(path), json=payload, timeout=timeout_sec)
    response.raise_for_status()
    return response.json()


def _post_multipart(path: str, files: Dict[str, Any], data: Dict[str, Any], timeout_sec: int = 60) -> requests.Response:
    response = requests.post(_api_url(path), files=files, data=data, timeout=timeout_sec)
    response.raise_for_status()
    return response


def _render_health_panel() -> None:
    st.sidebar.subheader("API")
    if st.sidebar.button("Check API", use_container_width=True):
        try:
            health_payload = _get_json("/health")
            if health_payload.get("model_loaded"):
                st.sidebar.success("API is ready")
            else:
                st.sidebar.warning(f"API reachable, but model is not loaded: {health_payload.get('model_error')}")
        except requests.RequestException as exc:
            st.sidebar.error(f"API is unavailable: {exc}")


def _load_model_info() -> Optional[Dict[str, Any]]:
    try:
        return _get_json("/model/info")
    except requests.RequestException as exc:
        st.error(f"Cannot load model info from API: {exc}")
        return None


def _single_prediction_tab(model_info: Dict[str, Any]) -> None:
    st.subheader("Single Prediction")

    numeric_features: List[str] = list(model_info.get("numeric_features", []))
    categorical_features: List[str] = list(model_info.get("categorical_features", []))

    default_location = "bedroom"
    default_scene = "bedroom"
    location_help = "Typical values: bathroom, bedroom, kitchen, frontyard, backyard, livingRoom."
    scene_help = "Detected scene class or your best guess."

    with st.form("single_prediction_form"):
        location_value = st.text_input("location", value=default_location, help=location_help)
        scene_value = st.text_input("scene_category", value=default_scene, help=scene_help)
        description_value = st.text_area(
            "description (optional)",
            value="",
            help="Free-text details about room condition.",
        )

        st.caption("Numeric features")
        left_col, right_col = st.columns(2)
        numeric_values: Dict[str, float] = {}
        for idx, feature in enumerate(numeric_features):
            column = left_col if idx % 2 == 0 else right_col
            with column:
                numeric_values[feature] = float(
                    st.number_input(
                        feature,
                        value=float(_default_numeric_value(feature)),
                        step=0.01,
                        format="%.4f",
                    )
                )

        submitted = st.form_submit_button("Estimate Repair Cost", use_container_width=True)

    if not submitted:
        return

    row: Dict[str, Any] = {feature: numeric_values.get(feature, 0.0) for feature in numeric_features}
    if "location" in categorical_features:
        row["location"] = location_value
    if "scene_category" in categorical_features:
        row["scene_category"] = scene_value
    if description_value.strip():
        row["description"] = description_value.strip()

    payload = {"rows": [row], "text_column": "description"}
    try:
        response_payload = _post_json("/predict/features", payload)
    except requests.RequestException as exc:
        st.error(f"Prediction request failed: {exc}")
        return

    predictions = response_payload.get("predictions", [])
    if not predictions:
        st.warning("Prediction response is empty.")
        return

    prediction = predictions[0]
    cost = float(prediction["predicted_repair_cost"])
    label = str(prediction["predicted_repair"])
    human_cost = str(prediction["predicted_repair_cost_label"])

    st.success("Prediction completed")
    first_col, second_col = st.columns(2)
    first_col.metric("Estimated cost", human_cost)
    second_col.metric("Repair type", label)
    st.caption(f"Raw value: {cost:.2f} RUB")


def _batch_prediction_tab() -> None:
    st.subheader("Batch Prediction from CSV")
    uploaded_file = st.file_uploader("Upload CSV with feature rows", type=["csv"])
    text_column = st.text_input(
        "text_column (optional)",
        value="description",
        help="Column with free text. Leave default unless your CSV uses another column name.",
    )

    if st.button("Run Batch Prediction", use_container_width=True):
        if uploaded_file is None:
            st.warning("Please upload a CSV file first.")
            return

        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
        form_data: Dict[str, Any] = {"response_format": "csv"}
        if text_column.strip():
            form_data["text_column"] = text_column.strip()

        with st.spinner("Running predictions..."):
            try:
                response = _post_multipart("/predict/csv", files=files, data=form_data)
            except requests.RequestException as exc:
                st.error(f"Batch prediction failed: {exc}")
                return

        predicted_csv = response.text
        result_df = pd.read_csv(io.StringIO(predicted_csv))

        st.success(f"Processed rows: {len(result_df)}")
        st.dataframe(result_df.head(50), use_container_width=True)
        st.download_button(
            label="Download predicted CSV",
            data=predicted_csv.encode("utf-8"),
            file_name=f"predicted_{uploaded_file.name}",
            mime="text/csv",
            use_container_width=True,
        )


def main() -> None:
    st.set_page_config(page_title="Repair Cost Estimator", page_icon="🛠️", layout="wide")
    st.title("Repair Cost Estimator")
    st.caption("FastAPI + Streamlit interface for apartment repair cost prediction.")

    default_api_url = os.getenv("API_BASE_URL", "http://localhost:8000")
    if "api_base_url" not in st.session_state:
        st.session_state.api_base_url = default_api_url

    st.session_state.api_base_url = st.sidebar.text_input(
        "FastAPI URL",
        value=st.session_state.api_base_url,
    )
    _render_health_panel()

    model_info = _load_model_info()
    if model_info is None:
        st.stop()

    metrics = model_info.get("metrics") or {}
    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("Target", str(model_info.get("target_name", "<unknown>")))
    info_col2.metric("Text weight", f"{float(model_info.get('text_weight', 0.0)):.2f}")
    info_col3.metric("MAE", f"{float(metrics.get('mae', 0.0)):.2f}")

    tab_single, tab_batch = st.tabs(["Single Row", "CSV Batch"])
    with tab_single:
        _single_prediction_tab(model_info)
    with tab_batch:
        _batch_prediction_tab()


if __name__ == "__main__":
    main()
