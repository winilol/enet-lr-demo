"""Postoperative pain risk scoring Streamlit application.

The inference order must match the training pipeline:
  fill all raw features with train_means
  -> StandardScaler.transform
  -> select features using sel_idx
  -> Platt-calibrated predict_proba
  -> compare with the threshold stored in the trained bundle
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import streamlit as st


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "model_export"
    / "elasticnet_platt_bundle.joblib"
)
# Optional local override for testing. Deployment uses the default path.
MODEL_PATH = Path(
    os.environ.get("PAIN_SCORE_MODEL_PATH", str(DEFAULT_MODEL_PATH))
).expanduser().resolve()

REQUIRED_KEYS = {
    "selected_feats",
    "train_means",
    "scaler",
    "sel_idx",
    "feature_names",
    "model_raw",
    "cal_platt",
    "threshold",
}

# Binary predictors used by the current model.
FEATURE_LABELS = {
    "VNC": "Multiple vessels compressing the trigeminal nerve",
    "Pharmacotherapy": "Favorable response to carbamazepine",
    "Medical_history": "Medical history value",
    "Maxillary_branch_involved": "Maxillary branch involvement",
    "Nutritional_risk": "Nutritional risk",
}
CATEGORICAL_FEATURES = {
    "VNC",
    "Pharmacotherapy",
    "Maxillary_branch_involved",
    "Nutritional_risk",
}


@st.cache_resource
def load_bundle(path: str) -> dict[str, Any]:
    """Load and validate the model bundle once per Streamlit process."""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Place the exported joblib file at this path."
        )
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict):
        raise TypeError("The model bundle must be a dictionary.")

    # Support exported field aliases without changing any model values.
    loaded = dict(artifact)
    if "model_raw" not in loaded and "base_model" in loaded:
        loaded["model_raw"] = loaded["base_model"]
    if "cal_platt" not in loaded and "calibrator" in loaded:
        loaded["cal_platt"] = loaded["calibrator"]
    meta = loaded.get("meta", {})
    if isinstance(meta, Mapping):
        aliases = {
            "prevalence": "prevalence_train",
            "model_name": "target_model",
            "calibration_method": "calibration_method",
            "threshold_scale": "probability_scale",
            "train_n": "n_train",
            "sklearn_version": "sklearn_version",
        }
        for canonical, actual in aliases.items():
            if canonical not in loaded and actual in meta:
                loaded[canonical] = meta[actual]

    missing = REQUIRED_KEYS.difference(loaded)
    if missing:
        raise KeyError(f"Model bundle is missing fields: {sorted(missing)}")
    if len(loaded["feature_names"]) != len(loaded["train_means"]):
        raise ValueError("feature_names and train_means have different lengths.")
    if len(loaded["selected_feats"]) != len(loaded["sel_idx"]):
        raise ValueError("selected_feats and sel_idx have different lengths.")
    expected_selected = [
        loaded["feature_names"][int(index)] for index in loaded["sel_idx"]
    ]
    if list(loaded["selected_feats"]) != expected_selected:
        raise ValueError("selected_feats does not match the order in sel_idx.")
    return loaded


def predict_probability(
    bundle: Mapping[str, Any],
    user_values: Mapping[str, float | int | None],
) -> float:
    """Calculate one Platt-calibrated probability.

    Features not present in the form are filled with development-cohort
    reference values. All raw features must be scaled before selecting the
    model subset because the training StandardScaler used the full vector.
    """
    feature_names = list(bundle["feature_names"])
    means = dict(bundle["train_means"])
    row: dict[str, float] = {}
    for feature in feature_names:
        raw_value = user_values.get(feature)
        if raw_value is None or pd.isna(raw_value):
            raw_value = means.get(feature)
        if raw_value is None or pd.isna(raw_value):
            raise ValueError(
                f"No value or fallback reference is available for feature {feature}."
            )
        row[feature] = float(raw_value)

    # The scaler was fitted on all features, so transform the complete row.
    features = pd.DataFrame([row], columns=feature_names)
    scaled = bundle["scaler"].transform(features)
    selected = scaled[:, np.asarray(bundle["sel_idx"], dtype=int)]
    probability = float(bundle["cal_platt"].predict_proba(selected)[0, 1])
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"The model returned an invalid probability: {probability}")
    return probability


def _feature_meta(bundle: Mapping[str, Any], feature: str) -> dict[str, Any]:
    metadata = bundle.get("feature_metadata", {})
    if isinstance(metadata, Mapping) and feature in metadata:
        return dict(metadata[feature])
    if feature in CATEGORICAL_FEATURES:
        return {"kind": "binary", "options": [0, 1]}
    return {
        "kind": "numeric",
        "mean": float(bundle["train_means"][feature]),
    }


def _binary_options(meta: Mapping[str, Any]) -> list[int]:
    values = meta.get("options", [0, 1])
    try:
        result = sorted({int(value) for value in values})
    except (TypeError, ValueError):
        result = [0, 1]
    return result if result else [0, 1]


def _numeric_metadata(
    meta: Mapping[str, Any],
    mean: float,
) -> tuple[float, float | None, float | None, float]:
    value = float(meta.get("mean", mean))
    min_value = meta.get("min")
    max_value = meta.get("max")
    min_value = float(min_value) if min_value is not None and np.isfinite(min_value) else None
    max_value = float(max_value) if max_value is not None and np.isfinite(max_value) else None
    step = 1.0 if float(value).is_integer() else 0.1
    return value, min_value, max_value, step


def render_feature_input(bundle: Mapping[str, Any], feature: str) -> float | int:
    """Render one predictor as a select box or numeric input."""
    meta = _feature_meta(bundle, feature)
    label = FEATURE_LABELS.get(feature, feature)
    default = float(bundle["train_means"][feature])
    kind = str(meta.get("kind", "numeric")).lower()

    if feature in CATEGORICAL_FEATURES or kind in {"binary", "categorical"}:
        options = _binary_options(meta)
        if len(options) < 2:
            options = [0, 1]
        default_option = int(round(default))
        default_index = options.index(default_option) if default_option in options else 0
        names = {0: "No", 1: "Yes"}
        return st.selectbox(
            label,
            options=options,
            index=default_index,
            format_func=lambda value: names.get(
                int(value), f"Category {int(value)}"
            ),
            help="Select the option that best reflects the clinical assessment.",
            key=f"input__{feature}",
        )

    value, min_value, max_value, step = _numeric_metadata(meta, default)
    return st.number_input(
        label,
        value=value,
        min_value=min_value,
        max_value=max_value,
        step=step,
        help="Enter the value using the same scale used for model development.",
        key=f"input__{feature}",
    )


def main() -> None:
    st.set_page_config(
        page_title="Postoperative Pain Risk Assessment",
        page_icon="🩺",
        layout="centered",
    )
    st.title("Postoperative Pain Risk Assessment")
    st.caption(
        "Enter the patient's clinical characteristics to estimate calibrated "
        "postoperative pain risk."
    )

    try:
        bundle = load_bundle(str(MODEL_PATH))
    except Exception as exc:
        st.error("The model could not be loaded. Risk assessment is unavailable.")
        st.code(str(MODEL_PATH), language=None)
        st.info(
            "Run the model export snippet and place "
            "model_export/elasticnet_platt_bundle.joblib in the project."
        )
        st.exception(exc)
        st.stop()

    selected_feats = list(bundle["selected_feats"])
    # Use the exported decision cutoff exactly as stored. Do not modify it.
    threshold = float(bundle["threshold"])

    model_col, calibration_col, output_col = st.columns(3)
    model_col.metric("Model", "ElasticNet LR")
    calibration_col.metric("Calibration", "Platt")
    output_col.metric("Output", "Pain risk")

    with st.expander("About this assessment", expanded=True):
        st.write(
            "This tool estimates the probability of postoperative pain using a "
            "parsimonious ElasticNet logistic regression model with Platt "
            "probability calibration."
        )
        readable_features = [
            FEATURE_LABELS.get(feature, feature) for feature in selected_feats
        ]
        st.markdown("**Clinical factors included:** " + "; ".join(readable_features))
        st.caption(
            "Factors not shown in the form are handled using development-"
            "cohort reference values required by the fitted model."
        )
    with st.form("pain_risk_form"):
        st.subheader("Clinical assessment")
        widget_columns = st.columns(2)
        user_values: dict[str, float | int] = {}
        for index, feature in enumerate(selected_feats):
            with widget_columns[index % 2]:
                user_values[feature] = render_feature_input(bundle, feature)
        submitted = st.form_submit_button(
            "Calculate risk",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        try:
            probability = predict_probability(bundle, user_values)
        except Exception as exc:
            st.error(f"Calculation failed: {exc}")
        else:
            is_high_risk = probability >= threshold
            st.divider()
            result_col, risk_col = st.columns(2)
            result_col.metric("Calibrated risk probability", f"{probability:.3f}")
            risk_text = "Higher-risk group" if is_high_risk else "Lower-risk group"
            if is_high_risk:
                st.error(f"Risk classification: {risk_text}")
            else:
                st.success(f"Risk classification: {risk_text}")
            risk_col.metric("Risk group", risk_text)

    st.divider()
    st.warning(
        "This tool is intended for research and clinical decision support only. "
        "It does not replace professional clinical judgment."
    )
    st.caption(
        "Do not enter names, medical record numbers, or other identifiable "
        "patient information. Risk estimates should be interpreted in the "
        "context of the full clinical assessment."
    )


if __name__ == "__main__":
    main()
