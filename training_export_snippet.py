"""Append this code to the end of the existing model-training script.

The following objects must already exist:
  results, selected_feats, train_means, scaler, sel_idx,
  feature_names, y_tr, X_train_raw, final_thresholds

Do not export the loop variable named cal_platt. After the loop finishes, it
refers to the last model rather than the primary ElasticNet model. The object
is retrieved explicitly from results['ElasticNet LR'].
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn


_PRIMARY_MODEL_NAME = "ElasticNet LR"
_primary_result = results.get(_PRIMARY_MODEL_NAME)
if _primary_result is None:
    raise RuntimeError("results does not contain the ElasticNet LR model")
if _primary_result["chosen"]["method"] == "failed":
    raise RuntimeError("ElasticNet LR training failed; no Platt model to export")

_expected_selected = [feature_names[index] for index in sel_idx]
if list(selected_feats) != _expected_selected:
    raise ValueError("selected_feats does not match the order in sel_idx")

# Optional metadata contains aggregate statistics only, never patient records.
_feature_metadata = {}
for _feature in feature_names:
    _series = X_train_raw[_feature].dropna()
    _mean = float(train_means[_feature])
    if _series.nunique() <= 2:
        _options = sorted(float(value) for value in _series.unique())
        _feature_metadata[_feature] = {
            "kind": "binary",
            "mean": _mean,
            "options": _options,
        }
    else:
        _feature_metadata[_feature] = {
            "kind": "numeric",
            "mean": _mean,
            "min": float(_series.min()),
            "max": float(_series.max()),
            "median": float(_series.median()),
        }

_threshold = float(
    final_thresholds[_PRIMARY_MODEL_NAME]["platt"]["youden"]
)
_bundle = {
    "selected_feats": [str(value) for value in selected_feats],
    "train_means": {
        str(feature): float(value) for feature, value in train_means.items()
    },
    "scaler": scaler,
    "sel_idx": [int(value) for value in sel_idx],
    "feature_names": [str(value) for value in feature_names],
    "model_raw": _primary_result["model_raw"],
    "cal_platt": _primary_result["cal_platt"],
    # This cutoff belongs to the Platt probability scale.
    "threshold": _threshold,
    "prevalence": float(np.mean(y_tr)),
    # Optional metadata for deployment and audit.
    "feature_metadata": _feature_metadata,
    "model_name": _PRIMARY_MODEL_NAME,
    "calibration_method": "sigmoid",
    "threshold_metric": "Youden",
    "threshold_scale": "platt",
    "train_n": int(len(y_tr)),
    "train_positive_n": int(np.sum(y_tr)),
    "sklearn_version": sklearn.__version__,
    "exported_at_utc": datetime.now(timezone.utc).isoformat(),
}

_bundle_dir = Path("model_export")
_bundle_dir.mkdir(parents=True, exist_ok=True)
_bundle_path = _bundle_dir / "elasticnet_platt_bundle.joblib"
_temporary_path = _bundle_path.with_suffix(".joblib.tmp")
joblib.dump(_bundle, _temporary_path, compress=3)
_temporary_path.replace(_bundle_path)

_sha256 = hashlib.sha256(_bundle_path.read_bytes()).hexdigest()
_summary = {
    "path": str(_bundle_path.resolve()),
    "selected_feats": _bundle["selected_feats"],
    "threshold": _threshold,
    "prevalence": _bundle["prevalence"],
    "train_n": _bundle["train_n"],
    "train_positive_n": _bundle["train_positive_n"],
    "sklearn_version": sklearn.__version__,
    "sha256": _sha256,
}
print("\nModel bundle export completed")
print(json.dumps(_summary, ensure_ascii=False, indent=2))
