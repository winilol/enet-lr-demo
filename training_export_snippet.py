"""追加到现有训练脚本末尾的模型导出代码。

运行训练脚本时，请确保以下变量已经存在：
  results, selected_feats, train_means, scaler, sel_idx,
  feature_names, y_tr, X_train_raw, final_thresholds

不要直接导出循环变量 cal_platt；循环在四个模型上结束后它不再是
ElasticNet 主模型。这里显式从 results['ElasticNet LR'] 获取对象。
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
    raise RuntimeError("results 中没有 ElasticNet LR，不能导出主模型")
if _primary_result["chosen"]["method"] == "failed":
    raise RuntimeError("ElasticNet LR 训练失败，未生成可导出的 Platt 模型")

_expected_selected = [feature_names[index] for index in sel_idx]
if list(selected_feats) != _expected_selected:
    raise ValueError("selected_feats 与 sel_idx 的特征顺序不一致")

# 可选元数据只包含聚合统计，不包含患者级记录；用于网页输入控件范围。
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
    # 该阈值来自 Platt 概率尺度，网页必须对 cal_platt 的输出使用此阈值。
    "threshold": _threshold,
    "prevalence": float(np.mean(y_tr)),
    # 以下为便于部署审计的可选元数据。
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
print("\n模型 bundle 导出完成")
print(json.dumps(_summary, ensure_ascii=False, indent=2))
