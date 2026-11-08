"""术后疼痛风险评分 Streamlit 应用。

推理顺序必须与训练脚本一致：
  train_means 填充全部原始特征
  -> StandardScaler.transform
  -> 按 sel_idx 选择特征
  -> Platt 校准模型 predict_proba
  -> 与 Platt 尺度 Youden 阈值比较
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
# 环境变量便于本地冒烟测试；Hugging Face 部署通常不需要设置。
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

# 当前主模型中的已知二分类变量。后续若增加变量，可在这里扩展。
FEATURE_LABELS = {
    "VNC": "VNC",
    "Pharmacotherapy": "药物治疗",
    "Medical_history": "病史时间（原始单位）",
    "Maxillary_branch_involved": "上颌支受累",
    "Nutritional_risk": "营养风险",
}
CATEGORICAL_FEATURES = {
    "VNC",
    "Pharmacotherapy",
    "Maxillary_branch_involved",
    "Nutritional_risk",
}


@st.cache_resource
def load_bundle(path: str) -> dict[str, Any]:
    """加载并校验模型 bundle；Streamlit 进程内只读取一次。"""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"未找到模型文件：{model_path}。请将训练脚本导出的 joblib 放到该路径。"
        )
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict):
        raise TypeError("模型 bundle 必须是 dict")

    # 兼容当前实际 bundle 的字段名。这里只做引用别名，不改变任何模型数值。
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
        raise KeyError(f"模型 bundle 缺少字段: {sorted(missing)}")
    if len(loaded["feature_names"]) != len(loaded["train_means"]):
        raise ValueError("feature_names 与 train_means 数量不一致")
    if len(loaded["selected_feats"]) != len(loaded["sel_idx"]):
        raise ValueError("selected_feats 与 sel_idx 数量不一致")
    expected_selected = [
        loaded["feature_names"][int(index)] for index in loaded["sel_idx"]
    ]
    if list(loaded["selected_feats"]) != expected_selected:
        raise ValueError("selected_feats 与 sel_idx 的特征顺序不一致")
    return loaded


def predict_probability(
    bundle: Mapping[str, Any],
    user_values: Mapping[str, float | int | None],
) -> float:
    """严格按照训练流程计算单例 Platt 校准概率。

    number/selectbox 未提供的变量使用训练集均值。不能先取 selected_feats
    再标准化，因为训练时的 StandardScaler 使用了全部原始特征。
    """
    feature_names = list(bundle["feature_names"])
    means = dict(bundle["train_means"])
    row: dict[str, float] = {}
    for feature in feature_names:
        raw_value = user_values.get(feature)
        if raw_value is None or pd.isna(raw_value):
            raw_value = means.get(feature)
        if raw_value is None or pd.isna(raw_value):
            raise ValueError(f"特征 {feature} 没有输入值，且训练均值不可用")
        row[feature] = float(raw_value)

    # scaler 在训练时 fit 于全部特征，因此这里也必须按完整列序转换。
    features = pd.DataFrame([row], columns=feature_names)
    scaled = bundle["scaler"].transform(features)
    selected = scaled[:, np.asarray(bundle["sel_idx"], dtype=int)]
    probability = float(bundle["cal_platt"].predict_proba(selected)[0, 1])
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"模型返回了无效概率: {probability}")
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
    """动态渲染一个特征。分类变量使用 selectbox，其他变量使用 number_input。"""
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
        names = {0: "否", 1: "是"}
        return st.selectbox(
            label,
            options=options,
            index=default_index,
            format_func=lambda value: f"{value} · {names.get(int(value), '类别')}",
            help=f"训练集默认值（均值）：{default:.3f}。"
            "若数据编码不是 0/1，请修改 FEATURE_LABELS 与 options。",
            key=f"input__{feature}",
        )

    value, min_value, max_value, step = _numeric_metadata(meta, default)
    return st.number_input(
        label,
        value=value,
        min_value=min_value,
        max_value=max_value,
        step=step,
        help=f"训练集默认值（均值）：{default:.3f}。"
        "需要调整可接受范围时修改 bundle 的 feature_metadata 或本页代码。",
        key=f"input__{feature}",
    )


def main() -> None:
    st.set_page_config(
        page_title="术后疼痛风险评分工具",
        page_icon="🩺",
        layout="centered",
    )
    st.title("术后疼痛风险评分工具（ElasticNet + Platt）")
    st.caption(
        "输入临床特征后，工具输出 Platt 校准概率，并按预设 Youden 阈值进行风险分层。"
    )

    try:
        bundle = load_bundle(str(MODEL_PATH))
    except Exception as exc:
        st.error("模型文件未成功加载，无法计算风险。")
        st.code(str(MODEL_PATH), language=None)
        st.info(
            "请先在训练脚本末尾运行导出片段，然后把 "
            "model_export/elasticnet_platt_bundle.joblib 放入当前项目。"
        )
        st.exception(exc)
        st.stop()

    selected_feats = list(bundle["selected_feats"])
    # 严格使用训练 bundle 原样导出的阈值，网页端不再二次调整。
    threshold = float(bundle["threshold"])
    prevalence = bundle.get("prevalence")

    info_col, threshold_col, prevalence_col = st.columns(3)
    info_col.metric("模型", "ElasticNet LR")
    threshold_col.metric("Platt 阈值", f"{threshold:.3f}")
    if prevalence is None:
        prevalence_col.metric("训练阳性率", "未记录")
    else:
        prevalence_col.metric("训练阳性率", f"{float(prevalence):.1%}")

    with st.expander("模型说明与选用特征", expanded=True):
        st.write(
            "模型先进行 ElasticNet Logistic Regression 特征选择，再用 sigmoid "
            "（Platt）校准输出概率。风险分层使用主分析在 Platt 概率尺度上得到的 "
            "Youden 阈值；不要与 raw 或 isotonic 尺度阈值混用。"
        )
        st.markdown("**最终选用特征：** " + "、".join(selected_feats))
        st.caption(
            "未在当前表单中展示的原始特征会使用训练集均值填充；"
            "标准化仍需完整特征向量，因此这些均值不是可省略的预处理步骤。"
        )
    with st.form("pain_risk_form"):
        st.subheader("输入临床特征")
        widget_columns = st.columns(2)
        user_values: dict[str, float | int] = {}
        for index, feature in enumerate(selected_feats):
            with widget_columns[index % 2]:
                user_values[feature] = render_feature_input(bundle, feature)
        submitted = st.form_submit_button(
            "计算风险",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        try:
            probability = predict_probability(bundle, user_values)
        except Exception as exc:
            st.error(f"计算失败：{exc}")
        else:
            is_high_risk = probability >= threshold
            st.divider()
            result_col, risk_col = st.columns(2)
            result_col.metric("Platt 校准概率", f"{probability:.3f}")
            risk_col.metric("决策阈值", f"{threshold:.3f}")
            risk_text = "高风险" if is_high_risk else "低风险"
            if is_high_risk:
                st.error(
                    f"风险分层：{risk_text}（预测概率 {probability:.3f} "
                    f"≥ 阈值 {threshold:.3f}）"
                )
            else:
                st.success(
                    f"风险分层：{risk_text}（预测概率 {probability:.3f} "
                    f"< 阈值 {threshold:.3f}）"
                )

    st.divider()
    st.warning("本工具仅供研究参考，不能替代临床判断。")
    st.caption(
        "请勿输入姓名、病历号、身份证号或其他可识别患者身份的信息。"
        "阈值、校准与模型性能基于既往研究队列，外部泛化需独立验证。"
    )


if __name__ == "__main__":
    main()
