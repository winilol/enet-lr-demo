# 术后疼痛风险评分工具

这是一个用于研究展示的 Streamlit 应用，包含：

- ElasticNet Logistic Regression 主模型；
- sigmoid（Platt）概率校准；
- Platt 概率尺度的 Youden 风险阈值；
- 动态临床特征输入、校准概率和风险分层。

本工具仅供研究参考，不能替代临床判断，也不能用于向患者提供诊断或治疗建议。

## 当前真实 bundle

当前部署 bundle：

```text
model_export/elasticnet_platt_bundle.joblib
SHA-256 f7ebbbaf3227d792960628cf6598d9752120e0c5b906c0daa847a8ae23d3609d
```

该文件使用 `base_model` 和 `calibrator` 字段；`app.py` 已在加载时将它们分别
映射为内部的 `model_raw` 和 `cal_platt`。字段别名不参与数值计算。已使用真实
bundle 验证：网页推理与直接执行
`StandardScaler.transform → sel_idx → calibrator.predict_proba` 的结果差异
小于 `1e-12`。

bundle 中的 `questionnaire` 元数据把 `Medical_history` 描述为“是/否”，但训练
均值约为 `59.76`，训练脚本和 scaler 都将它作为连续数值。当前网页严格沿用训练
脚本的连续值输入，没有根据问卷描述重新编码。正式公开前应在原数据字典中确认
该变量的单位与编码；任何二值化或重新编码都属于新模型，不能仅在网页端修改。

## 目录

```text
pain_score_web/
├── app.py
├── requirements.txt
├── runtime.txt
├── README.md
├── training_export_snippet.py
└── model_export/
    └── elasticnet_platt_bundle.joblib   # 训练脚本生成后放入
```

## 1. 从训练脚本导出模型

将 `training_export_snippet.py` 的全部代码追加到现有训练脚本末尾。必须放在以下
对象已经训练完成之后：

```text
results
selected_feats
train_means
scaler
sel_idx
feature_names
y_tr
X_train_raw
final_thresholds
```

导出代码会生成：

```text
model_export/elasticnet_platt_bundle.joblib
```

bundle 包含：

- `selected_feats`
- `train_means`
- `scaler`
- `sel_idx`
- `feature_names`
- `model_raw`
- `cal_platt`
- `threshold`
- `prevalence`
- 可选的训练范围和 sklearn 版本等聚合元数据

注意：主模型对象从 `results['ElasticNet LR']` 获取。不要直接使用训练循环结束
后的 `cal_platt`，那里通常是最后一个 SVM 模型的校准器。

运行训练脚本后，应看到类似：

```text
模型 bundle 导出完成
{
  "selected_feats": ["VNC", "Pharmacotherapy", ...],
  "threshold": 0.23,
  "sklearn_version": "...",
  "sha256": "..."
}
```

将生成的 `model_export/elasticnet_platt_bundle.joblib` 复制到本目录的
`model_export/`。

## 2. 本地运行

当前真实 bundle 的 `meta` 记录使用 Python 3.13.9、scikit-learn 1.7.2、
joblib 1.5.2、pandas 2.3.3 和 numpy 2.3.5。项目已将部署依赖固定到这些版本，
以避免跨版本反序列化造成模型偏移。先查看训练机器版本：

```bash
python -c "import sys, sklearn; print(sys.version); print(sklearn.__version__)"
```

然后运行：

```bash
cd /Users/liuyanchen/python/pain_score_web
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

浏览器通常会自动打开 `http://localhost:8501`。

使用训练集均值作为所有输入时，应能正常得到概率。手动修改输入后，页面按以下
顺序计算：

```text
全部原始特征 + train_means 填充
→ StandardScaler.transform
→ 按 sel_idx 取选中特征
→ cal_platt.predict_proba
→ 与 Platt 尺度 threshold 比较
```

这一顺序不能调换。先选 5 个特征再标准化会与训练流程不一致。

## 3. 部署到 GitHub + Streamlit Community Cloud

### 3.1 上传前检查

需要提交到 GitHub 的文件：

```text
app.py
requirements.txt
runtime.txt
README.md
training_export_snippet.py       # 可选，用于保留模型导出依据
model_export/elasticnet_platt_bundle.joblib
```

不要提交：

```text
原始训练 Excel
时间验证集 Excel
患者 ID、姓名、住院号
患者级预测结果
.venv/
.streamlit/secrets.toml
```

模型仓库可以选择 Private；Streamlit Community Cloud 可以从私有 GitHub 仓库部署
公开应用。若把仓库设为 Public，需要确认模型系数和 bundle 是否允许公开。

### 3.2 创建 GitHub 仓库并推送

先在 GitHub 创建一个空仓库，例如：

```text
pain-score-web
```

然后在本目录执行：

```bash
cd /Users/liuyanchen/python/pain_score_web
git init -b main
git add app.py requirements.txt runtime.txt README.md training_export_snippet.py model_export/elasticnet_platt_bundle.joblib .gitignore
git commit -m "Add calibrated postoperative pain risk web app"
git remote add origin https://github.com/你的用户名/pain-score-web.git
git push -u origin main
```

首次使用 Git 时，先配置提交身份：

```bash
git config --global user.name "你的姓名"
git config --global user.email "你的 GitHub 邮箱"
```

也可以直接通过 GitHub 网页的 **Add file → Upload files** 上传上述文件；必须
保留 `model_export/` 相对路径。

### 3.3 连接 Streamlit Community Cloud

1. 登录 https://share.streamlit.io/；
2. 使用 GitHub 账号授权 Streamlit；
3. 点击 **Create app** 或 **New app**；
4. 选择刚创建的 GitHub 仓库和 `main` 分支；
5. Main file path 填写 `app.py`；
6. 选择公开访问；
7. 点击 **Deploy**。

首次部署会读取 `requirements.txt` 和 `runtime.txt`。等待构建完成后，页面会
显示类似：

```text
https://你的应用名.streamlit.app
```

实际地址以 Streamlit Cloud 页面显示为准。若构建失败，在该应用的 **Manage app**
和日志中检查：

- 模型路径是否为 `model_export/elasticnet_platt_bundle.joblib`；
- Python 3.13 与 scikit-learn 1.7.2 是否安装成功；
- joblib 是否通过 Git LFS 指针方式错误上传；
- `feature_names` 与 `sel_idx` 是否对应。

### 3.4 用 UptimeRobot 定时访问

1. 登录或注册 https://uptimerobot.com/；
2. 点击 **New monitor**；
3. Monitor type 选择 **HTTP(S)**；
4. 名称填写 `pain-score-web health`；
5. URL 填写：

   ```text
   https://你的应用名.streamlit.app/_stcore/health
   ```

6. Monitoring interval 选择免费账户允许的最短间隔（通常约 5 分钟）；
7. 可增加 Keyword monitor，关键字为 `ok`；
8. 保存并等待状态变为 **Up**。

健康检查成功时应看到 HTTP 200，响应正文为：

```text
ok
```

UptimeRobot 的定时请求可以降低应用因长时间无访问而休眠的概率，但不能保证永不
休眠，也不能规避 Streamlit Community Cloud 的休眠策略、重启、资源上限或服务
条款。若要求生产级持续可用，应使用付费托管或独立云服务器。

### 3.5 论文中的示例表述

> The final ElasticNet logistic regression model with Platt calibration was
> implemented as a web-based risk calculator using Streamlit and deployed
> with Streamlit Community Cloud (https://你的应用名.streamlit.app). The tool
> reports the calibrated probability and applies the prespecified Youden
> threshold from the Platt probability scale. It is provided for research
> demonstration only and is not intended to replace clinical judgment.

中文示例：

> 本研究将经 Platt 校准的 ElasticNet 逻辑回归模型封装为交互式网页工具，并基于
> Streamlit Community Cloud 公开部署（https://你的应用名.streamlit.app）。工具
> 输出校准概率，并使用 Platt 概率尺度下预设的 Youden 阈值进行风险分层。该工具
> 仅供研究展示，不能替代临床判断。

如果期刊要求代码可复现，建议在论文补充材料中同时给出：

- GitHub 仓库地址或访问权限说明；
- Streamlit 应用公开地址；
- bundle 的 SHA-256；
- Python、scikit-learn、joblib 版本；
- 训练集样本量、阳性数和阈值；
- 访问日期。

## 4. 注意事项

1. 不要上传原始患者数据、姓名、住院号、影像或可识别身份的信息。
2. joblib 反序列化可执行任意代码，只加载自己训练并核验过 SHA-256 的文件。
3. 训练与部署环境的 scikit-learn 版本应尽量一致。版本跨度过大可能导致
   unpickle 警告或失败。
4. 当前阈值是 Platt 概率尺度上的 Youden 阈值，不能套用到 raw 或 isotonic
   概率。
5. 当前工具输出研究模型概率，不是临床确诊概率，也未在本 Space 中实时完成
   外部验证。
6. 分类变量的 0/1 方向必须与训练数据一致。如果编码不同，请修改 `app.py`
   中的 `FEATURE_LABELS`、`CATEGORICAL_FEATURES` 或导出 bundle 的
   `feature_metadata.options`。
7. 若以后特征中存在多分类变量，可在 `render_feature_input()` 中增加
   `st.selectbox` 分支，并将所有类别映射为训练时使用的数值编码。
