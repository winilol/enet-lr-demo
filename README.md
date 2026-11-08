# Postoperative Pain Risk Assessment

A web-based clinical risk scoring aid for estimating postoperative pain risk
from routinely collected clinical characteristics.

The deployed application uses:

- ElasticNet logistic regression for parsimonious feature selection and
  prediction;
- Platt calibration for probability estimation;
- the prespecified decision rule stored in the exported model bundle.

The model-estimated probability and risk group are displayed to support
structured clinical assessment. The application does not display model
coefficients, development-cohort prevalence, or internal decision cutoffs.

> This tool is intended for research and clinical decision support only. It
> does not replace professional clinical judgment.

## Clinical Inputs

The current model uses:

- multiple vessels compressing the trigeminal nerve;
- favorable response to carbamazepine;
- medical history value;
- maxillary branch involvement;
- nutritional risk.

Binary items are entered as **Yes/No**. Other variables use the numeric scale
used during model development.

## Model Bundle

The application expects:

```text
model_export/elasticnet_platt_bundle.joblib
```

The bundle contains the fitted model, scaler, development-cohort reference
values, selected feature indices, and the internal decision rule. The
application consumes these objects without refitting or recalibrating the
model.

Current bundle checksum:

```text
f7ebbbaf3227d792960628cf6598d9752120e0c5b906c0daa847a8ae23d3609d
```

To regenerate the bundle, append `training_export_snippet.py` to the original
training script after the fitted ElasticNet and Platt-calibrated model objects
have been created. The export must use `results["ElasticNet LR"]`, not the
calibration variable left over after a model loop.

## Repository Contents

```text
enet-lr-demo/
├── app.py
├── requirements.txt
├── runtime.txt
├── README.md
├── LICENSE
├── training_export_snippet.py
└── model_export/
    └── elasticnet_platt_bundle.joblib
```

## Run Locally

The model bundle was exported with Python 3.13.9 and records
scikit-learn 1.7.2. The dependency versions are pinned to avoid
cross-version prediction drift.

```bash
cd enet-lr-demo
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

The application will normally open at:

```text
http://localhost:8501
```

The prediction sequence is intentionally fixed:

```text
complete raw feature vector
→ fill unavailable features with development-cohort reference values
→ StandardScaler.transform
→ select model features by saved indices
→ Platt-calibrated predict_proba
→ apply the decision rule stored in the bundle
```

Do not change the order of these operations. Scaling the selected subset
directly would not reproduce the training pipeline.

## Deploy with GitHub and Streamlit Community Cloud

### 1. Verify the repository

The public repository is:

```text
https://github.com/winilol/enet-lr-demo
```

The application must be available at the repository root:

```text
app.py
```

The model must remain at:

```text
model_export/elasticnet_platt_bundle.joblib
```

### 2. Create the Streamlit app

1. Open <https://share.streamlit.io/>.
2. Sign in with GitHub and authorize Streamlit.
3. Select **Create app** or **New app**.
4. Choose repository `winilol/enet-lr-demo`.
5. Select branch `main`.
6. Set the main file path to `app.py`.
7. Choose a stable public app name.
8. Deploy the application.

Streamlit will install dependencies from `requirements.txt` and select the
Python version from `runtime.txt`.

The resulting public URL will follow a form similar to:

```text
https://your-app-name.streamlit.app
```

The exact URL is assigned by Streamlit Community Cloud.

### 3. Verify the deployed application

After the build reaches the running state:

1. Enter representative clinical values.
2. Confirm that the calibrated probability appears with three decimals.
3. Confirm that the risk group appears in the result panel.
4. Confirm that no internal cutoff, prevalence, or development-set metric is
   displayed.
5. Check the health endpoint:

```text
https://your-app-name.streamlit.app/_stcore/health
```

The response should contain:

```text
ok
```

The command-line equivalent is:

```bash
curl -i https://your-app-name.streamlit.app/_stcore/health
```

An HTTP 200 response indicates that the application server is responding.

## Keep the App Available

Streamlit Community Cloud may put inactive applications to sleep. A lightweight
external uptime check can reduce the likelihood of inactivity-related sleeping:

1. Sign in to <https://uptimerobot.com/>.
2. Select **New monitor**.
3. Choose monitor type **HTTP(S)**.
4. Enter the deployed health endpoint:

   ```text
   https://your-app-name.streamlit.app/_stcore/health
   ```

5. Set the shortest interval allowed by the account.
6. Optionally require the keyword `ok`.
7. Configure an email address for downtime alerts.
8. Save and verify that the monitor status is **Up**.

An uptime monitor is best-effort support. It cannot guarantee uninterrupted
availability or override platform restart, resource, and usage policies.

## Privacy and Security

- Do not upload raw patient data, source spreadsheets, medical record numbers,
  names, or other identifiable information.
- The repository should contain only application code, deployment metadata,
  and the fitted model bundle.
- A joblib file can execute code during deserialization. Load only a bundle
  that you created and whose SHA-256 checksum you have verified.
- Review the model and interface with the relevant clinical and data-governance
  teams before public use.

## Citation Suggestion

> The final ElasticNet logistic regression model with Platt calibration was
> implemented as a web-based clinical risk scoring aid using Streamlit and
> deployed with Streamlit Community Cloud
> (https://your-app-name.streamlit.app). The tool provides a calibrated risk
> estimate and risk group for research and clinical decision support. It is
> not intended to replace professional clinical judgment.
