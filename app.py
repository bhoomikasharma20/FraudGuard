import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string
from tensorflow.keras.models import load_model

app = Flask(__name__)

MODEL_PATH = "/content/credit_fraud_ann.keras"
SCALER_PATH = "/content/credit_fraud_scaler.pkl"
THRESHOLD_PATH = "/content/fraud_threshold.pkl"

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

model = load_model(MODEL_PATH)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

with open(THRESHOLD_PATH, "rb") as f:
    FRAUD_THRESHOLD = float(pickle.load(f))

if model.input_shape[-1] != 30 or scaler.n_features_in_ != 30:
    raise ValueError("Model and scaler must both accept 30 features.")

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FraudGuard | ANN Dashboard</title>
<style>
:root {
  --navy:#101a35; --blue:#4169e1; --bg:#f4f7fc;
  --muted:#738099; --green:#12805c; --red:#c0394b;
  --line:#e6ebf3; --white:#fff;
}
* {box-sizing:border-box}
html {scroll-behavior:smooth}
body {
  margin:0;background:var(--bg);color:#1d2942;
  font-family:Inter,Arial,sans-serif;
}
header {
  background:linear-gradient(120deg,#101a35,#203d70);
  color:white;padding:25px max(20px,calc((100% - 1150px)/2));
}
.brand {font-size:12px;letter-spacing:2px;color:#a9c5ff;font-weight:bold}
header h1 {margin:10px 0 6px;font-size:clamp(24px,4vw,34px)}
header p {color:#d2dcf0;margin:0;font-size:14px}
nav {
  display:flex;gap:22px;flex-wrap:wrap;margin-top:22px;
}
nav a {color:#fff;text-decoration:none;font-size:13px;opacity:.9}
main {max-width:1150px;margin:25px auto;padding:0 18px}
.section-title {margin:0 0 6px;font-size:22px}
.subtitle {color:var(--muted);font-size:13px;line-height:1.6;margin:0 0 20px}
.metrics {
  display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;
}
.card,.panel {
  background:white;border:1px solid var(--line);border-radius:15px;
  padding:20px;box-shadow:0 5px 20px #17274708;
}
.metric-label {font-size:12px;color:var(--muted)}
.metric-value {font-size:27px;font-weight:750;margin:10px 0 6px}
.metric-note {font-size:11px;color:var(--muted);line-height:1.5}
.layout {
  display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:20px;
}
.panel h3 {margin:0 0 8px;font-size:17px}
table {border-collapse:collapse;width:100%;font-size:13px;margin-top:15px}
th,td {padding:12px 9px;border-bottom:1px solid var(--line);text-align:center}
th {background:#f5f7fc;color:#5b6780;font-weight:600}
.good {color:var(--green);font-weight:bold}
.bad {color:var(--red);font-weight:bold}
.form-grid {
  display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:13px;
}
label {display:block;font-size:12px;font-weight:650;margin-bottom:6px}
input {
  width:100%;min-width:0;padding:11px 10px;border:1px solid #d8e0ed;
  border-radius:8px;background:white;font-size:14px;color:#172033;
}
input:focus {outline:2px solid #c6d6ff;border-color:var(--blue)}
button {
  background:var(--blue);color:white;border:0;border-radius:9px;
  padding:14px 18px;font-size:14px;font-weight:bold;cursor:pointer;width:100%;
  margin-top:20px;
}
button:hover {background:#3154c5}
.result {
  border:1px solid #bde8d7;background:#effaf5;border-radius:12px;
  padding:17px;margin:18px 0;
}
.result.fraud {border-color:#f4c5cb;background:#fff2f3}
.result h3 {margin:0 0 8px}
.score {font-size:25px;font-weight:750;margin:8px 0}
.error {background:#fff5e9;border-color:#f5d7a8}
.pill {
  display:inline-block;padding:5px 9px;border-radius:20px;
  background:#eaf0ff;color:#3154b5;font-size:11px;font-weight:bold;
}
.note {font-size:12px;color:var(--muted);line-height:1.7}
footer {text-align:center;padding:30px 15px;color:var(--muted);font-size:12px}
@media(max-width:800px) {
  .metrics {grid-template-columns:repeat(2,minmax(0,1fr))}
  .layout {grid-template-columns:1fr}
}
@media(max-width:480px) {
  .form-grid {grid-template-columns:repeat(2,minmax(0,1fr))}
  .card,.panel {padding:14px}
  .metric-value {font-size:23px}
}
</style>
</head>
<body>
<header>
  <div class="brand">FRAUDGUARD · MACHINE LEARNING</div>
  <h1>Credit Card Fraud Detection</h1>
  <p>Artificial Neural Network (ANN) · Academic Project Dashboard</p>
  <nav>
    <a href="#dashboard">Dashboard</a>
    <a href="#performance">Model Performance</a>
    <a href="#analyzer">Analyze Transaction</a>
  </nav>
</header>

<main>
<section id="dashboard">
  <h2 class="section-title">Model Overview</h2>
  <p class="subtitle">Evaluation metrics from the saved held-out test-set evaluation.</p>
  <div class="metrics">
    <div class="card">
      <div class="metric-label">Test Accuracy</div>
      <div class="metric-value">99.92%</div>
      <div class="metric-note">Overall correct predictions</div>
    </div>
    <div class="card">
      <div class="metric-label">Fraud Precision</div>
      <div class="metric-value">73.17%</div>
      <div class="metric-note">Flagged cases that were fraud</div>
    </div>
    <div class="card">
      <div class="metric-label">Fraud Recall</div>
      <div class="metric-value">81.08%</div>
      <div class="metric-note">Fraud cases detected</div>
    </div>
    <div class="card">
      <div class="metric-label">Fraud F1-score</div>
      <div class="metric-value">76.92%</div>
      <div class="metric-note">Balance of precision and recall</div>
    </div>
  </div>
  <div class="layout" id="performance">
    <div class="panel">
      <h3>Confusion Matrix</h3>
      <p class="subtitle">Held-out test set · 42,722 transactions</p>
      <table>
        <tr><th>Actual / Predicted</th><th>Genuine</th><th>Fraud</th></tr>
        <tr><th>Genuine</th><td class="good">42,626</td><td class="bad">22</td></tr>
        <tr><th>Fraud</th><td class="bad">14</td><td class="good">60</td></tr>
      </table>
      <p class="note">60 fraud cases detected; 14 fraud cases missed; 22 genuine cases incorrectly flagged.</p>
    </div>
    <div class="panel">
      <h3>Model Information</h3>
      <table>
        <tr><th>Property</th><th>Value</th></tr>
        <tr><td>Algorithm</td><td>Artificial Neural Network</td></tr>
        <tr><td>Input features</td><td>30</td></tr>
        <tr><td>Hidden layers</td><td>64 → 32 → 16</td></tr>
        <tr><td>Output</td><td>Sigmoid</td></tr>
        <tr><td>Test PR-AUC</td><td>0.751</td></tr>
        <tr><td>Decision threshold</td><td>{{ threshold }}</td></tr>
      </table>
      <p class="note">The metrics above are from the recorded evaluation run, not recalculated on each visit.</p>
    </div>
  </div>
</section>

<section id="analyzer" style="margin-top:30px">
  <h2 class="section-title">Transaction Analyzer</h2>
  <p class="subtitle">Enter all 30 model features. For meaningful analysis, use a real dataset row or valid feature extraction. Zero defaults are only placeholders.</p>

  {% if result %}
  <div class="result {% if is_fraud %}fraud{% endif %}">
    <span class="pill">ANALYSIS RESULT</span>
    <h3 style="margin-top:12px">{{ result }}</h3>
    <div class="score">Model output: {{ score }}%</div>
    <p class="note">Decision threshold: {{ threshold }}. This score is the model's output, not a guarantee of real-world fraud risk.</p>
  </div>
  {% endif %}

  {% if error %}
  <div class="result error"><strong>Input error:</strong> {{ error }}</div>
  {% endif %}

  <div class="panel">
    <form method="POST" action="#analyzer">
      <div class="form-grid">
      {% for feature in features %}
        <div>
          <label for="{{ feature }}">{{ feature }}</label>
          <input id="{{ feature }}" name="{{ feature }}" type="number"
                 step="any" required
                 value="{{ values.get(feature, '0') }}">
        </div>
      {% endfor %}
      </div>
      <button type="submit">Analyze Transaction</button>
    </form>
  </div>
  <p class="note" style="margin-top:14px">Educational prototype only. Do not use this application to approve, decline, or make real payment decisions.</p>
</section>
</main>
<footer>FraudGuard · Credit Card Fraud Detection · ANN Academic Project</footer>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    score = None
    error = None
    is_fraud = False
    values = {}

    if request.method == "POST":
        try:
            row = []
            for feature in FEATURES:
                raw = request.form.get(feature, "").strip()
                if not raw:
                    raise ValueError(f"{feature} is required.")
                value = float(raw)
                if not np.isfinite(value):
                    raise ValueError(f"{feature} must be a finite number.")
                row.append(value)
                values[feature] = raw

            transaction = pd.DataFrame([row], columns=FEATURES)
            scaled = scaler.transform(transaction)
            prediction = np.asarray(
                model.predict(scaled, verbose=0)
            ).reshape(-1)

            if prediction.size != 1:
                raise ValueError("Model did not return a single score.")

            probability = float(prediction[0])
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Model returned an invalid score.")

            score = round(probability * 100, 2)
            is_fraud = probability >= FRAUD_THRESHOLD
            result = (
                "Potential Fraud Detected"
                if is_fraud else "Likely Genuine Transaction"
            )
        except Exception as exc:
            error = str(exc)

    return render_template_string(
        HTML,
        features=FEATURES,
        values=values,
        result=result,
        score=score,
        error=error,
        is_fraud=is_fraud,
        threshold=f"{FRAUD_THRESHOLD:.2f}"
    )

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
        use_reloader=False
    )
