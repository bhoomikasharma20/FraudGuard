
import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# -------------------- Page setup --------------------
st.set_page_config(
    page_title="FraudGuard | ANN Detection",
    page_icon="🛡️",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "credit_fraud_ann.keras")
SCALER_PATH = os.path.join(BASE_DIR, "credit_fraud_scaler.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "fraud_threshold.pkl")

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


# -------------------- Load trained assets --------------------
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model(
        MODEL_PATH, compile=False
    )

    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)

    with open(THRESHOLD_PATH, "rb") as file:
        threshold = float(
            np.asarray(pickle.load(file)).reshape(-1)[0]
        )

    return model, scaler, threshold


# -------------------- Header --------------------
st.title("🛡️ FraudGuard")
st.subheader(
    "Credit Card Fraud Detection using Artificial Neural Network"
)

st.write(
    "Analyze transaction features using a trained ANN model "
    "and identify transactions that may require further review."
)

st.info(
    "Academic demonstration only. The result is a model score, "
    "not a guarantee that a transaction is fraudulent or genuine."
)


# -------------------- Load files --------------------
try:
    model, scaler, threshold = load_assets()
except Exception as error:
    st.error(
        "The model could not be loaded. Check that all three model "
        "files are present in the GitHub repository and compatible "
        "with requirements.txt."
    )
    st.code(str(error))
    st.stop()


# -------------------- Dashboard tabs --------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🏠 Overview",
        "🔍 Analyze Transaction",
        "📂 Bulk CSV Testing",
        "ℹ️ Model Information",
    ]
)


# -------------------- Tab 1: Overview --------------------
with tab1:
    st.markdown("### Project Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Model", "ANN")
    col2.metric("Input Features", "30")
    col3.metric("Decision Threshold", f"{threshold:.2f}")

    st.markdown("### How it works")
    st.markdown(
        """
        1. Enter the 30 transaction features or upload a CSV.
        2. FraudGuard scales features using the saved scaler.
        3. The trained neural network generates a fraud score.
        4. The score is compared with the decision threshold.
        """
    )

    st.caption(
        "Dataset: Credit Card Fraud Detection dataset. "
        "V1–V28 are anonymized numerical features."
    )


# -------------------- Tab 2: Single Transaction --------------------
with tab2:
    st.markdown("### Transaction Analyzer")

    st.write(
        "Enter values in the same feature order and units used "
        "during training. Default zeros are placeholders."
    )

    with st.form("transaction_form"):
        st.markdown("#### Transaction features")
        values = {}

        columns = st.columns(3)

        for index, feature in enumerate(FEATURES):
            with columns[index % 3]:
                values[feature] = st.number_input(
                    feature,
                    value=0.0,
                    format="%.8f",
                    key=f"feature_{feature}",
                    help=(
                        "Enter the original dataset value. "
                        "V1–V28 are transformed/anonymized features."
                    ),
                )

        submitted = st.form_submit_button(
            "🔎 Analyze Transaction",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        try:
            row = np.array(
                [[values[feature] for feature in FEATURES]],
                dtype=np.float64,
            )

            scaled_row = scaler.transform(row)

            score = float(
                np.asarray(
                    model.predict(scaled_row, verbose=0)
                ).reshape(-1)[0]
            )

            st.markdown("---")
            st.markdown("### Analysis Result")

            result_col, score_col = st.columns(2)

            with result_col:
                if score >= threshold:
                    st.error("🚨 Potential Fraud Detected")
                    st.write(
                        "The model score meets or exceeds "
                        "the selected fraud decision threshold."
                    )
                else:
                    st.success("✅ Likely Genuine Transaction")
                    st.write(
                        "The model score is below the "
                        "selected fraud decision threshold."
                    )

            with score_col:
                st.metric("ANN Fraud Score", f"{score:.2%}")
                st.progress(max(0.0, min(1.0, score)))
                st.caption(f"Decision threshold: {threshold:.2%}")

            st.caption(
                "The score is the model's sigmoid output and may "
                "not be a calibrated real-world probability."
            )

        except Exception as error:
            st.error("Prediction failed. Please check the input values.")
            st.code(str(error))


# -------------------- Tab 3: Bulk CSV Testing --------------------
with tab3:
    st.markdown("### Bulk Transaction Testing")

    st.write(
        "Upload a CSV containing the 30 transaction features. "
        "FraudGuard will analyze all rows automatically."
    )

    uploaded_file = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"],
        key="bulk_csv",
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            missing = [
                feature for feature in FEATURES
                if feature not in df.columns
            ]

            if missing:
                st.error(
                    "Required columns are missing: "
                    + ", ".join(missing)
                )
            else:
                X = df[FEATURES].apply(
                    pd.to_numeric,
                    errors="coerce",
                )

                if X.isna().any().any():
                    st.error(
                        "Some feature values are missing or invalid. "
                        "Please check the CSV file."
                    )
                else:
                    X_scaled = scaler.transform(X)

                    scores = np.asarray(
                        model.predict(X_scaled, verbose=0)
                    ).reshape(-1)

                    results = df.copy()

                    results["Fraud Score (%)"] = (
                        scores * 100
                    ).round(2)

                    results["Prediction"] = np.where(
                        scores >= threshold,
                        "Potential Fraud",
                        "Likely Genuine",
                    )

                    st.success(
                        f"Successfully analyzed {len(results)} transactions."
                    )

                    st.dataframe(
                        results,
                        use_container_width=True,
                    )

                    fraud_count = int(
                        np.sum(scores >= threshold)
                    )
                    genuine_count = len(scores) - fraud_count

                    col1, col2 = st.columns(2)
                    col1.metric(
                        "Potential Fraud",
                        fraud_count,
                    )
                    col2.metric(
                        "Likely Genuine",
                        genuine_count,
                    )

                    csv_output = results.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        "⬇️ Download Results CSV",
                        data=csv_output,
                        file_name="fraudguard_results.csv",
                        mime="text/csv",
                    )

                    if "Class" in df.columns:
                        st.info(
                            "The uploaded Class column is retained "
                            "for comparison only; it is not used "
                            "as a model input."
                        )

        except Exception as error:
            st.error("Could not analyze this CSV file.")
            st.code(str(error))


# -------------------- Tab 4: Model Information --------------------
with tab4:
    st.markdown("### Model Information")

    st.write("Algorithm: Artificial Neural Network (ANN)")
    st.write("Hidden layers: Dense 64 → Dense 32 → Dense 16")
    st.write("Regularization: Dropout layers")
    st.write("Output: Sigmoid score for binary classification")
    st.write(f"Decision threshold: {threshold:.4f}")

    st.markdown("### Held-out test evaluation")

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Accuracy", "99.92%")
    metric2.metric("Fraud Recall", "81.08%")
    metric3.metric("Fraud F1-score", "76.92%")

    st.caption(
        "Reported results are from the project's held-out test set. "
        "Performance on new transactions may differ."
    )

