# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and scaler
@st.cache_resource
def load_assets():
    model = joblib.load("fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_assets()

st.set_page_config(page_title="Credit Card Fraud Detection")
st.title("💳 Real-Time Credit Card Fraud Detection")

uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if 'Class' in df.columns:
        X = df.drop("Class", axis=1)
        y_true = df["Class"]
    else:
        X = df
        y_true = None

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    df["Predicted_Class"] = y_pred
    df["Fraud_Probability"] = np.round(y_proba, 2)

    st.subheader("🚨 Detected Fraudulent Transactions")
    st.write(df[df["Predicted_Class"] == 1].head(10))

    st.markdown(f"**Total Transactions:** {len(df)}")
    st.markdown(f"**Frauds Detected:** {df['Predicted_Class'].sum()}")

    if y_true is not None:
        from sklearn.metrics import classification_report
        report = classification_report(y_true, y_pred, digits=4)
        st.subheader("📊 Classification Report")
        st.code(report)

    # Chart: Predicted Class Distribution
    st.subheader("🔢 Prediction Summary Chart")
    class_counts = df["Predicted_Class"].value_counts().sort_index()
    fig1, ax1 = plt.subplots()
    sns.barplot(x=class_counts.index, y=class_counts.values, palette="Set2", ax=ax1)
    ax1.set_xticklabels(["Non-Fraud (0)", "Fraud (1)"])
    ax1.set_ylabel("Count")
    ax1.set_title("Predicted Class Distribution")
    st.pyplot(fig1)

    # Chart: Fraud Probability Distribution
    st.subheader("📉 Fraud Probability Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(df['Fraud_Probability'], bins=50, kde=True, ax=ax2)
    ax2.set_title("Fraud Probability Histogram")
    ax2.set_xlabel("Fraud Probability")
    ax2.set_ylabel("Frequency")
    st.pyplot(fig2)

    # Download button
    st.download_button(
        label="📥 Download Results",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="fraud_detection_results.csv",
        mime="text/csv"
    )
