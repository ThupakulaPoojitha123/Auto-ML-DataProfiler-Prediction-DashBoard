import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, confusion_matrix

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="AutoML Data Profiler", layout="wide")
st.title("🤖 AutoML Data Profiler & Prediction Dashboard")

# -----------------------------
# File Upload
# -----------------------------
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("✅ File uploaded successfully!")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # -----------------------------
    # Data Profiling
    # -----------------------------
    st.header("🔍 Data Profiling")
    st.write("**Dataset Shape:**", df.shape)
    st.write("**Column Data Types:**")
    st.write(df.dtypes)
    st.write("**Missing Values:**")
    st.write(df.isnull().sum())
    st.write("**Basic Statistics:**")
    st.write(df.describe(include="all"))

    # Correlation heatmap (numeric only)
    numeric_df = df.select_dtypes(include=np.number)
    if not numeric_df.empty:
        st.subheader("📈 Correlation Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(numeric_df.corr(), annot=True, cmap="Blues", ax=ax)
        st.pyplot(fig)

    # -----------------------------
    # Target Selection
    # -----------------------------
    st.header("🎯 Target Selection")
    target_col = st.selectbox("Select the target column:", df.columns)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Determine problem type
    problem_type = "Regression" if pd.api.types.is_numeric_dtype(y) else "Classification"
    st.info(f"Detected Problem Type: **{problem_type}**")

    # Handle categorical encoding
    X = pd.get_dummies(X, drop_first=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # -----------------------------
    # AutoML Model Training
    # -----------------------------
    st.header("⚙️ Model Training & Evaluation")

    results = {}

    if problem_type == "Regression":
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(random_state=42)
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            results[name] = {"RMSE": rmse, "R²": r2}

        results_df = pd.DataFrame(results).T
        st.dataframe(results_df)

        best_model_name = results_df["R²"].idxmax()
        best_model = models[best_model_name]
        st.success(f"🏆 Best Model: {best_model_name}")

        # Predictions Visualization
        preds = best_model.predict(X_test)
        scatter_fig = px.scatter(x=y_test, y=preds, labels={'x': 'Actual', 'y': 'Predicted'},
                                 title="Predicted vs Actual")
        st.plotly_chart(scatter_fig, use_container_width=True)

    else:  # Classification
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest Classifier": RandomForestClassifier(random_state=42)
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average='weighted', zero_division=0)
            rec = recall_score(y_test, preds, average='weighted', zero_division=0)
            results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec}

        results_df = pd.DataFrame(results).T
        st.dataframe(results_df)

        best_model_name = results_df["Accuracy"].idxmax()
        best_model = models[best_model_name]
        st.success(f"🏆 Best Model: {best_model_name}")

        # Confusion Matrix
        preds = best_model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        cm_fig = px.imshow(cm, text_auto=True, title="Confusion Matrix",
                           labels=dict(x="Predicted", y="Actual"))
        st.plotly_chart(cm_fig, use_container_width=True)

    # -----------------------------
    # Predictions & Download
    # -----------------------------
    st.header("📥 Predictions & Download")
    predictions = best_model.predict(X_test)
    results_output = X_test.copy()
    results_output["Actual"] = y_test.values
    results_output["Predicted"] = predictions

    st.dataframe(results_output.head())
    csv_data = results_output.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Predictions", csv_data, file_name="predictions.csv", mime="text/csv")

else:
    st.info("👈 Please upload a dataset to begin.")