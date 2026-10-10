import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Student Performance AI", page_icon="🎓", layout="wide")

@st.cache_data(show_spinner="Loading the public UCI dataset...")
def load_data():
    dataset = fetch_ucirepo(id=320)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    df = pd.concat([features, targets], axis=1)
    needed = ["studytime", "absences", "G1", "G2", "G3"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            "The dataset columns returned by UCI did not include: "
            + ", ".join(missing)
            + ". Check the UCI dataset or package version."
        )
    df = df[needed].apply(pd.to_numeric, errors="coerce").dropna()
    return df

@st.cache_resource(show_spinner="Training and evaluating models...")
def train_models():
    df = load_data()
    X = df[["G1", "G2", "studytime", "absences"]]
    y = df["G3"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    models = {
        "Baseline (mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression()),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=2)
        )
    }
    trained = {}
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        trained[name] = model
        rows.append({
            "Model": name,
            "MAE (lower is better)": mean_absolute_error(y_test, pred),
            "RMSE (lower is better)": mean_squared_error(y_test, pred) ** 0.5,
            "R² (closer to 1 is better)": r2_score(y_test, pred)
        })
    metrics = pd.DataFrame(rows).sort_values("MAE (lower is better)")
    return df, trained, metrics, X_test, y_test

st.title("🎓 AI-Based Student Performance Prediction")
st.write("A learning prototype that estimates a student's final grade from earlier grades and selected study-related factors.")
st.info("Educational use only. Predictions are estimates based on a historical Portuguese-school dataset; they are not guarantees and should not be used for high-stakes decisions.")

try:
    df, models, metrics, X_test, y_test = train_models()
except Exception as exc:
    st.error("The app could not load the UCI dataset. Check your internet connection and try again.")
    st.code(str(exc))
    st.markdown("Dataset source: [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)")
    st.stop()

with st.sidebar:
    st.header("Student inputs")
    g1 = st.slider("First-period grade (G1)", 0, 20, 10)
    g2 = st.slider("Second-period grade (G2)", 0, 20, 11)
    studytime = st.select_slider(
        "Weekly study-time category",
        options=[1, 2, 3, 4],
        value=2,
        format_func=lambda x: {1: "<2 hours", 2: "2–5 hours", 3: "5–10 hours", 4: ">10 hours"}[x]
    )
    absences = st.number_input("School absences", min_value=0, max_value=100, value=3, step=1)
    model_name = st.selectbox("Prediction model", ["Linear Regression", "Random Forest"])

input_df = pd.DataFrame([{
    "G1": g1, "G2": g2, "studytime": studytime, "absences": absences
}])
predicted = float(models[model_name].predict(input_df)[0])
predicted = min(20.0, max(0.0, predicted))

left, right = st.columns([1, 1])
with left:
    st.subheader("Estimated final grade")
    st.metric("Predicted G3 (out of 20)", f"{predicted:.1f}/20")
    st.caption("The estimate uses G1, G2, study-time category and absences.")
with right:
    st.subheader("General study suggestions")
    suggestions = []
    if g1 < 10 or g2 < 10:
        suggestions.append("Review topics where earlier assessments were difficult; ask a teacher for feedback.")
    if studytime <= 2:
        suggestions.append("Try a consistent weekly revision plan with short, focused sessions.")
    if absences >= 5:
        suggestions.append("Catch up on missed lessons and notes, and clarify gaps with a teacher.")
    if not suggestions:
        suggestions.append("Maintain a consistent routine and use practice questions to check understanding.")
    for item in suggestions:
        st.write("• " + item)
    st.caption("Suggestions are rule-based prompts, not medical, psychological or official academic advice.")

st.divider()
st.subheader("Model evaluation")
st.write("A fixed 80/20 train-test split (random_state=42) is used for a reproducible demonstration. Compare models using the test set; do not report metrics before running the app.")
st.dataframe(metrics.round(3), use_container_width=True, hide_index=True)
st.caption("MAE and RMSE are measured in grade points. R² can be negative if a model performs worse than predicting the test-set mean.")

best_model = metrics.iloc[0]["Model"]
if best_model == "Baseline (mean)":
    st.warning("The baseline performed best on this split. Do not claim the ML models outperform the baseline; investigate features and validation before drawing conclusions.")
else:
    st.success(f"Lowest test MAE on this run: {best_model}. This is a single hold-out split, not proof of general performance.")

st.subheader("Explore the dataset")
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    ax.hist(df["G3"], bins=11)
    ax.set_xlabel("Final grade (G3)")
    ax.set_ylabel("Number of students")
    ax.set_title("Distribution of final grades")
    st.pyplot(fig)
    plt.close(fig)
with col2:
    fig, ax = plt.subplots()
    ax.scatter(df["G2"], df["G3"], alpha=0.6)
    ax.set_xlabel("Second-period grade (G2)")
    ax.set_ylabel("Final grade (G3)")
    ax.set_title("Earlier grade vs final grade")
    st.pyplot(fig)
    plt.close(fig)

with st.expander("Dataset and limitations"):
    st.write(
        "The public UCI dataset contains records from two Portuguese secondary schools. "
        "This demo uses four selected inputs and predicts G3. G1 and G2 are strongly associated "
        "with G3, so the result is more suitable for a late-year estimate than a prediction made "
        "before any assessments. The dataset may not represent students in India or other settings. "
        "The model may make errors and should not determine grades, admissions, support eligibility, "
        "or other consequential decisions."
    )
    st.markdown("Reference: [Cortez (2008), UCI Student Performance dataset](https://doi.org/10.24432/C5TG7T)")
