"""
app.py
Streamlit app: interactive house price prediction using the trained
Linear Regression and Random Forest models, plus a model-performance
dashboard built from the saved evaluation artifacts.

Run with:  streamlit run app.py
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import full_preprocess
from generate_data import (
    LOCATIONS, PROPERTY_TYPES, FURNISHING, STREET_TYPES, CONDITIONS
)

BASE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE, "models")

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")


@st.cache_resource
def load_artifacts():
    lin_model = joblib.load(os.path.join(MODELS_DIR, "linear_model.pkl"))
    rf_model = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    scale_cols = joblib.load(os.path.join(MODELS_DIR, "scale_cols.pkl"))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
    with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    importance = pd.read_csv(os.path.join(MODELS_DIR, "feature_importance.csv"), index_col=0)
    importance.columns = ["Importance"]
    preds = pd.read_csv(os.path.join(MODELS_DIR, "test_predictions.csv"))
    return lin_model, rf_model, scaler, scale_cols, feature_columns, metrics, importance, preds


lin_model, rf_model, scaler, scale_cols, feature_columns, metrics, importance, preds = load_artifacts()


MIN_REALISTIC_PRICE = 220000  # matches the training data's price floor


def predict_price(inputs: dict):
    raw = pd.DataFrame([inputs])
    X, _ = full_preprocess(raw, is_training=False, fit_columns=feature_columns)
    X_scaled = X.copy()
    X_scaled[scale_cols] = scaler.transform(X[scale_cols])
    lin_pred_raw = float(lin_model.predict(X_scaled)[0])
    rf_pred_raw = float(rf_model.predict(X_scaled)[0])
    lin_pred = max(lin_pred_raw, MIN_REALISTIC_PRICE)
    rf_pred = max(rf_pred_raw, MIN_REALISTIC_PRICE)
    lin_clamped = lin_pred_raw < MIN_REALISTIC_PRICE
    rf_clamped = rf_pred_raw < MIN_REALISTIC_PRICE
    return lin_pred, rf_pred, lin_clamped, rf_clamped


st.title("🏠 House Price Prediction")
st.caption("Linear Regression vs. Random Forest — trained on a 1,200-record synthetic residential dataset")

tab_predict, tab_performance, tab_about = st.tabs(["🔮 Predict", "📊 Model Performance", "ℹ️ About"])

# ----------------------------------------------------------------------
with tab_predict:
    st.subheader("Enter property details")

    col1, col2, col3 = st.columns(3)
    with col1:
        area = st.slider("Area (sq. ft.)", 700, 8000, 2200, step=50)
        bedrooms = st.slider("Bedrooms", 1, 6, 3)
        bathrooms = st.slider("Bathrooms", 1, 5, 2)
        rooms = st.slider("Total Rooms", 2, 7, 4)

    with col2:
        build_year = st.slider("Build Year", 1985, 2024, 2015)
        lot_size = st.slider("Lot Size (sq. ft.)", 800, 15000, 3000, step=100)
        distance = st.slider("Distance to City Center (km)", 0.3, 35.0, 8.0, step=0.5)
        condition = st.selectbox("Property Condition", CONDITIONS, index=2)

    with col3:
        location = st.selectbox("Location", LOCATIONS)
        property_type = st.selectbox("Property Type", PROPERTY_TYPES)
        furnishing = st.selectbox("Furnishing", FURNISHING)
        street_type = st.selectbox("Street Type", STREET_TYPES)
        has_pool = st.checkbox("Has Swimming Pool", value=False)

    st.markdown("---")

    if st.button("Predict Price", type="primary", use_container_width=True):
        inputs = {
            "Area_SqFt": area,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Rooms": rooms,
            "Build_Year": build_year,
            "Lot_Size": lot_size,
            "Distance_to_Center": distance,
            "Furnishing": furnishing,
            "Property_Type": property_type,
            "Location": location,
            "Street_Type": street_type,
            "Has_Pool": has_pool,
            "Property_Condition": condition,
        }
        lin_pred, rf_pred, lin_clamped, rf_clamped = predict_price(inputs)
        avg_pred = (lin_pred + rf_pred) / 2

        c1, c2, c3 = st.columns(3)
        c1.metric("Linear Regression", f"₹{lin_pred:,.0f}")
        c2.metric("Random Forest", f"₹{rf_pred:,.0f}")
        c3.metric("Average Estimate", f"₹{avg_pred:,.0f}")

        if lin_clamped or rf_clamped:
            which = "Linear Regression" if lin_clamped and not rf_clamped else (
                "Random Forest" if rf_clamped and not lin_clamped else "Both models"
            )
            st.info(
                f"{which} extrapolated below a realistic price for this feature combination "
                "and was floored at the dataset's observed minimum. This tends to happen for "
                "very small, very old, or very far-from-center properties — combinations the "
                "model saw little of during training."
            )

        spread = abs(lin_pred - rf_pred)
        spread_pct = spread / avg_pred * 100
        if spread_pct > 15:
            st.warning(
                f"The two models disagree by {spread_pct:.1f}% (₹{spread:,.0f}). "
                "This usually happens for unusual feature combinations the models saw less of in training."
            )
        else:
            st.success(f"Both models agree within {spread_pct:.1f}% of each other.")

# ----------------------------------------------------------------------
with tab_performance:
    st.subheader("Test-set performance")

    m1, m2 = st.columns(2)
    for col, name in zip([m1, m2], ["Linear Regression", "Random Forest"]):
        with col:
            st.markdown(f"**{name}**")
            d = metrics[name]
            st.metric("R² Score", f"{d['R2']:.3f}")
            st.metric("MAE", f"₹{d['MAE']:,.0f}")
            st.metric("RMSE", f"₹{d['RMSE']:,.0f}")
            st.caption(f"5-fold CV R²: {d['CV_R2_mean']:.3f} ± {d['CV_R2_std']:.3f}")

    st.markdown("---")
    st.subheader("Predicted vs. Actual (test set)")
    chart_choice = st.radio("Model", ["Linear Regression", "Random Forest"], horizontal=True)
    pred_col = "Linear_Pred" if chart_choice == "Linear Regression" else "RF_Pred"
    chart_df = preds[["Actual", pred_col]].rename(columns={pred_col: "Predicted"})
    st.scatter_chart(chart_df, x="Actual", y="Predicted")

    st.markdown("---")
    st.subheader("Random Forest — Feature Importance (Top 10)")
    st.bar_chart(importance.head(10))

# ----------------------------------------------------------------------
with tab_about:
    st.subheader("About this project")
    st.markdown(
        """
This app predicts residential property prices using two models trained on a
1,200-record synthetic dataset spanning eight Indian cities:

- **Linear Regression** — interpretable baseline, OLS with standardized features.
- **Random Forest Regressor** — `n_estimators=100, max_depth=15, min_samples_split=5,
  min_samples_leaf=2`, trained as an ensemble of decision trees.

**Pipeline:** mean/mode imputation for missing values → IQR-based outlier
capping on Price, Area, and Lot Size → one-hot encoding for nominal fields
(Location, Property Type, Furnishing, Street Type) → ordinal encoding for
Property Condition → 6 engineered features (Property Age, Bath-to-Bed Ratio,
Lot per Bedroom, Total Living Units Score, Location Quality Index, Area per
Room) → StandardScaler on continuous fields → 80/20 train-test split.

**A note on methodology:** the original project report included a
"Price per Square Foot" engineered feature, which divides the target
(Price) by Area — this leaks the target into the inputs and inflates
accuracy artificially. This implementation replaces it with **Area per
Room**, which captures spaciousness without touching the target, so the
R² scores you see here are genuine out-of-sample performance.
        """
    )
