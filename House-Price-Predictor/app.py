"""
app.py
------
Streamlit dashboard for the House Price Predictor project.

Predicts residential house prices in Hyderabad, Telangana, India,
using a pre-trained Gradient Boosting / Random Forest / Linear
Regression model (whichever scored best during training).

Run:
    streamlit run app.py

Author: House Price Predictor Project
Python: 3.14.6
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from utils import (
    LOCATIONS,
    format_inr,
    get_locality_avg_price,
    get_locality_avg_price_per_sqft,
    investment_suggestion,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
CLEANED_DATA_PATH = os.path.join(BASE_DIR, "dataset", "cleaned_dataset.csv")
st.write("Current Working Directory:", os.getcwd())
st.write("Base Directory:", BASE_DIR)
st.write("Model Path:", MODEL_PATH)
st.write("Model Exists:", os.path.exists(MODEL_PATH))
st.write("Dataset Exists:", os.path.exists(CLEANED_DATA_PATH))

# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="House Price Predictor | Hyderabad",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.6rem;
            font-weight: 800;
            background: linear-gradient(90deg, #059669, #2563eb);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            font-size: 1.05rem;
            color: #64748b;
            margin-bottom: 1.5rem;
        }
        .result-card {
            background: linear-gradient(135deg, #059669 0%, #10b981 100%);
            padding: 1.6rem 1.8rem;
            border-radius: 16px;
            color: white;
            box-shadow: 0 8px 20px rgba(5, 150, 105, 0.25);
            margin-bottom: 1rem;
        }
        .result-card h2 {
            margin: 0;
            font-size: 2.1rem;
        }
        .result-card p {
            margin: 0.2rem 0 0 0;
            opacity: 0.9;
        }
        .info-card {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 0.8rem;
        }
        .badge {
            display: inline-block;
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.9rem;
        }
        .badge-good { background-color: #d1fae5; color: #065f46; }
        .badge-average { background-color: #fef3c7; color: #92400e; }
        .badge-premium { background-color: #dbeafe; color: #1e40af; }
        section[data-testid="stSidebar"] {
            background-color: #0f172a;
        }
        section[data-testid="stSidebar"] * {
            color: #e2e8f0 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# Cached loaders
# ------------------------------------------------------------------
@st.cache_resource
def load_model_bundle():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_cleaned_data():
    if os.path.exists(CLEANED_DATA_PATH):
        return pd.read_csv(CLEANED_DATA_PATH)
    return None


bundle = load_model_bundle()
df = load_cleaned_data()
st.write("Bundle:", bundle)
st.write("DataFrame:", df)


# ------------------------------------------------------------------
# Prediction function
# ------------------------------------------------------------------
def predict_price(bundle: dict, location: str, area: float, bhk: int,
                   bathrooms: int, parking: int, property_age: int,
                   floor_number: int, total_floors: int) -> float:
    """Build the feature vector exactly as done in train_model.py and predict."""
    model = bundle["model"]
    scaler = bundle["scaler"]
    label_encoder = bundle["label_encoder"]
    feature_columns = bundle["feature_columns"]
    onehot_columns = bundle["onehot_columns"]
    uses_scaled_input = bundle["uses_scaled_input"]

    floor_ratio = round(floor_number / max(total_floors, 1), 3)

    try:
        location_encoded = label_encoder.transform([location])[0]
    except ValueError:
        location_encoded = 0

    row = {
        "Area_SqFt": area,
        "BHK": bhk,
        "Bathrooms": bathrooms,
        "Parking": parking,
        "Property_Age": property_age,
        "Floor_Number": floor_number,
        "Total_Floors": total_floors,
        "Floor_Ratio": floor_ratio,
        "Location_Encoded": location_encoded,
    }
    for col in onehot_columns:
        row[col] = 1 if col == f"Loc_{location}" else 0

    X_input = pd.DataFrame([row])[feature_columns]

    if uses_scaled_input:
        X_input = scaler.transform(X_input)

    prediction = model.predict(X_input)[0]
    return max(0, prediction)


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏠 Project Info")
    st.markdown(
        "**House Price Predictor** estimates residential property "
        "prices across major localities in **Hyderabad, Telangana**, "
        "using a machine-learning regression model trained on "
        "property attributes such as area, BHK, floor, and locality."
    )

    st.markdown("---")
    st.markdown("## 📊 About Dataset")
    if df is not None:
        st.markdown(
            f"- **Rows:** {len(df):,}\n"
            f"- **Localities:** {df['Location'].nunique()}\n"
            f"- **Features:** {df.shape[1]}\n"
            f"- **Price range:** {format_inr(df['Price'].min())} – {format_inr(df['Price'].max())}"
        )
    else:
        st.warning("Dataset not found.")

    st.markdown("---")
    st.markdown("## 🎯 Model Accuracy")
    if bundle is not None:
        metrics = bundle["metrics"]
        st.markdown(f"**Model:** {bundle['model_name']}")
        st.metric("R² Score", f"{metrics['R2']*100:.2f}%")
        st.markdown(f"- MAE: {format_inr(metrics['MAE'])}\n- RMSE: {format_inr(metrics['RMSE'])}")
    else:
        st.warning("Model not found. Run `train_model.py` first.")

    st.markdown("---")
    st.markdown("## 👨‍💻 Developer Information")
    st.markdown(
        "**Project:** House Price Predictor\n\n"
        "**Tech Stack:** Python, Scikit-learn, Streamlit, Pandas\n\n"
        "**Python Version:** 3.14.6"
    )

    st.markdown("---")
    st.link_button("⭐ View on GitHub", "https://github.com/", use_container_width=True)


# ------------------------------------------------------------------
# Main page — Title
# ------------------------------------------------------------------
st.markdown('<div class="main-title">🏠 House Price Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-powered price estimation for residential properties '
    'in Hyderabad, Telangana — enter property details below to get an instant '
    'estimated market price.</div>',
    unsafe_allow_html=True,
)

if bundle is None or df is None:
    st.error(
        "⚠️ Model or dataset not found. Please run `python train_model.py` "
        "first to generate `model.pkl` and the cleaned dataset."
    )
    st.stop()

# Top metric row
m1, m2, m3, m4 = st.columns(4)
m1.metric("📦 Model", bundle["model_name"])
m2.metric("🎯 Accuracy (R²)", f"{bundle['metrics']['R2']*100:.1f}%")
m3.metric("🏘️ Localities Covered", df["Location"].nunique())
m4.metric("📊 Training Records", f"{len(df):,}")

st.markdown("---")

# ------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------
st.markdown("### 📝 Enter Property Details")

with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        location = st.selectbox("📍 Location", options=sorted(LOCATIONS), index=0)
        area = st.number_input("📐 Area (Square Feet)", min_value=200, max_value=15000,
                                value=1200, step=50)
        bhk = st.selectbox("🛏️ Bedrooms (BHK)", options=[1, 2, 3, 4, 5], index=1)

    with col2:
        bathrooms = st.selectbox("🚿 Bathrooms", options=[1, 2, 3, 4, 5, 6], index=1)
        parking = st.selectbox("🚗 Parking Spaces", options=[0, 1, 2, 3], index=1)
        property_age = st.selectbox(
            "🏗️ Property Age (years)",
            options=[0, 1, 2, 3, 5, 7, 10, 15, 20, 25], index=2,
        )

    with col3:
        total_floors = st.number_input("🏢 Total Floors in Building", min_value=1,
                                        max_value=50, value=10, step=1)
        floor_number = st.number_input("🔢 Floor Number", min_value=0,
                                        max_value=int(total_floors), value=min(3, int(total_floors)), step=1)

st.markdown("")
btn_col1, btn_col2, _ = st.columns([1, 1, 3])
predict_clicked = btn_col1.button("🔮 Predict Price", type="primary", use_container_width=True)
reset_clicked = btn_col2.button("🔄 Reset", use_container_width=True)

if reset_clicked:
    st.rerun()

# ------------------------------------------------------------------
# Prediction & Results
# ------------------------------------------------------------------
if predict_clicked:
    try:
        predicted_price = predict_price(
            bundle, location, area, bhk, bathrooms, parking,
            property_age, floor_number, total_floors,
        )
        price_per_sqft = predicted_price / area if area > 0 else 0
        locality_avg_price = get_locality_avg_price(df, location)
        locality_avg_psf = get_locality_avg_price_per_sqft(df, location)
        suggestion = investment_suggestion(predicted_price, locality_avg_price)

        st.markdown("### 💰 Prediction Result")

        res_col1, res_col2 = st.columns([2, 1])

        with res_col1:
            st.markdown(
                f"""
                <div class="result-card">
                    <p>Estimated Price</p>
                    <h2>{format_inr(predicted_price)}</h2>
                    <p>≈ {format_inr(price_per_sqft)} per sq. ft.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        badge_class = {
            "Good Investment": "badge-good",
            "Average Investment": "badge-average",
            "Premium Property": "badge-premium",
        }.get(suggestion, "badge-average")

        badge_emoji = {
            "Good Investment": "✅",
            "Average Investment": "➖",
            "Premium Property": "💎",
        }.get(suggestion, "➖")

        with res_col2:
            st.markdown(
                f"""
                <div class="info-card">
                    <p style="margin:0; color:#64748b; font-size:0.9rem;">Investment Suggestion</p>
                    <span class="badge {badge_class}">{badge_emoji} {suggestion}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="info-card">
                    <p style="margin:0; color:#64748b; font-size:0.9rem;">Model Confidence (R²)</p>
                    <h3 style="margin:0;">{bundle['metrics']['R2']*100:.1f}%</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### 📍 Locality Comparison")
        lc1, lc2, lc3 = st.columns(3)
        lc1.metric(f"Avg Price in {location}", format_inr(locality_avg_price))
        lc2.metric(f"Avg ₹/SqFt in {location}", format_inr(locality_avg_psf))
        diff_pct = ((predicted_price - locality_avg_price) / locality_avg_price * 100) if locality_avg_price else 0
        lc3.metric("Predicted vs Locality Avg", f"{diff_pct:+.1f}%")

        st.markdown("### ✅ Prediction Status")
        st.success("Prediction generated successfully using the trained ML model.")

    except Exception as e:
        st.error(f"⚠️ Something went wrong while predicting: {e}")

else:
    st.info("👆 Fill in the property details above and click **Predict Price** to see the estimate.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Scikit-learn — House Price Predictor Project")
