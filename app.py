import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
from pathlib import Path

st.set_page_config(page_title="HSE ML Homework 1", page_icon="🎯", layout="wide")

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "linear_regression.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

NUMERIC_COLS = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']

@st.cache_resource
def load_model():
    """Загружаем модель через pickle"""

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(FEATURE_NAMES_PATH, 'rb') as f:
        feature_names = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return model, feature_names, scaler

model, feature_names, scaler = load_model()

st.title("HSE ML Homework 1")

st.markdown("---")
file_uploaded = st.file_uploader("Загрузите файл", type=["csv", "xlsx"])

if file_uploaded is not None:
    df = pd.read_csv(file_uploaded)
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Объектов", len(df))
    with col2:
        st.metric("Признаков", df.shape[1])
    with col3:
        st.metric("Дубликаты", df.duplicated().sum())
    with col4:
        st.metric("Пропуски", df.isnull().sum().sum())

    st.markdown("---")
    st.subheader("EDA")


    fig = px.histogram(df, x='selling_price', nbins=50, title="Распределение selling_price")
    st.plotly_chart(fig, use_container_width=True)

    numeric_df = df.select_dtypes(include=['number'])
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', title="Корреляция")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.box(df, x='transmission', y='selling_price', color='transmission', 
                    title="selling_price by transmission")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Загрузите CSV")

fuel_options = ["Diesel", "Petrol", "LPG", "CNG"]
seller_type_options = ["Individual", "Dealer", "Trustmark Dealer"]
owner_options = ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"]
transmission_options = ["Manual", "Automatic"]
seats_options = [5, 4, 7, 8, 6, 9, 10, 14, 2]

st.markdown("---")
st.subheader("Предсказание цены")

# Чуть не успеваю по времени, поэтому настройку ползунков сделал через LLM
with st.form("predict_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        year = st.slider("Год выпуска", 1990, 2024, 2017, 1)
        km_driven = st.slider("Пробег (км)", 0, 500000, 50000, 1000)

    with c2:
        mileage = st.slider("Расход (km/l)", 5.0, 50.0, 18.0, 0.1)
        engine = st.slider("Объём двигателя (CC)", 500, 5000, 1200, 50)

    with c3:
        max_power = st.slider("Мощность (bhp)", 30.0, 400.0, 80.0, 1.0)

    c4, c5, c6 = st.columns(3)

    with c4:
        fuel = st.selectbox("fuel", fuel_options)
        transmission = st.selectbox("transmission", transmission_options)

    with c5:
        seller_type = st.selectbox("seller_type", seller_type_options)
        owner = st.selectbox("owner", owner_options)

    with c6:
        seats = st.selectbox("seats", seats_options)

    submitted = st.form_submit_button("Предсказать")

if submitted:
    numeric_row = {
        "year": year,
        "km_driven": km_driven,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats,
    }
    X_numeric = pd.DataFrame([numeric_row], columns=NUMERIC_COLS)
    
    X_scaled = pd.DataFrame(scaler.transform(X_numeric), columns=NUMERIC_COLS)

    row = {c: 0.0 for c in feature_names}
    
    for c in ["year", "km_driven", "mileage", "engine", "max_power"]:
        if c in row:
            row[c] = float(X_scaled[c].iloc[0])

    def set_ohe(prefix, value):
        col = f"{prefix}_{value}"
        if col in row:
            row[col] = 1.0

    set_ohe("fuel", fuel)
    set_ohe("seller_type", seller_type)
    set_ohe("transmission", transmission)
    set_ohe("owner", owner)
    set_ohe("seats", seats)

    X_one = pd.DataFrame([row], columns=feature_names)
    pred = float(model.predict(X_one)[0])

    st.success(f"Предсказанная цена: {pred:,.0f}".replace(",", " "))


st.markdown("---")
st.subheader("Коэффициенты модели")

coefs_abs = pd.Series(np.asarray(model.coef_).ravel(), index=feature_names).abs().sort_values(ascending=False)

fig = px.bar(
    coefs_abs.iloc[::-1].reset_index(),
    x=0,
    y="index",
    orientation="h",
    labels={"index": "feature", "0": "coef"},
    height=max(600, 18 * len(coefs_abs)),
)
st.plotly_chart(fig, use_container_width=True)

