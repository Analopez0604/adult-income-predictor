import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# ── Configuración de la página ─────────────────────────────────────────
st.set_page_config(
    page_title="Predictor de Brecha Salarial",
    page_icon="💰",
    layout="centered"
)

# ── Carga y entrenamiento del modelo (cacheado) ────────────────────────
@st.cache_resource
def cargar_y_entrenar():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    columnas = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race',
        'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
        'native_country', 'income'
    ]
    df = pd.read_csv(url, names=columnas, skipinitialspace=True)
    df.replace('?', np.nan, inplace=True)

    # Capping
    for var in ['capital_gain', 'capital_loss', 'fnlwgt', 'hours_per_week']:
        Q1, Q3 = df[var].quantile(0.25), df[var].quantile(0.75)
        IQR = Q3 - Q1
        df[var] = df[var].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)

    # Imputación
    for col in ['occupation', 'workclass', 'native_country']:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Codificación
    encoders = {}
    cat_cols = ['workclass', 'education', 'marital_status', 'occupation',
                'relationship', 'race', 'sex', 'native_country']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df.drop(columns=['income'])
    y = (df['income'] == '>50K').astype(int) if 'income' in df.columns else None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modelo = LogisticRegression(max_iter=1000, random_state=42)
    modelo.fit(X_scaled, y)

    modelo_dt = DecisionTreeClassifier(max_depth=6, random_state=42)
    modelo_dt.fit(X_scaled, y)

    return modelo, modelo_dt, scaler, encoders, X.columns.tolist()

# ── UI ─────────────────────────────────────────────────────────────────
st.title("💰 Predictor de Brecha Salarial")
st.markdown(
    "**Análisis de la Brecha Salarial — Adult Income Dataset**  \n"
    "Ingresa las características de una persona para predecir si su ingreso anual supera los **50 000 USD**."
)
st.markdown("---")

with st.spinner("Cargando modelo... (solo la primera vez tarda un momento)"):
    modelo_lr, modelo_dt, scaler, encoders, feature_cols = cargar_y_entrenar()

st.success("✅ Modelo listo")

# ── Formulario de entrada ──────────────────────────────────────────────
st.subheader("📋 Datos de la Persona")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Edad", 17, 90, 35)
    education_num = st.slider("Años de educación", 1, 16, 10,
                               help="1=Sin estudios, 9=Secundaria, 13=Universidad, 16=Doctorado")
    hours_per_week = st.slider("Horas de trabajo por semana", 1, 99, 40)
    capital_gain = st.number_input("Ganancia de capital (USD)", 0, 100000, 0)
    capital_loss = st.number_input("Pérdida de capital (USD)", 0, 5000, 0)

with col2:
    sex = st.selectbox("Sexo", ["Male", "Female"])
    workclass = st.selectbox("Sector laboral", [
        "Private", "Self-emp-not-inc", "Self-emp-inc",
        "Federal-gov", "Local-gov", "State-gov", "Without-pay"
    ])
    education = st.selectbox("Nivel educativo", [
        "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
        "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th",
        "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"
    ])
    marital_status = st.selectbox("Estado civil", [
        "Married-civ-spouse", "Divorced", "Never-married",
        "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
    ])
    occupation = st.selectbox("Ocupación", [
        "Tech-support", "Craft-repair", "Other-service", "Sales",
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
        "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
        "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"
    ])

col3, col4 = st.columns(2)
with col3:
    relationship = st.selectbox("Relación familiar", [
        "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"
    ])
    race = st.selectbox("Raza", [
        "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"
    ])
with col4:
    native_country = st.selectbox("País de origen", [
        "United-States", "Cuba", "Jamaica", "India", "Mexico",
        "South", "Japan", "China", "Philippines", "Germany",
        "Canada", "Puerto-Rico", "El-Salvador", "France", "Other"
    ])
    modelo_sel = st.radio("Modelo de predicción", ["Regresión Logística", "Árbol de Decisión"])

# ── Predicción ─────────────────────────────────────────────────────────
if st.button("🔍 Predecir ingreso", type="primary"):

    # Construir fila de entrada
    entrada = {
        'age': age,
        'workclass': workclass,
        'fnlwgt': 189778,  # valor medio del dataset
        'education': education,
        'education_num': education_num,
        'marital_status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'sex': sex,
        'capital_gain': capital_gain,
        'capital_loss': capital_loss,
        'hours_per_week': hours_per_week,
        'native_country': native_country
    }

    df_entrada = pd.DataFrame([entrada])

    # Codificar categóricas
    cat_cols = ['workclass', 'education', 'marital_status', 'occupation',
                'relationship', 'race', 'sex', 'native_country']
    for col in cat_cols:
        le = encoders[col]
        val = df_entrada[col].iloc[0]
        if val in le.classes_:
            df_entrada[col] = le.transform([val])
        else:
            df_entrada[col] = 0  # fallback

    df_entrada = df_entrada[feature_cols]
    X_entrada = scaler.transform(df_entrada)

    # Seleccionar modelo
    clf = modelo_lr if modelo_sel == "Regresión Logística" else modelo_dt
    pred = clf.predict(X_entrada)[0]
    prob = clf.predict_proba(X_entrada)[0]

    st.markdown("---")
    st.subheader("📊 Resultado de la Predicción")

    if pred == 1:
        st.success(f"### ✅ Ingreso estimado: **> 50 000 USD / año**")
    else:
        st.warning(f"### ⚠️ Ingreso estimado: **≤ 50 000 USD / año**")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Probabilidad ≤50K", f"{prob[0]*100:.1f}%")
    with col_b:
        st.metric("Probabilidad >50K", f"{prob[1]*100:.1f}%")

    st.progress(float(prob[1]))
    st.caption(f"Modelo usado: {modelo_sel}")

# ── Footer ─────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Proyecto Final — Seminario de Ciencia de los Datos | "
    "Institución Universitaria Pascual Bravo | 2025  \n"
    "Dataset: Adult Income (UCI ML Repository) — Becker & Kohavi, 1996"
)
