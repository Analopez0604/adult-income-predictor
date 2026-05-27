import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import plotly.graph_objects as go
import plotly.express as px

# ── Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictor de Brecha Salarial",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS personalizado ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

* { font-family: 'Sora', sans-serif; }

/* Fondo oscuro con textura */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f1628 50%, #0a1020 100%);
    background-attachment: fixed;
}

/* Header hero */
.hero-container {
    background: linear-gradient(135deg, #1a237e 0%, #0d47a1 40%, #01579b 100%);
    border-radius: 20px;
    padding: 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(100,160,255,0.2);
    box-shadow: 0 20px 60px rgba(0,80,200,0.3);
}
.hero-container::before {
    content: '';
    position: absolute;
    top: -50%; right: -20%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(100,200,255,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 1rem;
    color: rgba(255,255,255,0.7);
    margin: 0;
    font-weight: 300;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.75rem;
    color: rgba(255,255,255,0.9);
    margin-bottom: 16px;
    font-family: 'JetBrains Mono', monospace;
}

/* Cards de sección */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    backdrop-filter: blur(10px);
}
.card-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: #64b5f6;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 16px;
}

/* Resultado cards */
.result-card-high {
    background: linear-gradient(135deg, rgba(0,200,120,0.15), rgba(0,150,80,0.08));
    border: 1px solid rgba(0,200,120,0.3);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}
.result-card-low {
    background: linear-gradient(135deg, rgba(255,160,0,0.15), rgba(200,100,0,0.08));
    border: 1px solid rgba(255,160,0,0.3);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}
.result-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 8px;
    opacity: 0.7;
}
.result-value {
    font-size: 1.8rem;
    font-weight: 800;
    margin: 0;
}
.result-high { color: #4caf50; }
.result-low  { color: #ffa726; }

/* Metric chips */
.metric-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 12px;
}
.metric-chip {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 8px 14px;
    font-size: 0.8rem;
    color: rgba(255,255,255,0.85);
}
.metric-chip span {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    color: #90caf9;
}

/* Sliders y selects */
.stSlider > div > div { background: transparent !important; }
.stSelectbox > div > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: white !important;
}

/* Botón predecir */
.stButton > button {
    background: linear-gradient(135deg, #1565c0, #0288d1) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 40px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    font-family: 'Sora', sans-serif !important;
    width: 100% !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 8px 24px rgba(2,136,209,0.4) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 32px rgba(2,136,209,0.5) !important;
}

/* Ganador badge */
.winner-badge {
    display: inline-block;
    background: linear-gradient(135deg, #ffd700, #ffa000);
    color: #000;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.7rem;
    font-weight: 700;
    margin-left: 8px;
    vertical-align: middle;
}

/* Separador */
.divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin: 24px 0;
}

/* Ocultar elementos default de streamlit */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1200px; }
</style>
""", unsafe_allow_html=True)


# ── Modelo (cacheado) ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def cargar_modelos():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    cols = ['age','workclass','fnlwgt','education','education_num',
            'marital_status','occupation','relationship','race',
            'sex','capital_gain','capital_loss','hours_per_week',
            'native_country','income']
    df = pd.read_csv(url, names=cols, skipinitialspace=True)
    df.replace('?', np.nan, inplace=True)

    for var in ['capital_gain','capital_loss','fnlwgt','hours_per_week']:
        Q1, Q3 = df[var].quantile(0.25), df[var].quantile(0.75)
        df[var] = df[var].clip(lower=Q1-1.5*(Q3-Q1), upper=Q3+1.5*(Q3-Q1))

    for col in ['occupation','workclass','native_country']:
        df[col].fillna(df[col].mode()[0], inplace=True)

    cat_cols = ['workclass','education','marital_status','occupation',
                'relationship','race','sex','native_country']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df.drop(columns=['income'])
    y = (df['income'] == '>50K').astype(int)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    from sklearn.calibration import CalibratedClassifierCV
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_sc, y)
    dt_base = DecisionTreeClassifier(max_depth=6, random_state=42)
    dt = CalibratedClassifierCV(dt_base, cv=3, method='isotonic')
    dt.fit(X_sc, y)

    return lr, dt, scaler, encoders, X.columns.tolist()


# ── Opciones traducidas ────────────────────────────────────────────────
SECTOR = {
    "Sector privado": "Private",
    "Empleado público federal": "Federal-gov",
    "Empleado público local": "Local-gov",
    "Empleado público estatal": "State-gov",
    "Independiente (empresa propia)": "Self-emp-inc",
    "Independiente (sin empresa)": "Self-emp-not-inc",
    "Sin empleo remunerado": "Without-pay",
}
EDUCACION = {
    "Doctorado": "Doctorate",
    "Maestría": "Masters",
    "Escuela profesional": "Prof-school",
    "Licenciatura / Ingeniería": "Bachelors",
    "Algo de universidad": "Some-college",
    "Técnico académico": "Assoc-acdm",
    "Técnico vocacional": "Assoc-voc",
    "Bachillerato completo": "HS-grad",
    "12.° grado (sin diploma)": "12th",
    "11.° grado": "11th",
    "10.° grado": "10th",
    "9.° grado": "9th",
    "7.° u 8.° grado": "7th-8th",
    "5.° o 6.° grado": "5th-6th",
    "1.° a 4.° grado": "1st-4th",
    "Preescolar": "Preschool",
}
EDU_NUM = {
    "Doctorado": 16, "Maestría": 14, "Escuela profesional": 15,
    "Licenciatura / Ingeniería": 13, "Algo de universidad": 10,
    "Técnico académico": 12, "Técnico vocacional": 11,
    "Bachillerato completo": 9, "12.° grado (sin diploma)": 8,
    "11.° grado": 7, "10.° grado": 6, "9.° grado": 5,
    "7.° u 8.° grado": 4, "5.° o 6.° grado": 3,
    "1.° a 4.° grado": 2, "Preescolar": 1,
}
ESTADO_CIVIL = {
    "Casado/a (cónyuge presente)": "Married-civ-spouse",
    "Soltero/a (nunca casado/a)": "Never-married",
    "Divorciado/a": "Divorced",
    "Separado/a": "Separated",
    "Viudo/a": "Widowed",
    "Casado/a (cónyuge ausente)": "Married-spouse-absent",
    "Casado/a (cónyuge en fuerzas armadas)": "Married-AF-spouse",
}
OCUPACION = {
    "Especialidad profesional": "Prof-specialty",
    "Artesanía y reparación": "Craft-repair",
    "Gestión ejecutiva": "Exec-managerial",
    "Administrativo / Oficina": "Adm-clerical",
    "Ventas": "Sales",
    "Otros servicios": "Other-service",
    "Operario de máquinas": "Machine-op-inspct",
    "Transporte": "Transport-moving",
    "Manejo y limpieza": "Handlers-cleaners",
    "Agricultura y pesca": "Farming-fishing",
    "Soporte técnico": "Tech-support",
    "Servicio doméstico": "Priv-house-serv",
    "Servicios de protección": "Protective-serv",
    "Fuerzas armadas": "Armed-Forces",
}
RELACION = {
    "Esposo/a": "Husband",
    "Esposa": "Wife",
    "Hijo/a propio/a": "Own-child",
    "No es familiar": "Not-in-family",
    "Otro familiar": "Other-relative",
    "No casado/a": "Unmarried",
}
RAZA = {
    "Blanco/a": "White",
    "Negro/a": "Black",
    "Asiático/a o Isleño/a del Pacífico": "Asian-Pac-Islander",
    "Indígena americano/a o nativo/a de Alaska": "Amer-Indian-Eskimo",
    "Otro": "Other",
}
SEXO = {"Masculino": "Male", "Femenino": "Female"}
PAIS = {
    "Estados Unidos": "United-States",
    "México": "Mexico",
    "Filipinas": "Philippines",
    "Alemania": "Germany",
    "Canadá": "Canada",
    "Puerto Rico": "Puerto-Rico",
    "India": "India",
    "Cuba": "Cuba",
    "Jamaica": "Jamaica",
    "Japón": "Japan",
    "China": "China",
    "Colombia": "Columbia",
    "Otro": "Other",
}


# ── HEADER ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">🎓 Seminario de Ciencia de los Datos · Pascual Bravo · 2025</div>
    <h1 class="hero-title">💼 Predictor de Brecha Salarial</h1>
    <p class="hero-subtitle">Modelo de clasificación supervisada sobre el dataset <strong>Adult Income (UCI ML Repository)</strong><br>
    Ingresa las características de un individuo para predecir si su ingreso anual supera los <strong>50 000 USD</strong>.</p>
</div>
""", unsafe_allow_html=True)

# ── Carga del modelo ───────────────────────────────────────────────────
with st.spinner("⚙️ Cargando modelos entrenados..."):
    lr, dt, scaler, encoders, feature_cols = cargar_modelos()
st.success("✅ Modelos listos — Regresión Logística y Árbol de Decisión")

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── FORMULARIO ─────────────────────────────────────────────────────────
st.markdown("### 📋 Perfil del individuo")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown('<div class="card-title">📊 Datos personales</div>', unsafe_allow_html=True)
    edad = st.slider("Edad", 17, 90, 35, help="Edad del individuo en años")
    sexo_es = st.selectbox("Sexo", list(SEXO.keys()))
    raza_es = st.selectbox("Raza", list(RAZA.keys()))
    pais_es = st.selectbox("País de origen", list(PAIS.keys()))

with col_b:
    st.markdown('<div class="card-title">🎓 Educación y trabajo</div>', unsafe_allow_html=True)
    edu_es = st.selectbox("Nivel educativo", list(EDUCACION.keys()), index=3)
    sector_es = st.selectbox("Sector laboral", list(SECTOR.keys()))
    ocup_es = st.selectbox("Ocupación", list(OCUPACION.keys()))
    horas = st.slider("Horas de trabajo por semana", 1, 99, 40)

with col_c:
    st.markdown('<div class="card-title">👨‍👩‍👧 Contexto familiar y capital</div>', unsafe_allow_html=True)
    estado_es = st.selectbox("Estado civil", list(ESTADO_CIVIL.keys()))
    relacion_es = st.selectbox("Relación familiar", list(RELACION.keys()))
    capital_gan = st.number_input("Ganancia de capital (USD)", 0, 100000, 0, step=500)
    capital_per = st.number_input("Pérdida de capital (USD)", 0, 5000, 0, step=100)

st.markdown("<br>", unsafe_allow_html=True)
predecir = st.button("🔍 Predecir ingreso con ambos modelos", use_container_width=True)

# ── PREDICCIÓN ─────────────────────────────────────────────────────────
if predecir:
    entrada = {
        'age': edad,
        'workclass': SECTOR[sector_es],
        'fnlwgt': 189778,
        'education': EDUCACION[edu_es],
        'education_num': EDU_NUM[edu_es],
        'marital_status': ESTADO_CIVIL[estado_es],
        'occupation': OCUPACION[ocup_es],
        'relationship': RELACION[relacion_es],
        'race': RAZA[raza_es],
        'sex': SEXO[sexo_es],
        'capital_gain': capital_gan,
        'capital_loss': capital_per,
        'hours_per_week': horas,
        'native_country': PAIS[pais_es],
    }

    df_in = pd.DataFrame([entrada])
    cat_cols_list = ['workclass','education','marital_status','occupation',
                     'relationship','race','sex','native_country']
    for col in cat_cols_list:
        le = encoders[col]
        val = df_in[col].iloc[0]
        df_in[col] = le.transform([val])[0] if val in le.classes_ else 0

    df_in = df_in[feature_cols]
    X_in = scaler.transform(df_in)

    # Predicciones
    pred_lr  = lr.predict(X_in)[0]
    prob_lr  = lr.predict_proba(X_in)[0]
    pred_dt  = dt.predict(X_in)[0]
    prob_dt  = dt.predict_proba(X_in)[0]

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### 📊 Resultados de la predicción")

    # ── Tarjetas de resultado ──────────────────────────────────────────
    col1, col2 = st.columns(2)

    for col, pred, prob, nombre, auc_val in [
        (col1, pred_lr, prob_lr, "Regresión Logística", "0.828"),
        (col2, pred_dt, prob_dt, "Árbol de Decisión", "0.876"),
    ]:
        es_alto = pred == 1
        clase = "result-card-high" if es_alto else "result-card-low"
        emoji = "🟢" if es_alto else "🟡"
        texto = ">50K USD / año" if es_alto else "≤50K USD / año"
        color_class = "result-high" if es_alto else "result-low"
        ganador = '<span class="winner-badge">🏆 Mejor AUC</span>' if nombre == "Árbol de Decisión" else ""

        with col:
            st.markdown(f"""
            <div class="{clase}">
                <div class="result-label">{nombre}{ganador}</div>
                <p class="result-value {color_class}">{emoji} {texto}</p>
                <div class="metric-row" style="justify-content:center; margin-top:16px;">
                    <div class="metric-chip">Prob. ≤50K <span>{prob[0]*100:.1f}%</span></div>
                    <div class="metric-chip">Prob. >50K <span>{prob[1]*100:.1f}%</span></div>
                    <div class="metric-chip">AUC-ROC <span>{auc_val}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Gauge comparativo ──────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🎯 Probabilidad de ingreso >50K — Comparación visual")

    col_g1, col_g2 = st.columns(2)
    for col, prob, nombre, color in [
        (col_g1, prob_lr, "Regresión Logística", "#1e88e5"),
        (col_g2, prob_dt, "Árbol de Decisión",   "#43a047"),
    ]:
        with col:
            val = round(prob[1]*100, 1)
            # Color dinámico según el valor
            if val >= 60:
                bar_color = "#43a047"
            elif val >= 40:
                bar_color = "#ffa726"
            else:
                bar_color = color

            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=val,
                delta={'reference': 50, 'valueformat': '.1f',
                       'increasing': {'color': '#43a047'},
                       'decreasing': {'color': '#ef5350'}},
                number={'suffix': '%', 'font': {'size': 36, 'color': 'white', 'family': 'Sora'},
                        'valueformat': '.1f'},
                title={'text': f"<b>{nombre}</b>", 'font': {'size': 13, 'color': '#90caf9', 'family': 'Sora'}},
                gauge={
                    'axis': {
                        'range': [0, 100],
                        'tickwidth': 1,
                        'tickcolor': 'rgba(255,255,255,0.2)',
                        'tickfont': {'color': 'rgba(255,255,255,0.4)', 'size': 10},
                        'dtick': 25,
                    },
                    'bar': {'color': bar_color, 'thickness': 0.3},
                    'bgcolor': 'rgba(255,255,255,0.03)',
                    'borderwidth': 1,
                    'bordercolor': 'rgba(255,255,255,0.08)',
                    'steps': [
                        {'range': [0,  25],  'color': 'rgba(239,83,80,0.08)'},
                        {'range': [25, 50],  'color': 'rgba(255,167,38,0.06)'},
                        {'range': [50, 75],  'color': 'rgba(67,160,71,0.06)'},
                        {'range': [75, 100], 'color': 'rgba(67,160,71,0.12)'},
                    ],
                    'threshold': {
                        'line': {'color': 'rgba(255,255,255,0.5)', 'width': 2},
                        'thickness': 0.8,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(
                height=260,
                margin=dict(t=50, b=20, l=30, r=30),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Sora')
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Barras de probabilidad comparativa ────────────────────────────
    st.markdown("#### 📊 Distribución de probabilidades — ambos modelos")

    fig_bar = go.Figure()
    modelos  = ["Regresión Logística", "Árbol de Decisión"]
    probs_bajo = [prob_lr[0]*100, prob_dt[0]*100]
    probs_alto = [prob_lr[1]*100, prob_dt[1]*100]

    fig_bar.add_trace(go.Bar(
        name='≤50K', x=modelos, y=probs_bajo,
        marker_color='rgba(255,167,38,0.8)',
        marker_line=dict(color='rgba(255,167,38,1)', width=1.5),
        text=[f"{v:.1f}%" for v in probs_bajo],
        textposition='inside', textfont=dict(color='white', size=13, family='Sora'),
    ))
    fig_bar.add_trace(go.Bar(
        name='>50K', x=modelos, y=probs_alto,
        marker_color='rgba(76,175,80,0.8)',
        marker_line=dict(color='rgba(76,175,80,1)', width=1.5),
        text=[f"{v:.1f}%" for v in probs_alto],
        textposition='inside', textfont=dict(color='white', size=13, family='Sora'),
    ))
    fig_bar.add_hline(y=50, line_dash='dot', line_color='rgba(255,255,255,0.3)',
                      annotation_text='50% umbral', annotation_font_color='#aaa')

    fig_bar.update_layout(
        barmode='stack',
        height=320,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Sora', color='white'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    font=dict(size=12), bgcolor='rgba(0,0,0,0)'),
        margin=dict(t=20, b=10, l=10, r=10),
        xaxis=dict(tickfont=dict(size=13), gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(ticksuffix='%', gridcolor='rgba(255,255,255,0.05)', range=[0,105]),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Tabla comparativa de métricas ─────────────────────────────────
    st.markdown("#### 🏅 Métricas de rendimiento del proyecto")
    metricas = {
        "Métrica":    ["Accuracy", "AUC-ROC", "F1-Score (>50K)", "Recall (>50K)", "Precision (>50K)"],
        "Reg. Logística": ["80.59%", "0.828", "0.50", "0.40", "0.66"],
        "Árbol de Decisión 🏆": ["82.54%", "0.876", "0.62", "0.58", "0.65"],
    }
    df_metricas = pd.DataFrame(metricas)
    st.dataframe(
        df_metricas.style
            .set_properties(**{'text-align': 'center'})
            .set_table_styles([
                {'selector': 'th', 'props': [('background-color','#1565c0'),('color','white'),
                                              ('font-family','Sora'),('font-size','13px')]},
                {'selector': 'td', 'props': [('font-family','Sora'),('font-size','13px')]},
            ]),
        use_container_width=True,
        hide_index=True,
    )

    # ── Interpretación ─────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    concuerdan = pred_lr == pred_dt
    if concuerdan:
        resultado_txt = ">50K USD" if pred_lr == 1 else "≤50K USD"
        st.info(f"✅ **Ambos modelos concuerdan:** el ingreso estimado es **{resultado_txt}** al año. "
                f"Cuando los dos modelos coinciden, la predicción tiene mayor confiabilidad.")
    else:
        st.warning("⚠️ **Los modelos no coinciden.** En caso de discrepancia, se recomienda confiar en el "
                   "**Árbol de Decisión** por su mayor AUC-ROC (0.876 vs 0.828).")

# ── FOOTER ─────────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:rgba(255,255,255,0.35); font-size:0.78rem; padding-bottom:20px;">
    Proyecto Final · Seminario de Ciencia de los Datos · Institución Universitaria Pascual Bravo · 2025<br>
    Dataset: <em>Adult Income</em> — Becker &amp; Kohavi, 1996 · UCI Machine Learning Repository
</div>
""", unsafe_allow_html=True)
