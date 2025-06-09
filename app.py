
import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import io
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ------------------------------------------------------------------------------
# 1. Sidebar: cargador de archivos
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("\U0001F4C2 Dataset predeterminado")
    try:
        df = pd.read_csv("6 class csv.csv", sep=",")
        st.success("\u2705 Dataset cargado desde el repositorio.")
    except FileNotFoundError:
        st.error("\u274C No se encontró el archivo en el repositorio.")
        st.stop()

# --- Título y descripción ----------------------------------------------------------
st.title("Star Classifier")

## Contexto del Proyecto
st.markdown("""
## Contexto

La clasificación de estrellas es una tarea fundamental en astrofísica... (contenido omitido por brevedad)
""")

# --- EDA ----------------------------------------------------------------------
st.title("Análisis de datos exploratorios – EDA interactivo")
st.sidebar.title("Controles de EDA")
show_raw = st.sidebar.checkbox("Mostrar tabla completa", value=False)
num_bins = st.sidebar.slider("N° de bins (histograma)", 5, 50, 20)

if show_raw:
    st.subheader("\U0001F4C4 Datos crudos del dataset")
    st.dataframe(df)

st.subheader("\U0001F4CA Estadísticos descriptivos")
st.write(df.describe())

st.subheader("\U0001F4CC Distribución de clases • `Star type`")
fig, ax = plt.subplots()
sns.countplot(data=df, x="Star type", palette="viridis", ax=ax)
ax.set_title("Distribución de tipos estelares (clases 0–5)")
st.pyplot(fig)

with st.expander("\U0001F50E Distribución de variables categóricas"):
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        st.markdown(f"**{col}**")
        st.write(df[col].value_counts())

st.header("\U0001F4CC Detección de outliers")
num_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
var = st.selectbox("Selecciona una variable numérica", num_cols, index=0)
k = st.slider("Factor multiplicador del IQR (k)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
q1, q3 = df[var].quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - k * iqr, q3 + k * iqr
outliers = df[(df[var] < lower) | (df[var] > upper)]
st.write(f"\U0001F539 Total de outliers en `{var}`: **{len(outliers)}**")
if st.checkbox("Mostrar tabla de outliers"):
    st.dataframe(outliers[[var]])
fig, ax = plt.subplots()
sns.boxplot(x=df[var], ax=ax, color="#76b5c5")
ax.axvline(lower, color="red", linestyle="--", label="Límite inferior")
ax.axvline(upper, color="red", linestyle="--", label="Límite superior")
ax.set_title(f"Box-plot de `{var}` • IQR × {k}")
ax.legend()
st.pyplot(fig)

# ------------------------------------------------------------------------------
# Modelo predictivo
# ------------------------------------------------------------------------------
st.title("\U0001F52D Predicción del Tipo de Estrella")

st.sidebar.subheader("\U0001F4E6 Cargar modelo (.pkl)")
uploaded_pkl = st.sidebar.file_uploader("Subir archivo Pickle", type=["pkl", "pickle"])

@st.cache_resource(show_spinner="Cargando modelo...")
def load_pipeline_from_bytes(file_bytes: bytes):
    return pickle.load(io.BytesIO(file_bytes))

if uploaded_pkl is not None:
    try:
        model = load_pipeline_from_bytes(uploaded_pkl.getvalue())
        st.sidebar.success("\u2705 Modelo cargado correctamente.")
    except Exception as e:
        st.sidebar.error(f"\u274C Error al cargar el modelo: {e}")
        st.stop()
else:
    st.sidebar.info("\u231B Esperando archivo Pickle...")
    st.stop()

st.header("\U0001F9EA Ingresar características de la estrella")

temperature = st.number_input("Temperatura (K)", min_value=1900, max_value=50000, value=5000)
luminosity = st.number_input("Luminosidad (L/Lo)", min_value=0.0001, max_value=100000.0, value=1.0)
radius = st.number_input("Radio (R/Ro)", min_value=0.01, max_value=1000.0, value=1.0)
magnitude = st.number_input("Magnitud absoluta (Mv)", min_value=-10.0, max_value=25.0, value=5.0)

star_color = st.selectbox("Color de la estrella", [
    "Red", "Blue", "White", "Yellow", "Orange", "Blue-white", "Yellowish", "Whitish"
])
spectral_class = st.selectbox("Clase espectral", ["O", "B", "A", "F", "G", "K", "M"])

X_new = pd.DataFrame([{
    "Temperature (K)": temperature,
    "Luminosity(L/Lo)": luminosity,
    "Radius(R/Ro)": radius,
    "Absolute magnitude(Mv)": magnitude,
    "Star color": star_color,
    "Spectral Class": spectral_class
}])

if st.button("\U0001F50D Predecir tipo de estrella"):
    y_pred = model.predict(X_new)
    star_type_dict = {
        0: "Enana roja", 1: "Enana marrón", 2: "Enana blanca",
        3: "Secuencia principal", 4: "Subgigante", 5: "Gigante"
    }
    predicted_type = star_type_dict.get(y_pred[0], "Desconocido")
    st.success(f"\U0001F31F Tipo de estrella predicho: **{predicted_type}** (Clase {y_pred[0]})")

# ------------------------------------------------------------------------------
# Evaluación del modelo con archivo de prueba
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("\U0001F4C2 Cargar data_test.csv")
    uploaded_file = st.file_uploader(
        "Selecciona el CSV de prueba (debe contener las variables predictoras y `Star type`)",
        type=["csv"]
    )

if uploaded_file is None:
    st.info("Sube el archivo [data_test.csv] para evaluar el modelo.")
    st.stop()

test_df = pd.read_csv(uploaded_file)
required_cols = [
    "Temperature (K)", "Luminosity(L/Lo)", "Radius(R/Ro)",
    "Absolute magnitude(Mv)", "Star type"
]
missing = [c for c in required_cols if c not in test_df.columns]
if missing:
    st.error(f"Faltan columnas requeridas: {missing}")
    st.stop()

X_test = test_df[[c for c in required_cols if c != "Star type"]]
y_true = test_df["Star type"]

st.subheader("Clasificación real vs. predicha")
pasos = st.slider("Número de registros a evaluar", 1, len(X_test), value=min(10, len(X_test)))
y_pred = []

for i in range(pasos):
    x_row = X_test.iloc[[i]]
    pred = model.predict(x_row)[0]
    real = y_true.iloc[i]
    y_pred.append(pred)
    st.write(f"Registro {i+1}: Real = {real}, Predicho = {pred}")

st.subheader("\U0001F4CA Métricas de desempeño del modelo")
acc = accuracy_score(y_true[:pasos], y_pred)
prec = precision_score(y_true[:pasos], y_pred, average='weighted', zero_division=0)
rec = recall_score(y_true[:pasos], y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true[:pasos], y_pred, average='weighted', zero_division=0)

st.write(f"**Accuracy:** {acc:.3f}")
st.write(f"**Precision:** {prec:.3f}")
st.write(f"**Recall:** {rec:.3f}")
st.write(f"**F1-score:** {f1:.3f}")

st.subheader("\U0001F50D Matriz de confusión")
conf_mat = confusion_matrix(y_true[:pasos], y_pred)
fig, ax = plt.subplots()
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", cbar=False)
ax.set_xlabel("Predicho")
ax.set_ylabel("Real")
ax.set_title("Matriz de Confusión")
st.pyplot(fig)


