
import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import io
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import urllib.request
from sklearn.base import BaseEstimator, TransformerMixin

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

La clasificación de estrellas es una tarea fundamental en astrofísica, ya que nos permite entender la estructura, evolución y composición del universo.
 Al clasificar una estrella, se busca agruparla según sus características físicas, lo que nos ayuda a predecir su comportamiento, su historia evolutiva y su futuro.
 Este dataset es una colección estructurada de información física y categórica sobre diferentes tipos de estrellas, con el objetivo de construir un modelo que prediga
  su tipo estelar (Star type) a partir de sus propiedades observables.

**Las variables de entrada son:**

1. Temperature (K)
2. Luminosity(L/Lo)
3. Radius(R/Ro)
4. Absolute magnitude(Mv)
5. Spectral Class
6. Star color

**Variable objetivo (target):**

7. Star type

El conjunto de datos consta de **240 observaciones**.

## Objetivo del Proyecto

El objetivo principal es desarrollar un modelo predictivo que, dadas las características de las estrellas, estime el tipo de estrella.

## 📂 Fuente del Dataset

- https://www.kaggle.com/datasets/deepu1109/star-dataset

---
""")

# --- Título y descripción ----------------------------------------------------------
st.title("Análisis de datos exploratorios – EDA interactivo")
st.sidebar.title("Controles de EDA")
show_raw = st.sidebar.checkbox("Mostrar tabla completa", value=False)
num_bins = st.sidebar.slider("N° de bins (histograma)", 5, 50, 20)

if show_raw:
    st.subheader("\U0001F4C4 Datos crudos del dataset")
    st.dataframe(df)

st.subheader("\U0001F4CA Estadísticos descriptivos")
numeric_cols = df.select_dtypes(include=["number"]).drop(columns=["Star type"])
st.write(numeric_cols.describe())

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

# URL RAW del archivo .pkl en GitHub
pkl_url = "https://raw.githubusercontent.com/dcsv/star_classifier/main/best_model_pipeline_RF.pkl"

st.sidebar.subheader("\U0001F4E6 Cargar modelo desde GitHub")

# Clase personalizada
class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.8):
        self.threshold = threshold

    def fit(self, X, y=None):
        # aceptar DataFrame o ndarray
        if isinstance(X, pd.DataFrame):
            cols = list(X.columns)
            df   = X
        else:
            cols = [f"x{i}" for i in range(X.shape[1])]
            df   = pd.DataFrame(X, columns=cols)

        corr    = df.corr().abs()
        upper   = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        self.to_drop_        = [c for c in upper.columns if any(upper[c] > self.threshold)]
        self.feature_names_in_ = cols
        return self

    def transform(self, X):
        df = pd.DataFrame(X, columns=self.feature_names_in_)
        return df.drop(columns=self.to_drop_, errors="ignore").values

    def get_feature_names_out(self, input_features=None):
        # devuelve los nombres que quedan tras el drop
        if input_features is None:
            input_features = self.feature_names_in_
        return [c for c in input_features if c not in self.to_drop_]

@st.cache_resource(show_spinner="Cargando modelo desde GitHub...")
def load_pipeline_from_url(url: str):
    with urllib.request.urlopen(url) as response:
        return pickle.load(response)

try:
    model = load_pipeline_from_url(pkl_url)
    st.sidebar.success("\u2705 Modelo cargado correctamente desde GitHub.")
except Exception as e:
    st.sidebar.error(f"\u274C Error al cargar el modelo: {e}")
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
