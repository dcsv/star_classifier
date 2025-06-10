
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
import re

from matplotlib.colors import LogNorm
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

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

El dataset incluye cinco variables numéricas: temperatura (K), luminosidad relativa (L/Lo), radio relativo (R/Ro), magnitud absoluta (Mv)
y tipo de estrella (0: Enana Marrón, 1: Enana Roja, 2: Enana Blanca, 3: Secuencia Principal, 4: Supergigante, 5: Hipergigante).
También incorpora dos variables categóricas: color estelar (ej., blanco, azul, amarillo) y clase espectral (O, B, A, F, G, K, M).

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

# Se normalizan tipografías, eliminamos guiones,espacios extra y agrupamos alias semánticos antes de codificar “Star color”.
def clean_color(arr):
    """
    Acepta arr como DataFrame o ndarray.
    Convierte siempre a ndarray y aplana.
    Luego normaliza y agrupa alias.
    """
    # 1) asegúrate de trabajar con un ndarray de una sola columna
    a = np.asarray(arr).ravel()

    mapping = {
        'Yellow White':   'Yellowish White',
        'White Yellow':   'Yellowish White',
        'Yellowish White':'Yellowish White'
    }
    out = []
    for s in a:
        s2 = (s.lower().strip()
               .replace('-', ' ')
               .replace('_', ' '))
        s2 = re.sub(r'\s+', ' ', s2).title()
        out.append(mapping.get(s2, s2))
    #return np.array(out).reshape(-1, 1)
    return np.array(out, dtype=object).reshape(-1, 1)

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

temperature = st.number_input("Temperatura (K)", min_value=1939.0, max_value=40000.0, value=3068.0, format="%.6f")
luminosity = st.number_input("Luminosidad (L/Lo)", min_value=0.000080, max_value=849420.0, value=0.0024, format="%.6f")
radius = st.number_input("Radio (R/Ro)", min_value=0.0084, max_value=1948.5, value=0.17, format="%.6f")
magnitude = st.number_input("Magnitud absoluta (Mv)", min_value=-11.9, max_value=20.06, value=16.12, format="%.6f")

star_color = st.selectbox("Color de la estrella", [
    "Blue", "Blue White", "Orange", "Orange Red", "Pale Yellow Orange", "Red", "White", "Whitish", "Yellowish", "Yellowish White"
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
        0: "Enana marrón", 1: "Enana roja", 2: "Enana blanca",
        3: "Secuencia principal", 4: "Super gigante", 5: "Hipergigante"
    }
    predicted_type = star_type_dict.get(y_pred[0], "Desconocido")
    ##---
    st.subheader("Diagrama de Hertzsprung–Russell")

    # Datos
    temp_all   = df["Temperature (K)"]
    lum_all    = df["Luminosity(L/Lo)"]
    spec_all   = df["Spectral Class"]
    pred_temp  = temperature
    pred_lum   = luminosity

    # 1. Configuración de figura y ejes
    fig, ax = plt.subplots(figsize=(8,6), facecolor="black")
    ax.set_facecolor("black")

    # 2. Scatter de todas las clases espectrales
    spec_colors = {
        "O":"#9bb0ff", "B":"#aabfff", "A":"#cad7ff",
        "F":"#f8f7ff", "G":"#fff4ea", "K":"#ffd2a1",
        "M":"#ffcc6f"
    }
    sns.scatterplot(
        x=temp_all, y=lum_all, hue=spec_all,
        palette=spec_colors, s=30, alpha=0.8,
        ax=ax, legend=False
    )

    # 3. Invertir eje X y log en Y
    ax.invert_xaxis()
    ax.set_yscale("log")
    ax.set_xlim(40000, 2000)
    ax.set_ylim(1e-4, 1e6)

    ax.set_xlabel("Temperature (K)", color="white")
    ax.set_ylabel("Luminosity (L/Lo)", color="white")
    ax.tick_params(colors="white", which="both")

    # 4. Colorbar horizontal arco iris invertido
    norm = LogNorm(vmin=temp_all.min(), vmax=temp_all.max())
    sm   = plt.cm.ScalarMappable(cmap="nipy_spectral_r", norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(
        sm, ax=ax, orientation="horizontal",
        pad=0.12, fraction=0.05, aspect=40
    )
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.set_label("Temperatura (K)", color="white")
    cbar.ax.tick_params(colors="white")

    # 5. Etiquetas de regiones con borde blanco
    regions = {
        "White Dwarfs": (35000, 1e-3),
        "Main Sequence": (15000, 1e1),
        "Giants": (7000, 1e3),
        "Supergiants": (7000, 1e5)
    }
    for txt, (x, y) in regions.items():
        txt_obj = ax.text(
            x, y, txt, color="white", fontsize=12,
            path_effects=[pe.withStroke(linewidth=3, foreground="black")]
        )

    # 6. “Tu estrella” encima de todo en amarillo
    ax.scatter(
        pred_temp, pred_lum,
        marker="*", s=250, c="yellow",
        edgecolor="black", linewidth=1.5,
        zorder=5
    )

    # 7. Leyenda manual por delante
    handles = [
        Line2D([0],[0], marker="o", color=col, linestyle="", ms=8)
        for col in spec_colors.values()
    ] + [Line2D([0],[0], marker="*", color="yellow", linestyle="", ms=15)]
    labels = list(spec_colors.keys()) + [f"Tu estrella ({predicted_type})"]

    leg = ax.legend(
        handles, labels,
        title="Clase espectral",
        facecolor="black", framealpha=0.6,
        edgecolor="white",
        labelcolor="white",
        title_fontsize=12,
        loc="upper right"
    )
    for text in leg.get_texts():
        text.set_color("white")

    st.pyplot(fig)
    ##---
    st.success(f"\U0001F31F Tipo de estrella predicho: **{predicted_type}** (Clase {y_pred[0]})")

