
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
st.sidebar.write("🔍 Debug: se llegó a la barra lateral")
print("🔧 Debug: se llegó a la barra lateral (consola)")
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
