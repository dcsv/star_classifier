
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

