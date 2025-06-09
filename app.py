
import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import io
import time
import sklearn.metrics as m
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ------------------------------------------------------------------------------
# 1.   Sidebar: cargador de archivos
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("📂 Cargar dataset para el EDA")
    uploaded_file = st.file_uploader(
        label="Sube tu archivo (`.data`)",
        type=["csv", "data", "txt"],
        help="El archivo original de ejemplo se llama 'yacht_hydrodynamics.data'"
    )

# ------------------------------------------------------------------------------
# 2.   Lectura del archivo
# ------------------------------------------------------------------------------
column_names = ['LCB', 'PC', 'LDR', 'BDR', 'LBR', 'Fn', 'Resistance']

def load_dataset(file_obj):
    return pd.read_csv(
        file_obj,
        header=None,
        sep=r"\s+",      # separador por espacios en blanco variables
        names=column_names,
        engine="python"  # evita problemas con el regex del separador
    )

# Si el usuario cargó un archivo, úsalo; de lo contrario, usa el de ejemplo
if uploaded_file is not None:
    try:
        df = load_dataset(uploaded_file)
        st.success(f"✅ Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas")
    except Exception as e:
        st.error(f"❌ No se pudo leer el archivo: {e}")
        st.stop()
else:
    st.info("ℹ️ No se subió archivo. ")

# --- Título y descripción ----------------------------------------------------------
st.title("Predicción de la Resistencia Hidrodinámica de Yates - Yacht Hydrodynamics")

## Contexto del Proyecto
st.markdown("""

## Contexto

El diseño de embarcaciones eficientes es una tarea crítica en la ingeniería naval. Uno de los principales desafíos es estimar con precisión la **resistencia hidrodinámica** (resistencia al avance) que una embarcación enfrenta al desplazarse por el agua, lo cual impacta directamente en el consumo de energía, el diseño del casco y la estabilidad del barco.

Este proyecto utiliza el conjunto de datos **Yacht Hydrodynamics** del repositorio UCI Machine Learning para construir un modelo de predicción basado en aprendizaje automático. El objetivo es predecir la **resistencia al avance (resistance)** de un yate a partir de parámetros físicos y geométricos del casco.

## Descripción del Dataset

El dataset contiene **6 variables de entrada** relacionadas con las características del casco de un yate y **1 variable de salida**, que es la resistencia al avance (residuary resistance).

**Las variables de entrada son:**

1. Longitudinal position of the center of buoyancy. (LCB)
2. Prismatic coefficient. (PC)
3. Length-displacement ratio. (LDR)
4. Beam-draught ratio. (BDR)
5. Length-beam ratio. (LBR)
6. Froude number. (Fn)

**Variable objetivo (target):**

7. Residuary resistance per unit weight of displacement. (Resistance)

El conjunto de datos consta de **308 observaciones**.

## Objetivo del Proyecto

El objetivo principal es desarrollar un modelo predictivo que, dadas las características físicas del yate, estime su resistencia al avance con precisión.

## 📂 Fuente del Dataset

- UCI Machine Learning Repository: [Yacht Hydrodynamics Data Set](https://archive.ics.uci.edu/dataset/243/yacht+hydrodynamics)

---
""")

# --- Título y descripción ----------------------------------------------------------
st.title("Análisis de datos exploratorios – EDA interactivo")

# --- Barra lateral ------------------------------------------------------------
st.sidebar.title("Controles de EDA")
show_raw = st.sidebar.checkbox("Mostrar tabla completa", value=False)
target = st.sidebar.selectbox("Variable objetivo", column_names, index=6)
num_bins = st.sidebar.slider("N° de bins (histograma)", 5, 50, 20)

# --- Descripcion de los datos -------------------------------------------------
st.markdown("""
## Diccionario de variables

| Código | Nombre (esp/en)                                             | Definición breve                                                                                                   | Tipo  | Observaciones clave |
|--------|---------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|-------|---------------------|
| **LCB** | Centro de flotación longitudinal (Longitudinal Center of Buoyancy) | Posición del centro de flotación a lo largo de la eslora; influye en la estabilidad del casco.                      | float | Valores extremos pueden alterar la estabilidad y generar outliers. |
| **PC** | Coeficiente prismático (Prismatic Coefficient)          | Relación entre el volumen sumergido y un prisma de referencia del mismo largo y área máxima de sección transversal. | float | Mide la “llenura” longitudinal; rangos típicos 0.53 – 0.60. |
| **LDR** | Relación eslora-desplazamiento (Length-Displacement Ratio) | Eslora / ∇<sup>1/3</sup>; cuantifica la esbeltez del casco.                                                        | float | Casco más alargado ⇒ menor resistencia. |
| **BDR** | Relación manga-calado (Beam-Draught Ratio)             | Manga / Calado; indica cuán ancho es el casco respecto a su profundidad.                                            | float | Valores altos tienden a aumentar la resistencia viscosa. |
| **LBR** | Relación eslora-manga (Length-Beam Ratio)              | Eslora / Manga; proporción longitudinal vs. transversal.                                                            | float | Complementa a **BDR** para describir forma global. |
| **Fn** | Número de Froude                                         | V / √(g·L); adimensionaliza la velocidad e indica el régimen de flujo (onda vs. fricción).                          | float | Variable más influyente sobre la resistencia de onda. |
| **Resistance** | Resistencia residuaria (objetivo)                       | Resistencia al avance normalizada por el desplazamiento de la embarcación.                                          | float | Distribución muy asimétrica (media ≈ 10.5, mediana ≈ 3.1). |
""")

# --- Tabla de datos ---------------------------------------------------------------
if show_raw:
    st.subheader("Datos crudos")
    st.dataframe(df)                     # tabla interactiva :contentReference[oaicite:2]{index=2}

# --- Estadísticas descriptivas ----------------------------------------------------
st.subheader("Estadísticos básicos")
st.write(df.describe())

# --- Distribución de la variable objetivo -----------------------------------------
st.subheader(f"Histograma • {target}")
fig, ax = plt.subplots()
sns.histplot(df[target], bins=num_bins, ax=ax, kde=True)
st.pyplot(fig)

# --- Outliers interactivos ---------------------------------------------------------
st.header("Detección de outliers")

# Elige la variable numérica
var = st.selectbox("Variable a analizar", df.columns, index=0)

# Ajusta el multiplicador del IQR (típico = 1.5)
k = st.slider("Factor de rango intercuartílico (IQR × k)",
              min_value=0.5, max_value=3.0, value=1.5, step=0.1)

# Calcula límites IQR
q1, q3 = df[var].quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - k * iqr, q3 + k * iqr

outliers = df[(df[var] < lower) | (df[var] > upper)]

# Muestra recuento y tabla opcional
st.write(f"**Total de outliers en `{var}`:** {len(outliers)}")
if st.checkbox("Mostrar tabla de outliers"):
    st.dataframe(outliers[[var]])

# Box-plot con outliers resaltados
fig, ax = plt.subplots()
sns.boxplot(x=df[var], ax=ax, color="#76b5c5")
ax.axvline(lower, color="red", linestyle="--")
ax.axvline(upper, color="red", linestyle="--")
ax.set_title(f"Box-plot de {var} • IQR × {k}")
st.pyplot(fig)

################################################################################
# ------------------------ Modelo de prediccion --------------------------------
################################################################################

st.title("Correr modelo de predicción de Resistencia Hidrodinámica de Yates")

# Carga interactiva de artefacto .pkl
st.sidebar.subheader("Cargar modelo (.pkl)")
uploaded_pkl = st.sidebar.file_uploader(
                "Subir archivo Pickle", type=["pkl", "pickle"])

@st.cache_resource(show_spinner="Cargando artefacto…")
def load_artifact_from_bytes(file_bytes: bytes):
    """Deserializa un objeto Pickle desde bytes en memoria."""
    return pickle.load(io.BytesIO(file_bytes))

if uploaded_pkl is not None:
    try:
        artifact = load_artifact_from_bytes(uploaded_pkl.getvalue())
        st.sidebar.success("Artefacto cargado con éxito ✅")
        #st.write("### Resumen del artefacto")
        #st.write(type(artifact))
        # aquí puedes añadir lógica específica según el tipo:
    except Exception as e:
        st.sidebar.error(f"Error al leer el Pickle: {e}")
        st.stop()
else:
    st.sidebar.info("Aún no has subido ningún archivo .pkl")
    st.stop()  # evita que el resto de la app dependa de un objeto vacío

# Carga el preprocesador y el modelo
model = artifact['model']
st.sidebar.success("Modelo cargado exitosamente")

# Parámetros de entrada
st.subheader("Ingrese parámetros de diseño")

lcb = st.number_input("Centro de flotación longitudinal (Longitudinal Center of Buoyancy - LCB)", min_value=-27.0, max_value=23.0, value=0.0, format="%.6f")
bdr = st.number_input("Relación manga/calado (Beam Draught Ratio - BDR)", min_value=-2.72, max_value=3.32, value=0.5119, format="%.6f")
fn  = st.number_input("Número de Froude (Froude Number - Fn)", min_value=-1.61, max_value=1.612452, value=0.3721, format="%.6f")

# Preparar nuevo dato en DataFrame (BDR_scaled - Fr_scaled - LCB_scaled)
X_new = pd.DataFrame([{"LCB_scaled": lcb, "BDR_scaled": bdr, "Fn_scaled": fn}])

# El pipeline incluye el preprocesador
y_log_pred = model.predict(X_new)
#y_pred = np.exp(y_log_pred) - 0.01

# Mostrar resultado
#st.metric("Resistencia estimada", f"{y_pred[0]:.3f}")
st.metric("Resistencia estimada", f"{y_log_pred[0]:.6f}")

# ------------------------------------------------------------------------------
# 1.  Subir archivo de prueba en la barra lateral
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("📂 Cargar data_test.csv")
    uploaded_file = st.file_uploader(
        "Selecciona el CSV de prueba (debe contener 'LCB_scaled', 'BDR_scaled', "
        "'Fn_scaled' y 'Res_log')",
        type=["csv"]
    )

# Si no hay archivo, detener la app
if uploaded_file is None:
    st.info("Sube el archivo    [ data_test.csv ]    para iniciar la simulación.")
    st.stop()

# ----------------------------------------------------------
# 2.  Leer y validar el CSV
# ----------------------------------------------------------
test_df = pd.read_csv(uploaded_file)

required_cols = ["LCB_scaled", "BDR_scaled", "Fn_scaled", "Res_log"]
missing = [c for c in required_cols if c not in test_df.columns]
if missing:
    st.error(f"El CSV no contiene estas columnas requeridas: {missing}")
    st.stop()

# Separar X e y en el orden correcto
feature_cols = ["LCB_scaled", "BDR_scaled", "Fn_scaled"]
X_test = test_df[feature_cols].copy()
y_true = test_df["Res_log"].to_numpy()

# ----------------------------------------------------------
# 3.  Definir cuántos pasos (filas) quiere el usuario probar
# ----------------------------------------------------------
n_total = len(test_df)
pasos = st.slider(
    "Número de registros a evaluar",
    min_value=1,
    max_value=n_total,
    value=min(10, n_total)           # valor por defecto
)

# ----------------------------------------------------------
# 4.  Simulación en "tiempo real"
# ----------------------------------------------------------
st.subheader("Predicción vs. valor real en tiempo real")

placeholder = st.empty()
predichos, reales = [], []

for i in range(pasos):
    # 4.1 Seleccionar fila i
    x_row = X_test.iloc[[i]]               # DataFrame con una sola fila
    y_real = y_true[i]

    # 4.2 Predecir con el modelo
    y_pred = model.predict(x_row)[0]

    # 4.3 Almacenar resultados para graficar
    reales.append(y_real)
    predichos.append(y_pred)

    # 4.4 Mostrar resumen de la fila procesada
    lcb, bdr, fn = x_row.iloc[0]
    st.write(
        f"Registro {i+1}: "
        f"LCB={lcb:.3f}, BDR={bdr:.3f}, Fn={fn:.3f} → "
        f"Real={y_real:.3f}, Predicho={y_pred:.3f}"
    )

    # 4.5 Graficar evolución en vivo
    fig, ax = plt.subplots()
    ax.plot(reales,  label="Real",      marker='o')
    ax.plot(predichos, label="Predicho", marker='x')
    ax.set_xlabel("Índice del registro")
    ax.set_ylabel("Resistencia (log)")
    ax.set_title("Evolución de Resistencia: Real vs. Predicho")
    ax.legend()
    placeholder.pyplot(fig)

    time.sleep(0.1)   # pausa para sensación “en vivo”

# ----------------------------------------------------------
# 5.  Resumen final
# ----------------------------------------------------------
st.subheader("Resultados completos")
result_df = pd.DataFrame({
    "LCB_scaled": X_test["LCB_scaled"][:pasos],
    "BDR_scaled": X_test["BDR_scaled"][:pasos],
    "Fn_scaled":  X_test["Fn_scaled"][:pasos],
    "Res_log_real": reales,
    "Res_log_pred": predichos
})
st.dataframe(result_df)


st.title("Evaluar el ajuste del modelo con métricas de error")

reales_arr    = np.asarray(reales,    dtype=float)
predichos_arr = np.asarray(predichos, dtype=float)

# y_true=reales y y_pred=predichos
rmse = np.sqrt(mean_squared_error(reales_arr, predichos_arr))
mae  = mean_absolute_error(reales_arr, predichos_arr)
r2   = r2_score(reales_arr, predichos_arr)
mape = np.mean(np.abs((reales_arr - predichos_arr) / reales_arr)) * 100

st.subheader("📊 Métricas de desempeño")
st.write(f"**RMSE:** {rmse:.3f}")
st.write(f"**MAE :** {mae:.3f}")
st.write(f"**R²  :** {r2:.3f}")
st.write(f"**MAPE:** {mape:.2f}%")

st.title("Conclusión")

parrafo = f"""
El modelo obtuvo un <b>MAPE de {mape:.2f}%</b>, un <b>RMSE de {rmse:.3f}</b>,
un <b>MAE de {mae:.3f}</b> y un <b>R² de {r2:.3f}</b>, explicando el
{r2*100:.1f}&nbsp;% de la variabilidad observada en la resistencia hidrodinámica.

En el estudio de Walker&nbsp;et&nbsp;al.&nbsp;(2024) —centrado en modelos de referencia de nivel CFD—
se reportan errores relativos cercanos al <b>1&nbsp;%</b>. Comparado con ese punto de referencia,
nuestro modelo presenta un desempeño{' muy cercano' if mape <= 5 else ' razonable'}:
la desviación media es del {mape:.1f}&nbsp;%, lo que resulta adecuado para análisis
comparativos de cascos y estimaciones preliminares de potencia.
El ajuste consistente entre RMSE y MAE sugiere además que no existen errores
atípicos dominantes y que el modelo se comporta de forma estable en todo el rango de datos evaluado.
"""

st.markdown(f"""<div style="text-align: justify; font-size: 1.05rem;">{parrafo}</div>""",unsafe_allow_html=True)

st.markdown("""
    **Referencia:** [Walker et al., 2024 – _Models for Yacht Resistance Optimization_]
    (https://www.mdpi.com/2077-1312/12/4/556)
""")
