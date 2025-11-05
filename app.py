import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore")

# --- Librerias para ARIMA ---
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

st.set_page_config(page_title="EDA + ARIMA (BMV) • Multi-ticker", layout="wide")

# -------------- Sidebar --------------
st.sidebar.title("Carga de Datos")

# Subir archivo desde la computadora
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])

# Campos para mapear columnas
col_date = st.sidebar.text_input("Columna de fecha", value="date")
col_ticker = st.sidebar.text_input("Columna de ticker", value="instrument_id")
col_price = st.sidebar.text_input("Columna de precio ajustado", value="adj_close")

# Si el usuario sube un archivo
if uploaded_file is not None:
    try:
        # Leer directamente desde el archivo subido
        df = pd.read_csv(uploaded_file, parse_dates=[col_date])
        df = df.rename(columns={col_date: "date", col_ticker: "ticker", col_price: "adj_close"})
        
        st.success("✅ Archivo cargado correctamente")
        st.write("Vista previa de los datos:")
        st.dataframe(df.head(10))
        
        # Mostrar resumen general
        st.write("📊 Resumen estadístico:")
        st.write(df.describe(include="all"))
        
    except Exception as e:
        st.error(f"❌ Error al cargar los datos: {e}")
else:
    st.info("👆 Sube un archivo CSV para comenzar.")