# Dashboard
Dashboard donde estara el modelo ARIMA
# 📈 EDA + ARIMA (BMV) • Multi-ticker

Aplicación **Streamlit** para explorar series de tiempo de precios (multi-ticker) y ajustar modelos **SARIMAX (ARIMA estacional)** automáticamente. Incluye análisis exploratorio (EDA), descomposición STL, ACF/PACF, prueba ADF y pronóstico con bandas de confianza.

---

## ✨ Funcionalidades principales

- 📊 Carga de datos CSV con mapeo flexible de columnas (fecha, ticker, precio ajustado).
- ⏱️ Re-muestreo por frecuencia: Diario, Semanal, Bursátil, Mensual o Trimestral.
- 🔄 Transformaciones: **Precio**, **Log-precio**, **Retorno (%)**, **Retorno log (%)**.
- 🔍 EDA por ticker: resumen estadístico, ADF, STL, ACF, PACF, residuales.
- ⚙️ Ajuste automático de **(p,d,q)×(P,D,Q,m)** con heurísticas para `d` y `D`.
- 🔮 Pronóstico multi-paso con bandas de confianza al 95%.
- 📉 Métricas in-sample (RMSE, MAE).
- 📈 Comparación multi-ticker con métricas anuales: retorno, volatilidad y Sharpe.

---

## 🧩 Requisitos

- **Python 3.10 o superior**
- Compatible con **Windows / macOS / Linux**

### Dependencias necesarias

Crea un archivo `requirements.txt` con lo siguiente:

```txt
numpy>=1.23
pandas>=2.0
streamlit>=1.33
plotly>=5.16
matplotlib>=3.7
statsmodels>=0.14
scipy>=1.10
patsy>=0.5
```

Instálalas con:

```bash
pip install -r requirements.txt
```

---

## 🚀 Ejecución

1. Coloca tu archivo CSV dentro de una carpeta `datos/`, por ejemplo:  
   `datos/market_prices.csv`
2. Abre una terminal y ejecuta:

```bash
streamlit run app.py
```

3. Se abrirá la app en tu navegador en  
   👉 [http://localhost:8501](http://localhost:8501)

---

## 📄 Formato del archivo CSV

Debe tener al menos tres columnas:

| Columna       | Descripción                         | Ejemplo        |
|----------------|--------------------------------------|----------------|
| `date`         | Fecha (YYYY-MM-DD)                   | 2024-01-31     |
| `ticker`       | Identificador del instrumento        | BIMBOA_MX      |
| `adj_close`    | Precio ajustado                      | 77.45          |

Ejemplo:

```csv
date,ticker,adj_close
2023-01-31,BIMBOA_MX,77.45
2023-02-28,BIMBOA_MX,78.10
2023-01-31,WALMEX_MX,62.30
2023-02-28,WALMEX_MX,63.05
```

> Si tus columnas tienen otros nombres, puedes **mapearlas** en la barra lateral de la app.

---

## 🧭 Uso paso a paso

### 1️⃣ Cargar datos
- Escribe la ruta del CSV en la barra lateral.  
- Si las columnas no se llaman `date`, `ticker`, `adj_close`, ajusta el mapeo.  
- Elige la frecuencia (**D**, **W**, **B**, **M**, **Q**) y la transformación deseada.  

### 2️⃣ Configurar modelo
- Marca si quieres modelar estacionalidad (**SARIMA/STL**).  
- Ajusta el periodo `m` según la frecuencia (por defecto: D=7, W=52, B=5, M=12, Q=4).  
- Define el horizonte de pronóstico (pasos hacia adelante).  

### 3️⃣ Pestañas principales
#### 📊 Resumen multi-ticker
- Muestra las series re-muestreadas.  
- Calcula retorno anual, volatilidad anual y Sharpe ratio.  

#### 🔍 EDA por ticker
- Gráficos de serie, estadísticos, prueba ADF.  
- Descomposición STL (observado, tendencia, estacional, residuales).  
- ACF y PACF.  

#### 📈 ARIMA por ticker
- Ajusta modelo SARIMAX automáticamente.  
- Muestra el modelo seleccionado, ajuste, pronóstico y bandas de confianza.  
- Calcula RMSE y MAE del ajuste in-sample.  

---

## ⚙️ Detalles técnicos del modelo

- `infer_d(y)`: determina el número de diferencias `d` con la prueba ADF.  
- `infer_D(y, m)`: decide `D=1` si la autocorrelación en lag `m` > 0.4.  
- Rejilla de búsqueda:  
  - `p, q ∈ {0,1,2}`  
  - `P, Q ∈ {0,1}` (si hay estacionalidad)  
- Fallback: `(1, max(1,d), 1)` si no converge.  

---

## 🧪 Recomendaciones

- Asegúrate de que tu serie esté **ordenada por fecha** y sin valores nulos.  
- Usa retornos (no precios) si las series no son estacionarias.  
- Ajusta el parámetro `m` según el tipo de frecuencia:  
  - Diario → 7 (semanal)  
  - Semanal → 52  
  - Mensual → 12  
  - Trimestral → 4  
- El horizonte `h` se mide en pasos de la frecuencia elegida (meses si es M).  

---

## 🛠️ Solución de errores comunes

| Problema | Posible causa | Solución |
|-----------|----------------|-----------|
| ❌ “No pude leer el CSV” | Ruta incorrecta o columnas no mapeadas | Revisa la ruta y mapea columnas correctamente |
| ⚠️ “Serie vacía tras la transformación” | Muchos NaN al aplicar log o retornos | Usa otra transformación |
| ❗ “ValueError en STL o ACF” | Serie demasiado corta | Usa frecuencia más baja o un ticker con más datos |
| 📉 Pronóstico plano | Falta de estacionalidad real o `m` inadecuado | Cambia `m` o desactiva estacionalidad |

---

## 🧱 Estructura sugerida del proyecto

```
.
├── app.py
├── requirements.txt
├── README.md
└── datos/
    └── market_prices.csv
```

---

## 🧰 Mantenimiento

- Si cambias mucho de dataset, ejecuta:
  ```bash
  streamlit cache clear
  ```
- Si quieres más detalle en los logs, comenta la línea:
  ```python
  warnings.filterwarnings("ignore")
  ```

---

## 📝 Licencia

Puedes usar este código libremente con atribución (por ejemplo bajo licencia MIT).

---
