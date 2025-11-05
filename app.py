# app.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# --- stats & ts ---
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

st.set_page_config(page_title="EDA + ARIMA (BMV) • Multi-ticker", layout="wide")

# ===================== UTILIDADES =====================
@st.cache_data
def load_prices(path, date_col, ticker_col, price_col):
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.rename(columns={date_col: "date", ticker_col: "instrument_id", price_col: "adj_close"})
    df = df[["date", "instrument_id", "adj_close"]].dropna().sort_values("date")
    return df

def resample_wide(df, freq="D"):
    wide = df.pivot(index="date", columns="instrument_id", values="adj_close").sort_index()
    rule = {"D": "D", "W": "W", "B": "B", "M": "M", "Q": "Q"}[freq]
    if rule != "D":
        wide = wide.resample(rule).last()
    return wide

def series_for_ticker(wide, ticker, transform):
    s = wide[ticker].dropna()
    if transform == "Precio":
        return s
    if transform == "Log-precio":
        return np.log(s.replace(0, np.nan)).dropna()
    if transform == "Retorno (%)":
        return s.pct_change().dropna() * 100
    if transform == "Retorno log (%)":
        return (np.log(s).diff().dropna()) * 100
    return s

def adf_report(s):
    try:
        stat, p, *_ = adfuller(s.dropna(), autolag="AIC")
        return {"ADF": float(stat), "p-value": float(p)}
    except Exception:
        return {"ADF": np.nan, "p-value": np.nan}

def infer_d(y, max_d=2):
    """Determina d con ADF: si no rechaza raíz unitaria, difiere y vuelve a probar."""
    y_ = y.copy().dropna()
    for d in range(max_d + 1):
        try:
            p = adfuller(y_, autolag="AIC")[1]
        except Exception:
            return d
        if p < 0.05:
            return d
        y_ = y_.diff().dropna()
    return max_d

def infer_D(y, m, seasonal=True):
    """Heurística: si la ACF en lag m es fuerte, usar D=1."""
    if not seasonal or m <= 1:
        return 0
    try:
        a = acf(y.dropna(), nlags=max(m, 2), fft=True)
        return int(abs(a[m]) > 0.4)  # umbral sencillo
    except Exception:
        return 0

def fit_forecast_arima(y, h, seasonal, m):
    """
    y: pd.Series index datetime con freq explícita
    h: horizonte (pasos)
    seasonal: bool
    m: periodo estacional
    """
    # Asegura frecuencia explícita
    if y.index.inferred_freq is None:
        inferred = pd.infer_freq(y.index)
        if inferred is None:
            # forzar con la diferencia más común
            inferred = "D"
        y = y.asfreq(inferred)

    # Heurísticas para d/D
    d = infer_d(y)
    D = infer_D(y, m, seasonal=seasonal)

    # Cuadrícula pequeña (rápida en Streamlit)
    p_grid = [0, 1, 2]
    q_grid = [0, 1, 2]
    P_grid = [0, 1] if (seasonal and m > 1) else [0]
    Q_grid = [0, 1] if (seasonal and m > 1) else [0]

    best_aic = np.inf
    best_res = None
    best_spec = None

    for p in p_grid:
        for q in q_grid:
            for P in P_grid:
                for Q in Q_grid:
                    order = (p, d, q)
                    seas_order = (P, D, Q, m) if (seasonal and m > 1) else (0, 0, 0, 0)
                    try:
                        mod = SARIMAX(
                            y,
                            order=order,
                            seasonal_order=seas_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        )
                        res = mod.fit(disp=False)
                        if res.aic < best_aic:
                            best_aic = res.aic
                            best_res = res
                            best_spec = {"order": order, "seasonal_order": seas_order, "aic": float(res.aic)}
                    except Exception:
                        continue

    # Fallback si nada funcionó
    if best_res is None:
        order = (1, max(1, d), 1)
        seas_order = (1, D, 1, m) if (seasonal and m > 1) else (0, 0, 0, 0)
        mod = SARIMAX(y, order=order, seasonal_order=seas_order,
                      enforce_stationarity=False, enforce_invertibility=False)
        best_res = mod.fit(disp=False)
        best_spec = {"order": order, "seasonal_order": seas_order, "aic": float(best_res.aic)}

    # Ajustes y pronóstico con bandas
    fitted = best_res.fittedvalues.rename("fitted")
    fc_obj = best_res.get_forecast(steps=h)
    fc_mean = fc_obj.predicted_mean.rename("forecast")
    ci = fc_obj.conf_int(alpha=0.05)  # 95%
    ci.columns = ["lower", "upper"]
    resid = best_res.resid.rename("resid")

    meta = best_spec
    return fitted, fc_mean, resid, meta, ci

def plot_ts(y, fitted=None, fcst=None, ci=None, title="Serie y pronóstico"):
    fig = go.Figure()
    fig.add_scatter(x=y.index, y=y.values, name="Observado", mode="lines")
    if fitted is not None:
        fig.add_scatter(x=fitted.index, y=fitted.values, name="Ajuste (in-sample)", mode="lines")
    if fcst is not None:
        fig.add_scatter(x=fcst.index, y=fcst.values, name="Pronóstico", mode="lines")
    if ci is not None and len(ci) > 0:
        fig.add_scatter(x=ci.index, y=ci["upper"].values, name="IC 95% (sup)",
                        mode="lines", line=dict(width=0), showlegend=False)
        fig.add_scatter(x=ci.index, y=ci["lower"].values, name="IC 95% (inf)",
                        mode="lines", line=dict(width=0), fill='tonexty',
                        fillcolor='rgba(0,0,0,0.1)', showlegend=False)
    fig.update_layout(title=title, xaxis_title="", yaxis_title="")
    return fig

def plot_stl(y, period):
    stl = STL(y, period=period, robust=True).fit()
    comp = pd.DataFrame({
        "observed": y,
        "trend": stl.trend,
        "seasonal": stl.seasonal,
        "resid": stl.resid
    })
    comp = comp.reset_index().rename(columns={"index": "date"})
    fig = px.line(comp, x="date", y=["observed", "trend", "seasonal", "resid"],
                  labels={"value": "", "date": "Fecha", "variable": "Componente"},
                  title="Descomposición STL")
    return fig, comp.set_index("date")

# ===================== SIDEBAR =====================
st.sidebar.header("Datos")
path = st.sidebar.text_input("Ruta del CSV", "datos/market_prices.csv")
st.sidebar.caption("Columnas esperadas: date, ticker, adj_close")

with st.sidebar.expander("Mapeo de columnas (si tu CSV difiere)"):
    col_date = st.text_input("Columna fecha", value="date")
    col_ticker = st.text_input("Columna ticker", value="ticker")
    col_price = st.text_input("Columna precio ajustado", value="adj_close")

freq = st.sidebar.selectbox("Frecuencia de análisis", ["D","W","M","B","Q"], index=2)
transform = st.sidebar.selectbox("Transformación para EDA/ARIMA",
                                 ["Precio", "Log-precio", "Retorno (%)", "Retorno log (%)"], index=0)

# Estacionalidad
seasonal = st.sidebar.checkbox("Estacional (SARIMA/STL)", value=True)
m_default = {"D":7, "W":52, "B":5, "M":12, "Q":4}[freq]
m = st.sidebar.number_input("Periodo estacional (m)", min_value=1, value=m_default, step=1)

# Pronóstico
h = st.sidebar.number_input("Horizonte de pronóstico (pasos)", min_value=1, value=12, step=1)

# ===================== CARGA =====================
try:
    raw = load_prices(path, col_date, col_ticker, col_price)
except Exception as e:
    st.error(f"No pude leer el CSV: {e}")
    st.stop()

wide = resample_wide(raw, freq=freq)
tickers = sorted([t for t in wide.columns if wide[t].dropna().shape[0] > m + 24])

st.sidebar.header("Selección de tickers")
sel = st.sidebar.multiselect("Elige 1 o más tickers", options=tickers, default=tickers[:2] if tickers else [])

if not sel:
    st.warning("Selecciona al menos un ticker.")
    st.stop()

# ===================== LAYOUT PRINCIPAL =====================
st.title("EDA + ARIMA • Multi-ticker (BMV/MXN)")

tabs = st.tabs(["Resumen multi-ticker", "EDA por ticker", "ARIMA por ticker"])

# ---------- TAB 1: Resumen multi-ticker ----------
with tabs[0]:
    st.subheader("Vista rápida de precios (resampleados)")
    st.line_chart(wide[sel])

    # métricas comparativas (retorno y vol anualizadas sobre mensual)
    st.markdown("**Stats comparativas (últimos 3 años, con retornos mensuales)**")
    wide_m = wide.resample("M").last()
    rets_m = wide_m[sel].pct_change().dropna()
    if len(rets_m) > 36:
        rets_m = rets_m.iloc[-36:]
    ann = 12
    perf = pd.DataFrame({
        "Retorno anual (%)": (rets_m.mean() * ann * 100),
        "Vol anual (%)": (rets_m.std() * np.sqrt(ann) * 100),
        "Sharpe (rf=0)": (rets_m.mean() / rets_m.std())
    }).round(2).dropna()
    st.dataframe(perf)

# ---------- TAB 2: EDA ----------
with tabs[1]:
    t = st.selectbox("Ticker para EDA", options=sel, index=0, key="eda_ticker")
    y = series_for_ticker(wide, t, transform).dropna()
    if y.empty:
        st.info("La serie está vacía tras la transformación. Cambia 'Transformación'.")
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader(f"{t} • Serie transformada")
            st.line_chart(y)
        with c2:
            st.subheader("Resumen")
            st.write(pd.Series({
                "Observaciones": int(y.shape[0]),
                "Inicio": y.index.min(),
                "Fin": y.index.max(),
                "Media": float(y.mean()),
                "Std": float(y.std())
            }).round(4))
            st.write("**ADF (estacionariedad)**")
            st.write(adf_report(y))

        # STL
        st.markdown("### Descomposición STL")
        try:
            fig_stl, comp = plot_stl(y, period=m if seasonal else max(2, m))
            st.plotly_chart(fig_stl, use_container_width=True)
        except Exception as e:
            st.info(f"No se pudo ejecutar STL: {e}")

        # ACF y PACF
        st.markdown("### ACF y PACF")
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        plot_acf(y.dropna(), ax=ax1, lags=min(48, len(y)//2))
        st.pyplot(fig1)
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        plot_pacf(y.dropna(), ax=ax2, lags=min(48, len(y)//2), method="ywm")
        st.pyplot(fig2)

        # residuales de STL
        st.markdown("### Residuales (STL)")
        if "resid" in comp.columns:
            st.line_chart(comp["resid"])
        else:
            st.info("No hay residuales disponibles para mostrar.")

# ---------- TAB 3: ARIMA ----------
with tabs[2]:
    t2 = st.selectbox("Ticker para ARIMA", options=sel, index=0, key="arima_ticker")
    y2 = series_for_ticker(wide, t2, transform).dropna()
    if y2.empty:
        st.info("La serie está vacía tras la transformación. Cambia 'Transformación'.")
    else:
        # Asegura frecuencia explícita
        if y2.index.inferred_freq is None:
            inferred = pd.infer_freq(y2.index)
            if inferred is None:
                inferred = "M"  # fallback razonable
            y2 = y2.asfreq(inferred)

        fitted, fcst, resid, meta, ci = fit_forecast_arima(y2, h=h, seasonal=seasonal, m=m)

        st.write(f"**Modelo seleccionado**: {meta}")
        st.plotly_chart(
            plot_ts(y2, fitted, fcst, ci=ci, title=f"{t2} • {transform} + SARIMAX (statsmodels)"),
            use_container_width=True
        )

        # Residuales
        st.markdown("### Residuales del ajuste")
        st.line_chart(resid)

        # Métricas simples in-sample
        common_idx = y2.index.intersection(fitted.index)
        rmse = float(np.sqrt(np.mean((y2.loc[common_idx] - fitted.loc[common_idx])**2)))
        mae = float(np.mean(np.abs(y2.loc[common_idx] - fitted.loc[common_idx])))
        st.write(pd.Series({"RMSE (in-sample)": rmse, "MAE (in-sample)": mae}).round(4))
