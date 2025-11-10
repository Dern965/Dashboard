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

# ===================== PARÁMETROS FIJOS (ruta/columnas) =====================
DATA_PATH   = "datos/market_prices.csv"  # <-- tu CSV ya fijo
DATE_COL    = "date"                     # <-- columna fecha en tu CSV
TICKER_COL  = "instrument_id"                   # <-- (o "instrument_id" si así viene)
PRICE_COL   = "adj_close"                # <-- precio ajustado

# ===================== UTILIDADES =====================
@st.cache_data
def load_prices(path, date_col, ticker_col, price_col):
    df = pd.read_csv(path, parse_dates=[date_col])
    # renombra a nombres estándar internos
    df = df.rename(columns={date_col: "date", ticker_col: "instrument_id", price_col: "adj_close"})
    df = df[["date", "instrument_id", "adj_close"]].dropna().sort_values(["instrument_id","date"])
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
            # fallback razonable a diario
            inferred = "D"
        y = y.asfreq(inferred)

    # Heurísticas para d/D
    d = infer_d(y)
    D = infer_D(y, m, seasonal=seasonal)

    # Cuadrícula pequeña (rápida)
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

# ===================== CARGA (automática, sin inputs) =====================
try:
    raw = load_prices(DATA_PATH, DATE_COL, TICKER_COL, PRICE_COL)
except Exception as e:
    st.error(f"No pude leer el CSV en '{DATA_PATH}': {e}")
    st.stop()

# ===================== CONTROLES (solo análisis) =====================
st.sidebar.header("Parámetros de análisis")
freq = st.sidebar.selectbox("Frecuencia de análisis", ["D","W","M","B","Q"], index=2)
transform = st.sidebar.selectbox("Transformación para EDA/ARIMA",
                                 ["Precio", "Log-precio", "Retorno (%)", "Retorno log (%)"], index=0)
seasonal = st.sidebar.checkbox("Estacional (SARIMA/STL)", value=True)
m_default = {"D":7, "W":52, "B":5, "M":12, "Q":4}[freq]
m = st.sidebar.number_input("Periodo estacional (m)", min_value=1, value=m_default, step=1)
h = st.sidebar.number_input("Horizonte de pronóstico (pasos)", min_value=1, value=12, step=1)

# ===================== PREPROCESO =====================
wide = resample_wide(raw, freq=freq)
tickers_all = sorted([t for t in wide.columns if wide[t].dropna().shape[0] > m + 24])

st.sidebar.header("Selección de tickers (para pestañas 1–3)")
sel = st.sidebar.multiselect("Elige 1 o más tickers", options=tickers_all, default=tickers_all[:2] if tickers_all else [])

st.title("EDA + ARIMA • Multi-ticker (BMV/MXN)")
tabs = st.tabs(["Resumen multi-ticker", "EDA por ticker", "ARIMA por ticker", "Ranking inversión (35)"])

# ---------- TAB 1: Resumen multi-ticker ----------
with tabs[0]:
    if not sel:
        st.info("Selecciona al menos un ticker en la barra lateral.")
    else:
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
    if not sel:
        st.info("Selecciona al menos un ticker en la barra lateral.")
    else:
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
            try:
                st.line_chart(comp["resid"])
            except Exception:
                st.info("No hay residuales disponibles para mostrar.")

# ---------- TAB 3: ARIMA (individual) ----------
with tabs[2]:
    if not sel:
        st.info("Selecciona al menos un ticker en la barra lateral.")
    else:
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

# ---------- TAB 4: Ranking inversión (35) ----------
with tabs[3]:
    st.subheader("ARIMA batch en 35 emisoras + Top 5 de inversión")
    st.caption("El ranking se basa en el **retorno esperado (%)** del último paso del pronóstico, penalizado por **riesgo (%)** (ancho del IC95% relativo). Score = retorno − 0.5 × riesgo.")

    if len(tickers_all) == 0:
        st.info("No hay emisoras suficientes para correr el batch.")
    else:
        risk_penalty = 0.5  # puedes ajustar este penalizador

        results = []
        failed = []

        # Usamos SIEMPRE Precio para comparabilidad en ranking
        for t in tickers_all:
            try:
                yp = series_for_ticker(wide, t, "Precio").dropna()
                if yp.empty:
                    raise ValueError("Serie vacía")
                # Asegura frecuencia
                if yp.index.inferred_freq is None:
                    inf = pd.infer_freq(yp.index)
                    if inf is None:
                        inf = "M"
                    yp = yp.asfreq(inf)

                fitted, fcst, resid, meta, ci = fit_forecast_arima(yp, h=h, seasonal=seasonal, m=m)

                last_price = float(yp.iloc[-1])
                last_fc = float(fcst.iloc[-1])

                if last_price == 0 or np.isnan(last_price) or np.isnan(last_fc):
                    raise ValueError("Valores inválidos para retorno")

                expected_ret_pct = (last_fc - last_price) / abs(last_price) * 100.0

                # riesgo: ancho del IC95% en el último paso relativo al precio
                last_ci = ci.iloc[-1]
                width = float(last_ci["upper"] - last_ci["lower"])
                risk_pct = (width / (2.0 * abs(last_price))) * 100.0 if last_price != 0 else np.nan

                score = expected_ret_pct - risk_penalty * risk_pct

                results.append({
                    "ticker": t,
                    "last_price": round(last_price, 4),
                    "forecast_price": round(last_fc, 4),
                    "expected_ret_%": round(expected_ret_pct, 3),
                    "risk_%": round(risk_pct, 3),
                    "score": round(score, 3),
                    "order": meta.get("order"),
                    "seasonal_order": meta.get("seasonal_order"),
                    "aic": round(meta.get("aic", np.nan), 2)
                })
            except Exception as e:
                failed.append((t, str(e)))
                continue

        if len(results) == 0:
            st.error("No se pudieron generar pronósticos para las emisoras.")
        else:
            df_rank = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
            st.markdown("#### Tabla completa (ordenada por **score**)")
            st.dataframe(df_rank, use_container_width=True)

            top5 = df_rank.head(5).copy()
            st.markdown("### 🏆 Top 5 para invertir")
            st.dataframe(top5[["ticker","last_price","forecast_price","expected_ret_%","risk_%","score","order","seasonal_order","aic"]], use_container_width=True)

            # Visual rápido: barras de retorno esperado y riesgo
            fig_bar = go.Figure()
            fig_bar.add_bar(x=top5["ticker"], y=top5["expected_ret_%"], name="Retorno esperado (%)")
            fig_bar.add_bar(x=top5["ticker"], y=top5["risk_%"], name="Riesgo (%)")
            fig_bar.update_layout(barmode="group", title="Top 5: Retorno esperado vs Riesgo (IC95%)")
            st.plotly_chart(fig_bar, use_container_width=True)

            # Explicación corta por emisora
            st.markdown("### ¿Por qué estos 5?")
            for _, r in top5.iterrows():
                st.markdown(
                    f"- **{r['ticker']}**: se proyecta **{r['expected_ret_%']}%** en {h} pasos; "
                    f"riesgo relativo **{r['risk_%']}%** (IC95%). "
                    f"Score = {r['score']} (modelo {r['order']} estacional {r['seasonal_order']}, AIC {r['aic']})."
                )

            if failed:
                with st.expander("⚠️ Tickers con errores al modelar"):
                    st.write(pd.DataFrame(failed, columns=["ticker","error"]))
