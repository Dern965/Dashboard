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
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt

# ===================== CONFIG PÁGINA =====================
st.set_page_config(page_title="EDA + ARIMA (BMV) • Multi-ticker (modo explicativo)",
                   layout="wide")

# ===================== PARÁMETROS FIJOS (ruta/columnas) =====================
DATA_PATH   = "datos/market_prices.csv"   # CSV fijo
DATE_COL    = "date"                      # columna fecha
TICKER_COL  = "instrument_id"             # ticker/instrumento
PRICE_COL   = "adj_close"                 # precio ajustado

# ===================== UTILIDADES BÁSICAS =====================
@st.cache_data
def load_prices(path, date_col, ticker_col, price_col):
    df = pd.read_csv(path, parse_dates=[date_col])
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
    if not seasonal or m <= 1:
        return 0
    try:
        a = acf(y.dropna(), nlags=max(m, 2), fft=True)
        return int(abs(a[m]) > 0.4)  # umbral simple
    except Exception:
        return 0

def fit_forecast_arima(y, h, seasonal, m):
    # Asegura frecuencia explícita
    if y.index.inferred_freq is None:
        inferred = pd.infer_freq(y.index)
        if inferred is None:
            inferred = "D"
        y = y.asfreq(inferred)

    # Heurísticas d/D
    d = infer_d(y)
    D = infer_D(y, m, seasonal=seasonal)

    p_grid = [0, 1, 2]
    q_grid = [0, 1, 2]
    P_grid = [0, 1] if (seasonal and m > 1) else [0]
    Q_grid = [0, 1] if (seasonal and m > 1) else [0]

    best_aic = np.inf
    best_res, best_spec = None, None

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

    if best_res is None:
        order = (1, max(1, d), 1)
        seas_order = (1, D, 1, m) if (seasonal and m > 1) else (0, 0, 0, 0)
        mod = SARIMAX(y, order=order, seasonal_order=seas_order,
                      enforce_stationarity=False, enforce_invertibility=False)
        best_res = mod.fit(disp=False)
        best_spec = {"order": order, "seasonal_order": seas_order, "aic": float(best_res.aic)}

    fitted = best_res.fittedvalues.rename("Ajuste (in-sample)")
    fc_obj = best_res.get_forecast(steps=h)
    fc_mean = fc_obj.predicted_mean.rename("Pronóstico")
    ci = fc_obj.conf_int(alpha=0.05)
    ci.columns = ["lower", "upper"]
    resid = best_res.resid.rename("Residuales")

    meta = best_spec
    return fitted, fc_mean, resid, meta, ci

def plot_ts(y, fitted=None, fcst=None, ci=None, title="Serie y pronóstico", ylab="Valor"):
    fig = go.Figure()
    fig.add_scatter(x=y.index, y=y.values, name="Observado", mode="lines")
    if fitted is not None:
        fig.add_scatter(x=fitted.index, y=fitted.values, name="Ajuste (dentro de muestra)", mode="lines")
    if fcst is not None:
        fig.add_scatter(x=fcst.index, y=fcst.values, name="Pronóstico (fuera de muestra)", mode="lines")
    if ci is not None and len(ci) > 0:
        fig.add_scatter(x=ci.index, y=ci["upper"].values, name="IC95% (sup)",
                        mode="lines", line=dict(width=0), showlegend=False)
        fig.add_scatter(x=ci.index, y=ci["lower"].values, name="IC95% (inf)",
                        mode="lines", line=dict(width=0), fill='tonexty',
                        fillcolor='rgba(0,0,0,0.12)', showlegend=False)
    fig.update_layout(title=title, xaxis_title="Fecha", yaxis_title=ylab)
    return fig

def plot_stl(y, period):
    stl = STL(y, period=period, robust=True).fit()
    comp = pd.DataFrame({
        "observado": y,
        "tendencia": stl.trend,
        "estacional": stl.seasonal,
        "residuo": stl.resid
    })
    comp = comp.reset_index().rename(columns={"index": "date"})
    fig = px.line(comp, x="date",
                  y=["observado", "tendencia", "estacional", "residuo"],
                  labels={"value": "", "date": "Fecha", "variable": "Componente"},
                  title="Descomposición STL: ¿qué parte es tendencia, estacionalidad y ruido?")
    return fig, comp.set_index("date")

# ===================== INTERPRETADORES (texto para humanos) =====================
def explain_adf(adf_dict, transform_name):
    p = adf_dict.get("p-value", np.nan)
    if np.isnan(p):
        return "No se pudo calcular la prueba ADF."
    if p < 0.05:
        return (f"ADF p={p:.3f} → **Estacionaria**. "
                f"Con esta transformación (“{transform_name}”) la serie no muestra raíz unitaria; "
                f"se puede modelar sin diferenciar (o con poca diferencia).")
    else:
        return (f"ADF p={p:.3f} → **No estacionaria**. "
                f"La serie necesita diferencia(s) o una transformación distinta para estabilizar nivel/varianza.")

def explain_stl_stats(comp):
    var_total = np.var(comp["observado"].values, ddof=1)
    parts = {}
    for c in ["tendencia","estacional","residuo"]:
        parts[c] = np.var(comp[c].values, ddof=1) / var_total if var_total > 0 else np.nan
    msg = "Aporte a la variabilidad: "
    msg += " | ".join([
        f"Tendencia ≈ {parts['tendencia']*100:.1f}%",
        f"Estacionalidad ≈ {parts['estacional']*100:.1f}%",
        f"Ruido ≈ {parts['residuo']*100:.1f}%"
    ])
    return msg

def explain_acf_pacf(y):
    # lags "marcados" por fuera de 95% aprox.
    n = len(y)
    if n < 10:
        return "Serie muy corta para interpretar ACF/PACF."
    conf = 1.96/np.sqrt(n)
    acfs = acf(y, nlags=min(24, n//2), fft=True)
    pacfs = pacf(y, nlags=min(24, n//2), method="ywm")
    sig_acf = [i for i,a in enumerate(acfs) if i>0 and abs(a)>conf]
    sig_pacf = [i for i,a in enumerate(pacfs) if i>0 and abs(a)>conf]
    hint = []
    if sig_acf[:1]:
        hint.append(f"ACF con picos en lags {sig_acf[:3]} → posible componente MA.")
    if sig_pacf[:1]:
        hint.append(f"PACF con picos en lags {sig_pacf[:3]} → posible componente AR.")
    if not hint:
        hint = ["Sin picos fuertes → modelo simple (p y q bajos) podría bastar."]
    return " ".join(hint)

def explain_residuals(resid):
    desc = resid.describe()[["mean","std","min","max"]].to_dict()
    mean_ok = abs(desc["mean"]) < (0.05 * desc["std"] if desc["std"]>0 else 1e-6)
    lb = acorr_ljungbox(resid.dropna(), lags=[10], return_df=True)
    p_lb = float(lb["lb_pvalue"].iloc[-1])
    parts = []
    parts.append(f"Media ≈ {desc['mean']:.4f} (std {desc['std']:.4f}) → "
                 + ("centrada en 0." if mean_ok else "sesgo leve."))
    parts.append(f"Ljung–Box p={p_lb:.3f} → "
                 + ("sin autocorrelación fuerte residual." if p_lb>0.05 else "queda autocorrelación en residuos."))
    return " ".join(parts)

def explain_model_meta(meta, freq_label, h):
    order = meta.get("order", (np.nan, np.nan, np.nan))
    seas = meta.get("seasonal_order", (0,0,0,0))
    aic  = meta.get("aic", np.nan)
    p,d,q = order
    P,D,Q,m = seas
    if m and m>1:
        seas_txt = f"con estacionalidad ({P},{D},{Q}) cada {m} {freq_label}."
    else:
        seas_txt = "sin componente estacional."
    return (f"Modelo elegido: ARIMA({p},{d},{q}) {seas_txt} "
            f"AIC={aic:.1f}. Se pronostran **{h}** pasos hacia delante.")

def explain_perf_table(perf_df):
    if perf_df.empty:
        return "Sin suficientes datos para comparar desempeño."
    # mejor Sharpe
    t_best = perf_df["Sharpe (rf=0)"].idxmax()
    best_row = perf_df.loc[t_best]
    return (f"En los últimos 3 años (retornos mensuales): **{t_best}** destaca por Sharpe={best_row['Sharpe (rf=0)']:.2f}. "
            f"Retorno anual ≈ {best_row['Retorno anual (%)']:.1f}% con volatilidad ≈ {best_row['Vol anual (%)']:.1f}%.")

def explain_top5_row(r, h):
    return (f"**{r['ticker']}**: precio actual {r['last_price']}, "
            f"pronóstico {r['forecast_price']} en {h} pasos → retorno esperado **{r['expected_ret_%']}%**; "
            f"riesgo (ancho IC95% relativo) **{r['risk_%']}%**. "
            f"**Score = {r['score']}** (ARIMA{r['order']} con estacional {r['seasonal_order']}, AIC {r['aic']}).")

# ===================== CARGA (automática) =====================
try:
    raw = load_prices(DATA_PATH, DATE_COL, TICKER_COL, PRICE_COL)
except Exception as e:
    st.error(f"No pude leer el CSV en '{DATA_PATH}': {e}")
    st.stop()

# ===================== CONTROLES =====================
st.sidebar.header("Parámetros de análisis")
freq = st.sidebar.selectbox("Frecuencia de análisis", ["D","W","M","B","Q"], index=2,
                            help="Intervalo temporal al que se agregan los precios (último valor de cada periodo).")
transform = st.sidebar.selectbox("Transformación para análisis",
                                 ["Precio", "Log-precio", "Retorno (%)", "Retorno log (%)"], index=0,
                                 help="Los retornos (% o log) estabilizan la varianza y facilitan pruebas/modelos.")
seasonal = st.sidebar.checkbox("¿Considerar estacionalidad? (SARIMA/STL)", value=True)
m_default = {"D":7, "W":52, "B":5, "M":12, "Q":4}[freq]
m = st.sidebar.number_input("Periodo estacional (m)", min_value=1, value=m_default, step=1,
                            help="Ej.: mensual=12, semanal=52, diario≈7 si hay patrón semanal.")
h = st.sidebar.number_input("Horizonte de pronóstico (pasos)", min_value=1, value=12, step=1)

# ===================== PREPROCESO =====================
wide = resample_wide(raw, freq=freq)
tickers_all = sorted([t for t in wide.columns if wide[t].dropna().shape[0] > m + 24])

st.title("EDA + ARIMA • Multi-ticker (BMV/MXN) — Modo explicativo")
st.caption("Panel interactivo con interpretaciones automáticas en español para cada resultado. "
           "Usa la barra lateral para ajustar frecuencia, transformación y horizonte.")

tabs = st.tabs(["Resumen multi-ticker", "EDA por ticker", "Pronósticos ARIMA", "Ranking de inversión (35)"])

# ---------- TAB 1: Resumen multi-ticker ----------
with tabs[0]:
    st.subheader("1) Vista rápida de precios por emisora")
    st.write(
        "Gráfica histórica tras el remuestreo elegido. Usa las casillas para limitar el rango o "
        "poner todas las series en una escala comparable."
    )

    if not tickers_all:
        st.info("No hay suficientes datos tras el remuestreo. Revisa la frecuencia elegida.")
    else:
        sel = st.multiselect(
            "Elige 1 o más tickers para visualizar",
            options=tickers_all,
            default=tickers_all[:2] if tickers_all else [],
        )

        if not sel:
            st.info("Selecciona al menos un ticker en el control superior.")
        else:
            # ====== Limpieza mínima para que la gráfica vaya fina ======
            tmp = wide[sel].copy()

            # 1) Asegura índice de fechas ordenado
            tmp = tmp.sort_index()
            if not isinstance(tmp.index, pd.DatetimeIndex):
                tmp.index = pd.to_datetime(tmp.index, errors="coerce")
            tmp = tmp.dropna(how="all")  # elimina filas totalmente vacías

            # 2) Opciones de vista
            copt1, copt2 = st.columns(2)
            with copt1:
                ultimos_3y = st.checkbox("Mostrar solo últimos 3 años", value=False)
            with copt2:
                normalizar = st.checkbox("Comparar en escala comparable (índice = 100)", value=False)

            if ultimos_3y and not tmp.empty:
                inicio = tmp.index.max() - pd.DateOffset(years=3)
                tmp = tmp.loc[tmp.index >= inicio]

            if normalizar and not tmp.empty:
                # Usa la primera fila disponible (tras ffill/bfill) como base
                base = tmp.ffill().bfill().iloc[0]
                tmp = (tmp.divide(base)) * 100

            # 3) ¡Gráfica como en el original!
            st.line_chart(tmp, use_container_width=True)

            # ====== Tabla comparativa (últimos 3 años, retornos mensuales) ======
            st.markdown("#### Desempeño comparativo (últimos 3 años, retornos **mensuales**)")
            wide_m = wide.resample("M").last()
            rets_m = wide_m[sel].pct_change().dropna()
            if len(rets_m) > 36:
                rets_m = rets_m.iloc[-36:]

            if rets_m.empty or rets_m.std().isna().all():
                st.info("Aún no hay suficientes meses con datos para calcular el comparativo.")
            else:
                ann = 12
                perf = pd.DataFrame({
                    "Retorno anual (%)": (rets_m.mean() * ann * 100),
                    "Vol anual (%)": (rets_m.std() * np.sqrt(ann) * 100),
                    "Sharpe (rf=0)": (rets_m.mean() / rets_m.std())
                }).round(2).dropna()
                st.dataframe(perf, use_container_width=True)

                if not perf.empty:
                    st.success(explain_perf_table(perf))

                with st.expander("¿Cómo leer estas métricas?"):
                    st.markdown(
                        "- **Retorno anual (%):** promedio de crecimiento al año suponiendo capitalización mensual.\n"
                        "- **Vol anual (%):** variación típica anual; a mayor vol, más riesgo.\n"
                        "- **Sharpe (rf=0):** retorno/volatilidad; mayor es mejor (ajusta por riesgo)."
                    )

# ---------- TAB 2: EDA ----------
with tabs[1]:
    st.subheader("2) Exploración de una serie (EDA)")
    st.caption("Aquí desglosamos **nivel/tendencia**, **estacionalidad** y **ruido**, y verificamos estacionariedad.")
    if not tickers_all:
        st.info("No hay emisoras suficientes para EDA.")
    else:
        t = st.selectbox("Ticker para EDA", options=tickers_all, index=0, key="eda_ticker")
        y = series_for_ticker(wide, t, transform).dropna()
        if y.empty:
            st.info("La serie está vacía tras la transformación. Cambia la opción de 'Transformación'.")
        else:
            c1, c2 = st.columns([2.2, 1])
            with c1:
                st.plotly_chart(
                    plot_ts(y, title=f"{t} • {transform}", ylab=transform),
                    use_container_width=True
                )
            with c2:
                st.markdown("**Resumen de la serie**")
                st.write(pd.Series({
                    "Observaciones": int(y.shape[0]),
                    "Inicio": y.index.min(),
                    "Fin": y.index.max(),
                    "Media": float(y.mean()),
                    "Desv. estándar": float(y.std())
                }).round(4))
                st.markdown("**Prueba ADF (estacionariedad)**")
                adf = adf_report(y)
                st.write(adf)
                st.info(explain_adf(adf, transform))

            st.markdown("### Descomposición STL (tendencia/estacionalidad/ruido)")
            try:
                fig_stl, comp = plot_stl(y, period=m if seasonal else max(2, m))
                st.plotly_chart(fig_stl, use_container_width=True)
                st.success(explain_stl_stats(comp))
            except Exception as e:
                st.warning(f"No se pudo ejecutar STL: {e}")

            st.markdown("### ACF y PACF (huellas AR/MA)")
            c3, c4 = st.columns(2)
            with c3:
                fig1, ax1 = plt.subplots(figsize=(6, 3))
                plot_acf(y.dropna(), ax=ax1, lags=min(48, len(y)//2))
                ax1.set_title("ACF: autocorrelación por rezago")
                st.pyplot(fig1)
            with c4:
                fig2, ax2 = plt.subplots(figsize=(6, 3))
                plot_pacf(y.dropna(), ax=ax2, lags=min(48, len(y)//2), method="ywm")
                ax2.set_title("PACF: autocorrelación parcial por rezago")
                st.pyplot(fig2)
            st.info(explain_acf_pacf(y))

            st.markdown("### Residuales de STL (ruido)")
            try:
                st.line_chart(comp["residuo"], height=160)
            except Exception:
                st.info("No hay residuales disponibles para mostrar.")

            with st.expander("Glosario rápido"):
                st.markdown(
                    "- **Estacionaria:** sus propiedades no cambian en el tiempo (media/varianza estables).\n"
                    "- **Tendencia:** movimientos de largo plazo (sube/baja sostenidamente).\n"
                    "- **Estacionalidad:** patrón repetitivo en ciclos (mensual, semanal, etc.).\n"
                    "- **Ruido:** variación aleatoria no explicada por tendencia/estacionalidad."
                )

# ---------- TAB 3: ARIMA (individual) ----------
with tabs[2]:
    st.subheader("3) Pronósticos ARIMA con explicación")
    st.caption("Selecciona una emisora; el modelo se elige por AIC sobre una cuadricula pequeña. "
               "Se muestran pronóstico e intervalos de confianza (incertidumbre).")
    if not tickers_all:
        st.info("No hay emisoras suficientes para modelar.")
    else:
        t2 = st.selectbox("Ticker para ARIMA", options=tickers_all, index=0, key="arima_ticker")
        y2 = series_for_ticker(wide, t2, transform).dropna()
        if y2.empty:
            st.info("La serie está vacía tras la transformación. Cambia 'Transformación'.")
        else:
            # Frecuencia explícita
            if y2.index.inferred_freq is None:
                inferred = pd.infer_freq(y2.index)
                if inferred is None:
                    inferred = "M"
                y2 = y2.asfreq(inferred)

            fitted, fcst, resid, meta, ci = fit_forecast_arima(y2, h=h, seasonal=seasonal, m=m)

            freq_label = {"D":"días","W":"semanas","B":"días hábiles","M":"meses","Q":"trimestres"}[freq]
            st.success(explain_model_meta(meta, freq_label, h))

            st.plotly_chart(
                plot_ts(y2, fitted, fcst, ci=ci,
                        title=f"{t2} • {transform} + SARIMAX (con IC95%)",
                        ylab=transform),
                use_container_width=True
            )

            st.markdown("#### Diagnóstico de residuales")
            st.line_chart(resid, height=160)
            st.info(explain_residuals(resid))

            st.markdown("#### Error del ajuste (dentro de muestra)")
            common_idx = y2.index.intersection(fitted.index)
            rmse = float(np.sqrt(np.mean((y2.loc[common_idx] - fitted.loc[common_idx])**2)))
            mae = float(np.mean(np.abs(y2.loc[common_idx] - fitted.loc[common_idx])))
            st.write(pd.Series({"RMSE (in-sample)": rmse, "MAE (in-sample)": mae}).round(4))
            with st.expander("¿Cómo leer RMSE/MAE?"):
                st.markdown(
                    "- **MAE:** error medio absoluto; en unidades de la serie (o % si usaste retornos).\n"
                    "- **RMSE:** penaliza más los errores grandes. Ambos comparan ajuste vs. realidad."
                )

# ---------- TAB 4: Ranking inversión (35) ----------
with tabs[3]:
    st.subheader("4) Ranking de inversión (ARIMA batch en 35 emisoras)")
    st.caption(
        "El **Score** resume atractivo esperado ajustado por riesgo del último paso del pronóstico.\n\n"
        "**Fórmula:** `Score = retorno_esperado(%) − 0.5 × riesgo(%)`.\n"
        "- **Retorno esperado (%):** cambio % entre el precio pronosticado y el precio actual.\n"
        "- **Riesgo (%):** ancho del intervalo de confianza 95% relativo al precio (incertidumbre).\n"
        "- El penalizador 0.5 puedes ajustarlo en el código según tu perfil de riesgo."
    )

    if len(tickers_all) == 0:
        st.info("No hay emisoras suficientes para correr el batch.")
    else:
        risk_penalty = 0.5
        results, failed = [], []

        for t in tickers_all:
            try:
                yp = series_for_ticker(wide, t, "Precio").dropna()
                if yp.empty:
                    raise ValueError("Serie vacía")
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
            st.markdown("#### Tabla completa (ordenada por **Score**)")
            st.dataframe(df_rank, use_container_width=True)

            top5 = df_rank.head(5).copy()
            st.markdown("### 🏆 Top 5 (mayor Score)")
            st.dataframe(top5[["ticker","last_price","forecast_price","expected_ret_%","risk_%",
                               "score","order","seasonal_order","aic"]],
                         use_container_width=True)

            fig_bar = go.Figure()
            fig_bar.add_bar(x=top5["ticker"], y=top5["expected_ret_%"], name="Retorno esperado (%)")
            fig_bar.add_bar(x=top5["ticker"], y=top5["risk_%"], name="Riesgo (%)")
            fig_bar.update_layout(barmode="group",
                                  title="Top 5: Retorno esperado vs Riesgo (IC95%)",
                                  xaxis_title="Ticker", yaxis_title="%")
            st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("### ¿Por qué estos 5?")
            for _, r in top5.iterrows():
                st.markdown(f"- {explain_top5_row(r, h)}")

            if failed:
                with st.expander("⚠️ Tickers con errores al modelar (detalle)"):
                    st.write(pd.DataFrame(failed, columns=["ticker","error"]))
