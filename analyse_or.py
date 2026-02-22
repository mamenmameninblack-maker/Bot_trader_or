import pandas as pd
import numpy as np
import pytz
from datetime import datetime

# ============================================================
# INTEGRATION PEPPERSTONE (A1 — API RÉELLE UNIQUEMENT)
# ============================================================

from ctrader_api import fetch_candles_ctrader as fetch_candles

# ============================================================
# CONFIG ANALYSE
# ============================================================

CONFIG_ANALYSE = {
    "market_data": {
        "timeframes": ["M5", "M15", "H1"],
        "volume_window": [50, 150],
        "order_book_depth": [3, 5],
        "liquidity_threshold_pct": [0.3, 0.8],
        "volatility_window_hours": [2, 8],
        "funding_update_minutes": 60,
        "open_interest_change_strong_pct": [3, 5],
    },
    "quant_models": {
        "regression": {
            "window_points": [150, 400],
            "type": "ridge",
            "min_r2": 0.25,
            "alpha": 1.0,
        },
        "correlation": {
            "window_hours": [4, 24],
            "min_corr": 0.55,
        },
        "cointegration": {
            "adf_p_value_max": 0.05,
            "hedge_ratio_recalc_hours": [2, 4],
        },
        "mean_reversion": {
            "half_life_minutes": [30, 360],
            "z_entry": 1.0,
            "z_exit": 0.35,
        },
        "vol_clustering": {
            "model": "GARCH_1_1",
            "horizon_hours": [4, 12],
            "high_vol_multiplier": 1.5,
        },
    },
    "order_flow": {
        "imbalance_threshold": [0.55, 0.45],
        "aggressiveness_ratio_min": 1.2,
        "mkt_limit_ratio_momentum": 1.3,
        "absorption_multiplier": 2.0,
        "liquidity_depth_pct": [0.5, 1.0],
    },
    "indicators": {
        "adaptive_ma": {
            "window": [20, 50],
            "volatility_source": "ATR_14_20",
        },
        "rsi_vol_adj": {
            "period": [10, 14],
            "lower": 40,
            "upper": 60,
        },
        "filters": {
            "kalman_process_noise": [0.005, 0.02],
            "hp_lambda": [14400, 43200],
        },
    },
    "regimes": {
        "trend_slope_ema50_pct": [0.1, 0.2],
        "range_atr_pct": [0.4, 0.8],
        "high_vol_atr_pct": [1.0, 1.5],
        "breakout_volume_multiplier": [1.5, 2.0],
        "mean_reversion_z_max": 1.0,
    },
    "risk": {
        "risk_per_trade_pct": [0.3, 0.75],
        "daily_dd_max_pct": [1.5, 3.0],
        "sl_atr_multiplier": [1.5, 2.5],
        "tp_sl_rr": [1.5, 2.2],
        "cooldown_after_loss_hours": [1, 4],
        "max_trades_per_day": [3, 10],
    },
    "psychology": {
        "no_trade_big_candle_range_multiplier": [2.0, 3.0],
        "max_consecutive_losses": 3,
        "no_size_increase_after_loss": True,
        "news_blackout_minutes": [15, 30],
    },
}

TZ_PARIS = pytz.timezone("Europe/Paris")
# ============================================================
# INDICATEURS TECHNIQUES
# ============================================================

def ema(series, period):
    return pd.Series(series).ewm(span=period, adjust=False).mean()


def rsi(series, period=14):
    series = pd.Series(series)
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain = pd.Series(gain).rolling(period).mean()
    loss = pd.Series(loss).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def macd(series, fast=12, slow=26, signal=9):
    series = pd.Series(series)
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def stochastic(high, low, close, k=5, d=3, smooth=3):
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    lowest = low.rolling(k).min()
    highest = high.rolling(k).max()
    k_raw = 100 * (close - lowest) / (highest - lowest + 1e-9)
    k_smooth = k_raw.rolling(smooth).mean()
    d_line = k_smooth.rolling(d).mean()
    return k_smooth, d_line


def vwap(df):
    pv = df["close"] * df["volume"]
    return pv.cumsum() / (df["volume"].cumsum() + 1e-9)


def atr(df, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ============================================================
# SUPPORT / RÉSISTANCE
# ============================================================

def find_support_resistance(close, lookback=50):
    close = pd.Series(close).dropna().tail(lookback)
    if len(close) < 5:
        return float(close.min()), float(close.max())

    supports, resistances = [], []

    for i in range(2, len(close) - 2):
        if close.iloc[i] < close.iloc[i - 1] and close.iloc[i] < close.iloc[i + 1]:
            supports.append(close.iloc[i])
        if close.iloc[i] > close.iloc[i - 1] and close.iloc[i] > close.iloc[i + 1]:
            resistances.append(close.iloc[i])

    support = max(supports) if supports else float(close.min())
    resistance = min(resistances) if resistances else float(close.max())

    return round(support, 2), round(resistance, 2)


# ============================================================
# DÉTECTION DE PICS / CREUX
# ============================================================

def detect_peaks_valleys_rsi(rsi_vals):
    rsi_vals = pd.Series(rsi_vals).dropna()
    if len(rsi_vals) < 3:
        return 0
    if rsi_vals.iloc[-1] < rsi_vals.iloc[-2] and rsi_vals.iloc[-2] > rsi_vals.iloc[-3]:
        return -1
    if rsi_vals.iloc[-1] > rsi_vals.iloc[-2] and rsi_vals.iloc[-2] < rsi_vals.iloc[-3]:
        return +1
    return 0


def detect_peaks_valleys_stoch(k_vals):
    k_vals = pd.Series(k_vals).dropna()
    if len(k_vals) < 3:
        return 0
    if k_vals.iloc[-1] < k_vals.iloc[-2] and k_vals.iloc[-2] > k_vals.iloc[-3]:
        return -1
    if k_vals.iloc[-1] > k_vals.iloc[-2] and k_vals.iloc[-2] < k_vals.iloc[-3]:
        return +1
    return 0


def detect_peaks_valleys_macd(hist_vals):
    hist_vals = pd.Series(hist_vals).dropna()
    if len(hist_vals) < 3:
        return 0
    if hist_vals.iloc[-1] < hist_vals.iloc[-2] and hist_vals.iloc[-2] > hist_vals.iloc[-3]:
        return -1
    if hist_vals.iloc[-1] > hist_vals.iloc[-2] and hist_vals.iloc[-2] < hist_vals.iloc[-3]:
        return +1
    return 0


# ============================================================
# MODÈLES QUANTITATIFS
# ============================================================

def ridge_regression_signal(y, x=None, alpha=1.0, min_r2=0.25):
    y = pd.Series(y).dropna()
    n = len(y)

    if n < 20:
        return {"r2": 0.0, "slope": 0.0, "valid": False}

    if x is None:
        x = np.arange(n)

    x = np.array(x).reshape(-1, 1)
    y_arr = y.values.reshape(-1, 1)

    X = np.hstack([np.ones((n, 1)), x])
    I = np.eye(X.shape[1])
    I[0, 0] = 0

    beta = np.linalg.inv(X.T @ X + alpha * I) @ (X.T @ y_arr)
    y_pred = X @ beta

    ss_res = np.sum((y_arr - y_pred) ** 2)
    ss_tot = np.sum((y_arr - y_arr.mean()) ** 2)

    r2 = 1 - ss_res / (ss_tot + 1e-9)
    slope = float(beta[1])
    valid = r2 >= min_r2

    return {"r2": float(r2), "slope": slope, "valid": valid}


def compute_cointegration(series1, series2):
    s1 = pd.Series(series1).dropna()
    s2 = pd.Series(series2).dropna()

    n = min(len(s1), len(s2))
    if n < 50:
        return {"p_value": 1.0, "hedge_ratio": 0.0, "cointegrated": False}

    s1 = s1.tail(n)
    s2 = s2.tail(n)

    X = np.vstack([np.ones(n), s2.values]).T
    y = s1.values.reshape(-1, 1)

    beta = np.linalg.inv(X.T @ X + 1e-6 * np.eye(2)) @ (X.T @ y)
    hedge_ratio = float(beta[1])

    corr = np.corrcoef(s1, s2)[0, 1]
    p_value_approx = 1 - abs(corr)

    cointegrated = p_value_approx < CONFIG_ANALYSE["quant_models"]["cointegration"]["adf_p_value_max"]

    return {
        "p_value": float(p_value_approx),
        "hedge_ratio": hedge_ratio,
        "cointegrated": bool(cointegrated),
    }


def compute_mean_reversion_stats(spread):
    spread = pd.Series(spread).dropna()

    if len(spread) < 30:
        return {"half_life": None, "z_score": 0.0}

    spread_lag = spread.shift(1).dropna()
    spread_ret = spread.diff().dropna()

    X = np.vstack([np.ones(len(spread_lag)), spread_lag.values]).T
    y = spread_ret.values.reshape(-1, 1)

    beta = np.linalg.inv(X.T @ X + 1e-6 * np.eye(2)) @ (X.T @ y)
    b = float(beta[1])

    half_life = None if b >= 0 else -np.log(2) / b
    z_score = (spread - spread.mean()) / (spread.std() + 1e-9)

    return {"half_life": half_life, "z_score": float(z_score.iloc[-1])}


def compute_garch_volatility(returns, horizon=50):
    r = pd.Series(returns).dropna()

    if len(r) < 50:
        return {"current_vol": float(r.std()), "high_vol": False}

    r = r.tail(horizon)
    var = r.var()

    omega = 0.000001
    alpha = 0.05
    beta = 0.9

    sigma2 = var

    for ret in r:
        sigma2 = omega + alpha * (ret ** 2) + beta * sigma2

    current_vol = np.sqrt(sigma2)
    avg_vol = r.std()

    high_vol = current_vol > CONFIG_ANALYSE["quant_models"]["vol_clustering"]["high_vol_multiplier"] * avg_vol

    return {"current_vol": float(current_vol), "high_vol": bool(high_vol)}


def kalman_filter_1d(series, process_noise=0.01, measurement_noise=1.0):
    series = pd.Series(series).dropna()

    if len(series) == 0:
        return series

    x_est = series.iloc[0]
    p_est = 1.0
    estimates = []

    for z in series:
        x_pred = x_est
        p_pred = p_est + process_noise

        K = p_pred / (p_pred + measurement_noise)

        x_est = x_pred + K * (z - x_pred)
        p_est = (1 - K) * p_pred

        estimates.append(x_est)

    return pd.Series(estimates, index=series.index)


def apply_hp_filter(series, window=50):
    series = pd.Series(series).dropna()

    if len(series) < window:
        trend = series.rolling(len(series)).mean()
    else:
        trend = series.rolling(window).mean()

    cycle = series - trend

    return trend, cycle
# ============================================================
# MOMENTUM MULTI-TIMEFRAME
# ============================================================

def compute_mtf_momentum():
    total_score = 0
    notes = []
    timeframes = ["M5", "M15", "H1"]

    for tf in timeframes:
        df = fetch_candles("XAUUSD", tf, 200)

        if df is None or len(df) < 50:
            notes.append(f"{tf} : données insuffisantes (API non connectée)")
            continue

        close = df["close"]
        high = df["high"]
        low = df["low"]

        rsi_tf = rsi(close, 14)
        stoch_k, stoch_d = stochastic(high, low, close)
        macd_line, signal_line, hist = macd(close)

        score = 0
        score += detect_peaks_valleys_rsi(rsi_tf)
        score += detect_peaks_valleys_stoch(stoch_k)
        score += detect_peaks_valleys_macd(hist)

        total_score += score

        if score > 0:
            notes.append(f"{tf} : momentum haussier")
        elif score < 0:
            notes.append(f"{tf} : momentum baissier")
        else:
            notes.append(f"{tf} : momentum neutre")

    return total_score, notes


# ============================================================
# DÉTECTION DE RÉGIME
# ============================================================

def detect_regime(df):
    close = df["close"]
    ema50 = ema(close, 50)
    atr_val = atr(df, 14)

    if len(ema50) < 11:
        return "neutre", ["Pas assez de données pour détecter le régime"]

    last_close = close.iloc[-1]
    last_ema50 = ema50.iloc[-1]
    last_atr = atr_val.iloc[-1]

    slope_ema50 = ((ema50.iloc[-1] - ema50.iloc[-10]) /
                   (abs(ema50.iloc[-10]) + 1e-9)) * 100

    atr_pct = (last_atr / last_close) * 100

    reg_conf = CONFIG_ANALYSE["regimes"]

    regime = "neutre"
    notes = []

    if abs(slope_ema50) >= reg_conf["trend_slope_ema50_pct"][0]:
        regime = "trend"
        notes.append(f"Trend détecté (slope EMA50 = {slope_ema50:.3f}%)")

    if reg_conf["range_atr_pct"][0] <= atr_pct <= reg_conf["range_atr_pct"][1]:
        regime = "range"
        notes.append(f"Range détecté (ATR% = {atr_pct:.3f}%)")

    if atr_pct >= reg_conf["high_vol_atr_pct"][0]:
        regime = "high_vol"
        notes.append(f"High vol détecté (ATR% = {atr_pct:.3f}%)")

    vol = df["volume"]
    if len(vol) > 50:
        last_vol = vol.iloc[-1]
        avg_vol = vol.tail(50).mean()
        if last_vol > reg_conf["breakout_volume_multiplier"][0] * avg_vol:
            regime = "breakout"
            notes.append("Breakout de volume détecté")

    return regime, notes


# ============================================================
# ANALYSE PRINCIPALE XAU/USD
# ============================================================

def analyze_xauusd():
    df = fetch_candles("XAUUSD", "M5", 300)

    if df is None or len(df) < 100:
        return {
            "signal": "NEUTRE",
            "notes": ["Pas assez de données (Pepperstone non connecté)."],
            "regime": "inconnu",
            "risk_profile": None,
            "psychology_flags": {},
            "last_price": None,
            "entry_price": None,
            "trend": "neutre",
            "support": None,
            "resistance": None,
            "momentum_score": 0,
            "atr_value": None,
        }

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    ema20 = ema(close, 20)
    ema200 = ema(close, 200)
    rsi9 = rsi(close, 9)
    macd_line, signal_line, hist = macd(close)
    stoch_k, stoch_d = stochastic(high, low, close)
    vwap_j = vwap(df)
    atr_val = atr(df, 14)

    last_close = close.iloc[-1]
    last_ema200 = ema200.iloc[-1]
    last_rsi9 = rsi9.iloc[-1]
    last_hist = hist.iloc[-1]
    last_vwap = vwap_j.iloc[-1]
    last_atr = atr_val.iloc[-1]

    if last_close > last_ema200:
        trend = "haussière"
    elif last_close < last_ema200:
        trend = "baissière"
    else:
        trend = "neutre"

    support, resistance = find_support_resistance(close)

    momentum_score, mtf_notes = compute_mtf_momentum()

    regime, regime_notes = detect_regime(df)

    reg_conf = CONFIG_ANALYSE["quant_models"]["regression"]
    ridge_res = ridge_regression_signal(
        y=np.log(close.tail(reg_conf["window_points"][0])),
        alpha=reg_conf["alpha"],
        min_r2=reg_conf["min_r2"],
    )

    returns = close.pct_change().dropna()
    garch_res = compute_garch_volatility(returns)

    # 🔥 Cooldown volatilité extrême
    if garch_res["high_vol"]:
        return {
            "signal": "COOLDOWN_VOL",
            "notes": ["Volatilité extrême détectée (GARCH)."],
            "regime": "high_vol",
            "risk_profile": None,
            "psychology_flags": {"cooldown": True},
            "last_price": None,
            "entry_price": None,
            "trend": "neutre",
            "support": None,
            "resistance": None,
            "momentum_score": 0,
            "atr_value": None,
        }

    spread_vwap = close - vwap_j
    mr_stats = compute_mean_reversion_stats(spread_vwap)

    risk_profile = {
        "risk_per_trade_pct": CONFIG_ANALYSE["risk"]["risk_per_trade_pct"],
        "sl_atr_multiplier": CONFIG_ANALYSE["risk"]["sl_atr_multiplier"],
        "tp_sl_rr": CONFIG_ANALYSE["risk"]["tp_sl_rr"],
    }

    psychology_flags = {
        "anti_fomo": False,
        "cooldown": False,
    }

    last_range = high.iloc[-1] - low.iloc[-1]
    avg_range = (high - low).tail(50).mean()
    big_candle_mult = CONFIG_ANALYSE["psychology"]["no_trade_big_candle_range_multiplier"][0]

    if last_range > big_candle_mult * avg_range:
        psychology_flags["anti_fomo"] = True

    signal = "NEUTRE"
    entry_price = round(last_close, 2)
    notes = []

    if trend == "haussière" and momentum_score >= 2 and not psychology_flags["anti_fomo"]:
        signal = "ACHAT"
        entry_price = round(last_close * 0.998, 2)
        notes.append("Momentum haussier confirmé.")

    elif trend == "baissière" and momentum_score <= -2 and not psychology_flags["anti_fomo"]:
        signal = "VENTE"
        entry_price = round(last_close * 1.002, 2)
        notes.append("Momentum baissier confirmé.")

    else:
        notes.append("Momentum insuffisant pour une décision forte.")

    notes.extend(mtf_notes)
    notes.extend(regime_notes)

    notes.append(
        f"Ridge R²={ridge_res['r2']:.3f}, slope={ridge_res['slope']:.6f}, valid={ridge_res['valid']}."
    )
    notes.append(
        f"GARCH vol={garch_res['current_vol']:.6f}, high_vol={garch_res['high_vol']}."
    )
    mr_stats = compute_mean_reversion_stats(spread_vwap)
    notes.append(f"Mean reversion z-score={mr_stats['z_score']:.3f}.")

    return {
        "signal": signal,
        "entry_price": entry_price,
        "momentum_score": momentum_score,
        "trend": trend,
        "support": support,
        "resistance": resistance,
        "last_price": round(last_close, 2),
        "regime": regime,
        "risk_profile": risk_profile,
        "psychology_flags": psychology_flags,
        "notes": notes,
        "atr_value": float(last_atr),
    }
