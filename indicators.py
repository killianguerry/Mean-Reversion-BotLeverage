"""
indicators.py
-------------
Indicateurs techniques utilisés par la stratégie de mean reversion :
Bollinger Bands, RSI (Wilder), ATR (Wilder), ADX (Wilder), resample hebdomadaire.
"""

import numpy as np
import pandas as pd


def bollinger_bands(close: pd.Series, period: int = 20, n_std: float = 2.0):
    """Retourne (bande_milieu, bande_haute, bande_basse)."""
    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = mid + n_std * std
    lower = mid - n_std * std
    return mid, upper, lower


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI selon la méthode de lissage de Wilder."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - 100 / (1 + rs)
    result[avg_loss == 0] = 100
    return result


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (lissage de Wilder). df doit avoir High, Low, Close."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index (lissage de Wilder). df doit avoir High, Low, Close."""
    high, low, close = df["High"], df["Low"], df["Close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_ = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean() / atr_
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean() / atr_

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def consecutive_below(series: pd.Series, threshold: pd.Series) -> pd.Series:
    """Nombre de bougies consécutives (incluse) où series < threshold.
    Repart à 0 dès que la condition n'est plus vraie."""
    below = (series < threshold).astype(int)
    grp = (below == 0).cumsum()
    return below.groupby(grp).cumsum()


def consecutive_above(series: pd.Series, threshold: pd.Series) -> pd.Series:
    """Nombre de bougies consécutives (incluse) où series > threshold.
    Repart à 0 dès que la condition n'est plus vraie."""
    above = (series > threshold).astype(int)
    grp = (above == 0).cumsum()
    return above.groupby(grp).cumsum()


def resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Agrège les données daily en barres hebdomadaires (semaine finissant le vendredi)."""
    weekly = df.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
    }).dropna()
    return weekly


def map_weekly_to_daily(daily_index: pd.DatetimeIndex, weekly_series: pd.Series) -> pd.Series:
    """
    Aligne une série hebdomadaire (RSI, ADX...) sur l'index daily, en n'utilisant
    QUE la dernière semaine déjà CLÔTURÉE (pas de lookahead bias).
    """
    # Décale d'une semaine : la valeur d'une semaine ne devient "connue"
    # qu'au premier jour de la semaine suivante.
    shifted = weekly_series.shift(1)
    # reindex avec forward-fill sur le calendrier daily
    aligned = shifted.reindex(
        pd.date_range(shifted.index.min(), daily_index.max(), freq="D")
    ).ffill()
    return aligned.reindex(daily_index)
