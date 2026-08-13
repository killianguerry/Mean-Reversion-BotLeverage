"""
data_loader.py
--------------
Charge des données de prix (OHLC) depuis un CSV exporté de Yahoo Finance
(gère les formats de date anglais ET français, ex: "Jul 7, 2026" ou "nov. 7, 2023"),
ou directement depuis Yahoo Finance via yfinance si le package est installé.
"""

import re
import pandas as pd

_MONTHS = {
    "jan": 1, "janv": 1, "feb": 2, "fevr": 2, "mar": 3, "mars": 3,
    "apr": 4, "avr": 4, "may": 5, "mai": 5, "jun": 6, "juin": 6,
    "jul": 7, "juil": 7, "aug": 8, "aout": 8, "sep": 9, "sept": 9,
    "oct": 10, "nov": 11, "dec": 12, "dece": 12,
}


def _strip_accents(s: str) -> str:
    return (
        s.replace("é", "e").replace("è", "e").replace("û", "u")
        .replace("ô", "o").replace("î", "i").replace("ê", "e")
    )


def _parse_date(raw: str) -> pd.Timestamp:
    """Parse une date au format 'Jul 7, 2026' ou 'nov. 7, 2023' (FR ou EN)."""
    s = _strip_accents(raw.strip().replace(".", "").lower())
    parts = s.replace(",", "").split()
    month = None
    for key, val in _MONTHS.items():
        if parts[0].startswith(key):
            month = val
            break
    if month is None:
        raise ValueError(f"Mois non reconnu dans la date: {raw}")
    day, year = int(parts[1]), int(parts[2])
    return pd.Timestamp(year=year, month=month, day=day)


def load_csv(path: str) -> pd.DataFrame:
    """
    Charge un CSV au format Yahoo Finance:
    Date, Open, High, Low, Close, Adj Close, Volume
    Les nombres peuvent contenir des séparateurs de milliers ("29,362.20").
    Retourne un DataFrame trié par date croissante, indexé par Date,
    avec colonnes Open, High, Low, Close (float).
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    dates = pd.DatetimeIndex(df["Date"].apply(_parse_date).values)

    def to_float(col):
        return (
            df[col].astype(str).str.replace(",", "", regex=False).astype(float).values
        )

    out = pd.DataFrame({
        "Open": to_float("Open"),
        "High": to_float("High"),
        "Low": to_float("Low"),
        "Close": to_float("Close"),
    }, index=dates)

    out.index.name = "Date"
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def load_yfinance(ticker: str, period: str = "3y", interval: str = "1d") -> pd.DataFrame:
    """
    Alternative: télécharge les données directement via yfinance.
    Nécessite `pip install yfinance` et une connexion internet.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance n'est pas installé. Fais `pip install yfinance` "
            "ou utilise load_csv() avec un fichier téléchargé manuellement."
        ) from exc

    data = yf.download(ticker, period=period, interval=interval, progress=False)
    if data.empty:
        raise ValueError(f"Aucune donnée retournée pour {ticker}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    out = data[["Open", "High", "Low", "Close"]].copy()
    out.index.name = "Date"
    return out
