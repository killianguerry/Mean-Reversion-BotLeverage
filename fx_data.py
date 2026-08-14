"""
fx_data.py
----------
Récupère et met en cache l'historique du taux de change EUR/USD (nombre de
dollars pour 1 euro), utilisé par portfolio_sim.py pour convertir les
positions en actions US (cotées en USD) vers un capital de référence en EUR.

Source : Yahoo Finance, ticker "EURUSD=X" (même pipeline que les prix
actions, via yfinance). Mis en cache en Parquet comme le reste des données,
pour éviter de retélécharger à chaque lancement.

Si le téléchargement échoue (pas de réseau, Yahoo indisponible), l'appelant
doit retomber sur un taux constant — voir PortfolioSimParams.eur_usd_fallback_rate
dans portfolio_sim.py. C'est moins réaliste (le taux de change bouge de
±10-15% sur plusieurs années) mais permet de continuer à travailler hors-ligne.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("fx_data")

DEFAULT_CACHE_PATH = "data/eurusd_history.parquet"
DEFAULT_MAX_AGE_DAYS = 1.0
DEFAULT_FALLBACK_RATE = 1.08  # utilisé uniquement si le téléchargement échoue


def _cache_age_days(path: Path) -> float:
    if not path.exists():
        return float("inf")
    return (time.time() - path.stat().st_mtime) / 86400


def get_eurusd_series(
    period: str = "5y",
    cache_path: str = DEFAULT_CACHE_PATH,
    max_age_days: float = DEFAULT_MAX_AGE_DAYS,
    force_refresh: bool = False,
) -> pd.Series:
    """Retourne une Series indexée par date (normalisée, sans heure), valeur
    = nombre de USD pour 1 EUR (ex: 1.08 -> 1€ = 1.08$).

    Lève une exception en cas d'échec — c'est volontaire : cette fonction ne
    doit jamais retourner silencieusement un taux inventé. À l'appelant de
    décider du repli (taux constant) et de le signaler à l'utilisateur.
    """
    path = Path(cache_path)
    age = _cache_age_days(path)

    if not force_refresh and age <= max_age_days:
        log.info(f"Cache EUR/USD valide ({path}, âge {age:.2f} j) -> chargement direct.")
        return pd.read_parquet(path)["rate"]

    log.info("Téléchargement de l'historique EUR/USD (Yahoo Finance, EURUSD=X)...")
    hist = yf.Ticker("EURUSD=X").history(period=period, interval="1d", auto_adjust=False)
    if hist.empty:
        raise RuntimeError(
            "Impossible de récupérer l'historique EUR/USD depuis Yahoo Finance "
            "(réseau indisponible ou ticker EURUSD=X inaccessible)."
        )

    series = hist["Close"].copy()
    series.index = pd.to_datetime(series.index.date)  # normalise (retire l'heure/le fuseau)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    series.name = "rate"

    path.parent.mkdir(parents=True, exist_ok=True)
    series.to_frame().to_parquet(path)
    log.info(f"Cache EUR/USD mis à jour : {path} ({len(series)} points, {series.index.min().date()} -> {series.index.max().date()})")
    return series


def get_eurusd_series_safe(
    period: str = "5y",
    cache_path: str = DEFAULT_CACHE_PATH,
    max_age_days: float = DEFAULT_MAX_AGE_DAYS,
    force_refresh: bool = False,
    fallback_rate: float = DEFAULT_FALLBACK_RATE,
) -> tuple[pd.Series, bool]:
    """Comme get_eurusd_series, mais ne lève jamais d'exception : retombe sur
    une série constante à fallback_rate en cas d'échec. Retourne (série, used_live)
    où used_live indique si les vraies données historiques ont pu être utilisées."""
    try:
        series = get_eurusd_series(period, cache_path, max_age_days, force_refresh)
        return series, True
    except Exception as exc:
        log.warning(f"Repli sur taux EUR/USD constant ({fallback_rate}) : {exc}")
        # Série "plate" sur une période large ; portfolio_sim la reindexera
        # (ffill) sur le calendrier réel de la simulation.
        idx = pd.date_range("2000-01-01", "2035-12-31", freq="D")
        return pd.Series(fallback_rate, index=idx, name="rate"), False
