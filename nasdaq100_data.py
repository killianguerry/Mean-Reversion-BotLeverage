"""
nasdaq100_data.py
------------------
Module central de données du bot : récupère AUTOMATIQUEMENT
- la liste actuelle des 100 tickers du Nasdaq-100 (Wikipedia)
- leur historique de prix (Yahoo Finance via yfinance)
et les met en cache localement (Parquet) pour éviter de retélécharger à
chaque lancement.

Aucune action manuelle n'est requise : pas de CSV à fournir, pas de script
séparé à lancer avant d'utiliser le bot. C'est ce module que main.py et
streamlit_app.py appellent pour obtenir les données.

Utilisation directe en ligne de commande (facultatif, pour forcer un
rafraîchissement du cache sans lancer de backtest) :
    python nasdaq100_data.py --refresh
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

WIKI_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"
WIKI_REST_HTML_URL = "https://en.wikipedia.org/api/rest_v1/page/html/Nasdaq-100"
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
DEFAULT_CACHE_PATH = "data/nasdaq100_universe.parquet"
DEFAULT_MAX_AGE_DAYS = 1.0  # au-delà, le cache est considéré périmé et retéléchargé

# Filet de sécurité ultime : liste statique des tickers Nasdaq-100, à jour au
# 20 janvier 2026. N'est utilisée QUE si les 3 méthodes de récupération en
# ligne échouent toutes (Wikipedia inaccessible depuis l'environnement où
# tourne le bot). Le index étant rebalancé seulement quelques fois par an,
# cette liste reste raisonnablement à jour même après plusieurs mois.
STATIC_FALLBACK_TICKERS = [
    "ADBE", "AMD", "ABNB", "ALNY", "GOOGL", "GOOG", "AMZN", "AEP", "AMGN", "ADI",
    "AAPL", "AMAT", "APP", "ARM", "ASML", "ADSK", "ADP", "AXON", "BKR", "BKNG",
    "AVGO", "CDNS", "CHTR", "CTAS", "CSCO", "CCEP", "CTSH", "CMCSA", "CEG", "CPRT",
    "CSGP", "COST", "CRWD", "CSX", "DDOG", "DXCM", "FANG", "DASH", "EA", "EXC",
    "FAST", "FER", "FTNT", "GEHC", "GILD", "HON", "IDXX", "INSM", "INTC", "INTU",
    "ISRG", "KDP", "KLAC", "KHC", "LRCX", "LIN", "MAR", "MRVL", "MELI", "META",
    "MCHP", "MU", "MSFT", "MSTR", "MDLZ", "MPWR", "MNST", "NFLX", "NVDA", "NXPI",
    "ORLY", "ODFL", "PCAR", "PLTR", "PANW", "PAYX", "PYPL", "PDD", "PEP", "QCOM",
    "REGN", "ROP", "ROST", "SNDK", "STX", "SHOP", "SBUX", "SNPS", "TMUS", "TTWO",
    "TSLA", "TXN", "TRI", "VRSK", "VRTX", "WMT", "WBD", "WDC", "WDAY", "XEL", "ZS",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("nasdaq100_data")

ProgressCallback = Optional[Callable[[float, str], None]]


# ---------------------------------------------------------------------------
# 1. Liste des tickers (Wikipedia)
# ---------------------------------------------------------------------------

def clean_ticker(raw: str) -> str:
    """Nettoie un ticker pour Yahoo Finance (ex: 'BRK.B' -> 'BRK-B')."""
    t = str(raw).strip().upper()
    return t.replace(".", "-").replace(" ", "")


CANDIDATE_TICKER_COLUMNS = ["Ticker", "Symbol", "Ticker symbol"]


def _extract_tickers_pandas(html: str) -> Optional[List[str]]:
    """1er essai : pandas.read_html(), rapide et suffisant la plupart du
    temps. Peut échouer à trouver la bonne table sur certaines pages
    Wikipedia complexes (tables 'sortable', cellules avec liens imbriqués)."""
    try:
        tables = pd.read_html(io.StringIO(html))
    except Exception:
        return None

    for table in tables:
        cols = [str(c).strip() for c in table.columns]
        for cand in CANDIDATE_TICKER_COLUMNS:
            if cand in cols:
                found = table[cand].astype(str).tolist()
                if 90 <= len(found) <= 110:
                    return found
    return None


def _extract_tickers_bs4(html: str) -> Optional[List[str]]:
    """2e essai (filet de sécurité) : parsing manuel avec BeautifulSoup,
    plus robuste face aux tables Wikipedia complexes que pandas.read_html()
    parvient parfois pas à interpréter correctement."""
    soup = BeautifulSoup(html, "html.parser")
    for table in soup.find_all("table", class_=lambda c: c and "wikitable" in c):
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        matching = [h for h in headers if h in CANDIDATE_TICKER_COLUMNS]
        if not matching:
            continue
        ticker_idx = headers.index(matching[0])

        tickers = []
        for row in table.find_all("tr")[1:]:
            cells = row.find_all(["td", "th"])
            if len(cells) > ticker_idx:
                tickers.append(cells[ticker_idx].get_text(strip=True))

        if 90 <= len(tickers) <= 110:
            return tickers
    return None


def _extract_tickers_wikitext(wikitext: str) -> Optional[List[str]]:
    """3e méthode (filet de sécurité ultime avant la liste statique) :
    parse le wikitexte brut (source de l'article, récupéré via l'API
    MediaWiki) plutôt que le HTML rendu. Contourne tout problème de rendu
    ou de blocage spécifique à la page HTML publique."""
    table_blocks = re.findall(r"\{\|.*?\n\|\}", wikitext, re.DOTALL)
    for block in table_blocks:
        if not re.search(r"!\s*Ticker\b", block, re.IGNORECASE):
            continue

        tickers = []
        for row in block.split("|-"):
            row = row.strip()
            if not row or row.startswith("!") or row.startswith("{|"):
                continue
            cell_lines = [
                l.strip() for l in row.split("\n")
                if l.strip().startswith("|") and not l.strip().startswith("|}")
            ]
            if not cell_lines:
                continue
            first_cell = cell_lines[0].lstrip("|").strip()
            first_cell = first_cell.split("||")[0].strip()

            wikilink = re.match(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", first_cell)
            if wikilink:
                first_cell = wikilink.group(2) or wikilink.group(1)
            first_cell = re.sub(r"'{2,}", "", first_cell).strip()

            if first_cell and first_cell.replace("-", "").replace(".", "").isalnum() and first_cell.isupper():
                tickers.append(first_cell)

        if 90 <= len(tickers) <= 110:
            return tickers
    return None


def _fetch_wikitext(page_title: str = "Nasdaq-100") -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    }
    params = {
        "action": "query",
        "prop": "revisions",
        "titles": page_title,
        "rvprop": "content",
        "rvslots": "main",
        "formatversion": "2",
        "format": "json",
    }
    response = requests.get(WIKI_API_URL, params=params, headers=headers, timeout=15)
    response.raise_for_status()
    data = response.json()
    return data["query"]["pages"][0]["revisions"][0]["slots"]["main"]["content"]


def _fetch_html(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    return response.text


def get_nasdaq100_tickers(wiki_url: str = WIKI_URL) -> List[str]:
    """Récupère la liste actuelle des tickers du Nasdaq-100, en essayant
    plusieurs méthodes indépendantes dans l'ordre, la plus fiable d'abord :

    1. API REST Wikipedia (HTML propre, généralement moins sujette aux
       blocages anti-bot que la page publique classique)
    2. Page Wikipedia HTML classique (pandas puis BeautifulSoup)
    3. API MediaWiki (wikitexte brut, contourne tout problème de rendu HTML)
    4. Liste statique intégrée au code, en tout dernier recours, si les 3
       méthodes en ligne échouent (Wikipedia inaccessible depuis cet
       environnement) — pour que le bot reste utilisable quoi qu'il arrive.
    """
    log.info("Récupération de la liste des tickers du Nasdaq-100...")
    errors = []

    try:
        html = _fetch_html(WIKI_REST_HTML_URL)
        tickers = _extract_tickers_pandas(html) or _extract_tickers_bs4(html)
        if tickers:
            log.info(f"{len(tickers)} tickers récupérés via l'API REST Wikipedia.")
            return sorted({clean_ticker(t) for t in tickers if t and t.lower() != "nan"})
        errors.append("API REST : page récupérée mais aucune table de tickers trouvée")
    except Exception as exc:
        errors.append(f"API REST : {exc}")

    try:
        html = _fetch_html(wiki_url)
        tickers = _extract_tickers_pandas(html) or _extract_tickers_bs4(html)
        if tickers:
            log.info(f"{len(tickers)} tickers récupérés via la page Wikipedia classique.")
            return sorted({clean_ticker(t) for t in tickers if t and t.lower() != "nan"})
        errors.append("Page HTML classique : page récupérée mais aucune table de tickers trouvée")
    except Exception as exc:
        errors.append(f"Page HTML classique : {exc}")

    try:
        wikitext = _fetch_wikitext()
        tickers = _extract_tickers_wikitext(wikitext)
        if tickers:
            log.info(f"{len(tickers)} tickers récupérés via l'API wikitexte.")
            return sorted({clean_ticker(t) for t in tickers if t and t.lower() != "nan"})
        errors.append("API wikitexte : contenu récupéré mais aucune table de tickers trouvée")
    except Exception as exc:
        errors.append(f"API wikitexte : {exc}")

    log.warning(
        "Les 3 méthodes de récupération Wikipedia ont échoué. Détails : " + " | ".join(errors) +
        ". Utilisation de la liste statique de secours intégrée au code (peut être "
        "légèrement obsolète, mise à jour au 20 janvier 2026)."
    )
    return sorted(set(STATIC_FALLBACK_TICKERS))


# ---------------------------------------------------------------------------
# 2. Test de connexion (diagnostic rapide avant de lancer 100 téléchargements)
# ---------------------------------------------------------------------------

def self_test() -> None:
    """Petit téléchargement de contrôle. Lève une RuntimeError explicite en
    cas d'échec (cause la plus fréquente : yfinance trop ancien -> Yahoo a
    changé son API et les vieilles versions renvoient des données vides)."""
    log.info(f"yfinance version installée : {yf.__version__}")
    log.info("Test de connexion à Yahoo Finance (AAPL, 5 jours)...")
    try:
        test = yf.download("AAPL", period="5d", interval="1d", progress=False, auto_adjust=False)
    except Exception as exc:
        test, test_error = None, exc
    else:
        test_error = None

    if test is None or test.empty:
        raise RuntimeError(
            "Échec du test de connexion à Yahoo Finance. Pistes de résolution :\n"
            "  1) pip install --upgrade yfinance  (cause la plus fréquente : "
            "version trop ancienne, Yahoo change son API régulièrement)\n"
            "  2) Vérifie ta connexion internet.\n"
            "  3) Réseau d'entreprise/école ou VPN : Yahoo Finance peut être bloqué.\n"
            "  4) Rate limiting temporaire de Yahoo : attends quelques minutes.\n"
            f"Détail de l'erreur : {test_error}"
        )
    log.info("Test de connexion OK.")


# ---------------------------------------------------------------------------
# 3. Téléchargement OHLCV par lots, avec retry + fallback individuel
# ---------------------------------------------------------------------------

def _reshape_batch(data: pd.DataFrame, batch: List[str]) -> tuple[List[pd.DataFrame], List[str]]:
    frames, failed = [], []
    for ticker in batch:
        try:
            sub = data.copy() if len(batch) == 1 else data[ticker].copy()
            sub = sub.dropna(how="all")
            if sub.empty:
                raise ValueError("données vides")
            sub = sub.reset_index()
            sub["Ticker"] = ticker
            frames.append(sub)
        except Exception as exc:
            log.warning(f"  {ticker}: échec dans le lot bulk ({exc}), retry individuel.")
            failed.append(ticker)
    return frames, failed


def download_ohlcv(
    tickers: List[str],
    period: str = "5y",
    interval: str = "1d",
    batch_size: int = 10,
    pause: float = 1.5,
    progress_callback: ProgressCallback = None,
) -> pd.DataFrame:
    """Télécharge OHLCV pour tous les tickers, par lots, avec retry/backoff et
    fallback individuel. Retourne un DataFrame long : Date, Ticker, Open,
    High, Low, Close, Adj Close, Volume."""
    all_frames: List[pd.DataFrame] = []
    failed: List[str] = []

    batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]
    log.info(f"Téléchargement de {len(tickers)} tickers en {len(batches)} lot(s) de {batch_size}...")

    for b_idx, batch in enumerate(batches, start=1):
        log.info(f"Lot {b_idx}/{len(batches)} : {', '.join(batch)}")
        if progress_callback:
            progress_callback(0.1 + 0.8 * (b_idx - 1) / len(batches), f"Lot {b_idx}/{len(batches)} : {', '.join(batch)}")

        data = None
        for attempt in range(1, 4):
            try:
                data = yf.download(batch, period=period, interval=interval, group_by="ticker",
                                    auto_adjust=False, threads=True, progress=False)
                if data is not None and not data.empty:
                    break
                log.warning(f"  Lot {b_idx} : réponse vide (tentative {attempt}/3).")
            except Exception as exc:
                log.warning(f"  Lot {b_idx} : échec (tentative {attempt}/3) : {exc}")
                data = None
            time.sleep(pause * attempt)

        if data is not None and not data.empty:
            frames, batch_failed = _reshape_batch(data, batch)
            all_frames.extend(frames)
            failed.extend(batch_failed)
        else:
            log.warning(f"  Lot {b_idx} : échec définitif du bulk, retry individuel.")
            failed.extend(batch)

        time.sleep(pause)

    still_failed: List[str] = []
    for ticker in failed:
        success = False
        for attempt in range(1, 4):
            try:
                sub = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
                if sub.empty:
                    raise ValueError("données vides")
                sub = sub.reset_index()
                sub["Ticker"] = ticker
                all_frames.append(sub)
                success = True
                break
            except Exception as exc:
                log.warning(f"  {ticker} : échec tentative {attempt}/3 ({exc})")
                time.sleep(pause * attempt)
        if not success:
            log.error(f"  Échec définitif pour {ticker}.")
            still_failed.append(ticker)

    if still_failed:
        log.warning(f"{len(still_failed)} ticker(s) exclus du dataset : {still_failed}")

    if not all_frames:
        raise RuntimeError("Aucune donnée n'a pu être téléchargée pour aucun ticker.")

    result = pd.concat(all_frames, ignore_index=True)
    result.columns = [str(c).strip() for c in result.columns]
    if "Adj Close" not in result.columns and "Close" in result.columns:
        result["Adj Close"] = result["Close"]

    keep = [c for c in ["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in result.columns]
    result = result[keep].sort_values(["Ticker", "Date"]).reset_index(drop=True)

    if progress_callback:
        progress_callback(0.95, f"{result['Ticker'].nunique()} tickers téléchargés.")

    log.info(f"Dataset : {result['Ticker'].nunique()} tickers, {len(result)} lignes.")
    return result


# ---------------------------------------------------------------------------
# 4. Cache + point d'entrée principal utilisé par main.py / streamlit_app.py
# ---------------------------------------------------------------------------

def _cache_age_days(path: Path) -> float:
    if not path.exists():
        return float("inf")
    return (time.time() - path.stat().st_mtime) / 86400


def _long_to_dict(long_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if not pd.api.types.is_datetime64_any_dtype(long_df["Date"]):
        long_df = long_df.copy()
        long_df["Date"] = pd.to_datetime(long_df["Date"])
    per_ticker = {}
    for ticker, sub in long_df.groupby("Ticker"):
        df = sub.set_index("Date")[["Open", "High", "Low", "Close"]].sort_index().dropna()
        if len(df) > 100:  # assez d'historique pour calculer RSI hebdo / ADX / Bollinger
            per_ticker[ticker] = df
    return per_ticker


def get_universe(
    cache_path: str = DEFAULT_CACHE_PATH,
    max_age_days: float = DEFAULT_MAX_AGE_DAYS,
    force_refresh: bool = False,
    period: str = "5y",
    interval: str = "1d",
    progress_callback: ProgressCallback = None,
) -> Dict[str, pd.DataFrame]:
    """
    Point d'entrée unique pour obtenir les données du Nasdaq-100.

    - Si un cache local existe et a moins de `max_age_days` jours : le
      charge directement, AUCUN appel réseau.
    - Sinon (ou si force_refresh=True) : télécharge automatiquement depuis
      Wikipedia + Yahoo Finance, sauvegarde le cache, puis retourne les données.

    Retourne un dict {ticker: DataFrame OHLC indexé par Date}.
    """
    path = Path(cache_path)
    age = _cache_age_days(path)

    if not force_refresh and age <= max_age_days:
        log.info(f"Cache valide ({path}, âge {age:.2f} j) -> chargement direct, pas de téléchargement.")
        if progress_callback:
            progress_callback(1.0, "Données chargées depuis le cache local.")
        long_df = pd.read_parquet(path)
        return _long_to_dict(long_df)

    log.info(f"Cache absent/périmé (âge {age:.2f} j > {max_age_days} j) -> téléchargement automatique.")
    if progress_callback:
        progress_callback(0.02, "Test de connexion à Yahoo Finance...")
    self_test()

    if progress_callback:
        progress_callback(0.05, "Récupération de la liste des tickers (Wikipedia)...")
    tickers = get_nasdaq100_tickers()

    long_df = download_ohlcv(tickers, period=period, interval=interval, progress_callback=progress_callback)

    path.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_parquet(path, index=False)
    log.info(f"Cache mis à jour : {path}")
    if progress_callback:
        progress_callback(1.0, "Téléchargement terminé.")

    return _long_to_dict(long_df)


# ---------------------------------------------------------------------------
# CLI (facultatif) : forcer un rafraîchissement du cache indépendamment
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Gestion des données Nasdaq-100 (cache + téléchargement automatique)")
    p.add_argument("--refresh", action="store_true", help="Force le retéléchargement même si le cache est frais")
    p.add_argument("--max-age-days", type=float, default=DEFAULT_MAX_AGE_DAYS)
    p.add_argument("--period", type=str, default="5y")
    p.add_argument("--out", type=str, default=DEFAULT_CACHE_PATH)
    return p.parse_args()


def main():
    args = parse_args()
    try:
        universe = get_universe(
            cache_path=args.out,
            max_age_days=args.max_age_days,
            force_refresh=args.refresh,
            period=args.period,
        )
    except RuntimeError as exc:
        log.error(str(exc))
        sys.exit(1)
    log.info(f"{len(universe)} tickers disponibles avec assez d'historique pour le backtest.")


if __name__ == "__main__":
    sys.exit(main())
