"""
optimizer.py
------------
Optimiseur de réglages pour la stratégie mean reversion. Recherche aléatoire
sur l'espace des paramètres (bien plus efficace qu'une grille exhaustive vu
le nombre de réglages combinables), avec validation systématique hors
échantillon : l'optimisation se fait sur une période d'entraînement, et les
meilleurs candidats sont re-testés sur une période de test qu'ils n'ont
jamais vue, pour éviter le sur-ajustement (le même protocole qu'on a suivi
manuellement tout au long du projet).

Utilisé par main.py (--optimize) et streamlit_app.py (section Optimiseur).
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtester import StrategyParams
from portfolio import run_universe

ProgressCallback = Optional[Callable[[float, str], None]]

# ---------------------------------------------------------------------------
# Espace de recherche par défaut : les valeurs testées manuellement tout au
# long du projet, qui ont chacune démontré un effet réel sur la performance.
# ---------------------------------------------------------------------------
DEFAULT_SEARCH_SPACE: Dict[str, list] = {
    "bb_period": [10, 15, 20, 25, 30],
    "bb_std": [1.0, 1.5, 2.0, 2.5, 3.0],
    "rsi_long_threshold": [45, 50, 55, 60],
    "adx_daily_threshold": [10, 15, 20, 25],
    "adx_weekly_threshold": [10, 15, 20, 25],
    "atr_mult": [1.5, 2.0, 2.5, 3.0, 3.5],
    "confirm_bars": [1, 2, 3],
    "rsi_daily_threshold": [None, 30, 35, 40, 45],
}

# Colonnes de résultat (pas des réglages) à exclure quand on reconstruit un
# StrategyParams à partir d'une ligne de résultat
_METRIC_COLUMNS = {
    "n_trades", "trades_per_year", "win_rate", "strat_perf", "bh_perf",
    "max_dd", "perf_risk_ratio", "train_strat_perf", "train_win_rate",
    "train_perf_risk_ratio",
}

METRIC_LABELS = {
    "strat_perf": "Performance totale (%)",
    "perf_risk_ratio": "Ratio performance / drawdown",
    "win_rate": "Win rate (%)",
}


def split_train_test(per_ticker: Dict[str, pd.DataFrame], train_frac: float = 0.7) -> Tuple[dict, dict, pd.Timestamp]:
    """Découpe chaque ticker en une période d'entraînement et une période de
    test, sur la même date de coupure pour tous (pas de fuite d'info du
    futur). La période de test conserve un peu d'historique avant la
    coupure pour que les indicateurs (RSI hebdo, ADX...) soient déjà
    calculables dès le premier jour de test."""
    all_dates = sorted(set().union(*[set(df.index) for df in per_ticker.values()]))
    split_idx = int(len(all_dates) * train_frac)
    split_date = pd.Timestamp(all_dates[split_idx])
    lookback_start = split_date - pd.Timedelta(days=100)

    train_data = {t: df[df.index <= split_date] for t, df in per_ticker.items()}
    test_data = {t: df[df.index > lookback_start] for t, df in per_ticker.items()}
    return train_data, test_data, split_date


def evaluate(data: Dict[str, pd.DataFrame], params: StrategyParams) -> Optional[dict]:
    """Lance le backtest sur l'univers fourni et calcule les métriques
    agrégées du portefeuille équipondéré. Retourne None si aucun trade n'a
    pu être généré (évite de fausser le classement avec des configs mortes)."""
    try:
        summary_df, portfolio_curve, portfolio_bh, details = run_universe(data, params)
    except Exception:
        return None

    all_trades = [t for d in details.values() for t in d["result"].trades if t.pnl_pct is not None]
    if not all_trades:
        return None

    wins = sum(1 for t in all_trades if t.pnl_pct > 0)
    win_rate = wins / len(all_trades) * 100
    strat_perf = (portfolio_curve.iloc[-1] / portfolio_curve.iloc[0] - 1) * 100
    bh_perf = (portfolio_bh.iloc[-1] / portfolio_bh.iloc[0] - 1) * 100
    running_max = portfolio_curve.cummax()
    max_dd = ((portfolio_curve - running_max) / running_max).min() * 100
    years = (max(d.index.max() for d in data.values()) - min(d.index.min() for d in data.values())).days / 365.25

    return dict(
        n_trades=len(all_trades),
        trades_per_year=len(all_trades) / max(years, 0.1),
        win_rate=win_rate,
        strat_perf=strat_perf,
        bh_perf=bh_perf,
        max_dd=max_dd,
        perf_risk_ratio=(strat_perf / abs(max_dd)) if max_dd else 0.0,
    )


def _sample_params(rng: random.Random, search_space: Dict[str, list]) -> dict:
    """Tire un jeu de réglages au hasard dans l'espace de recherche."""
    return dict(
        bb_period=rng.choice(search_space["bb_period"]),
        bb_std=rng.choice(search_space["bb_std"]),
        rsi_long_threshold=rng.choice(search_space["rsi_long_threshold"]),
        adx_daily_threshold=rng.choice(search_space["adx_daily_threshold"]),
        adx_weekly_threshold=rng.choice(search_space["adx_weekly_threshold"]),
        atr_mult=rng.choice(search_space["atr_mult"]),
        confirm_bars=rng.choice(search_space["confirm_bars"]),
        rsi_daily_threshold=rng.choice(search_space["rsi_daily_threshold"]),
    )


def _clean_cfg_from_row(row: pd.Series) -> dict:
    """Extrait un dict de réglages StrategyParams propre à partir d'une
    ligne de résultat (DataFrame pandas) :
    - convertit les types numpy (np.float64, np.int64...) en types Python
      natifs, indispensable pour la sérialisation JSON (sauvegarde de profils)
    - retire les champs NaN (paramètres d'un autre mode de sortie que celui
      tiré pour ce trial) plutôt que de les transmettre tels quels, pour que
      StrategyParams utilise sa valeur par défaut sur ces champs-là.
    """
    int_fields = {"bb_period", "confirm_bars"}
    cfg = {}
    for k in row.index:
        if k in _METRIC_COLUMNS or str(k).startswith("train_"):
            continue
        v = row[k]
        if isinstance(v, float) and pd.isna(v):
            continue
        if isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = float(v)
        elif isinstance(v, np.bool_):
            v = bool(v)
        cfg[k] = int(v) if k in int_fields else v
    return cfg


def optimize(
    per_ticker: Dict[str, pd.DataFrame],
    n_trials: int = 60,
    top_k: int = 10,
    metric: str = "strat_perf",
    train_frac: float = 0.7,
    seed: int = 42,
    search_space: Optional[Dict[str, list]] = None,
    progress_callback: ProgressCallback = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """
    Optimise les réglages de la stratégie par recherche aléatoire.

    1. Découpe les données en train (train_frac) / test.
    2. Tire n_trials configurations au hasard, les évalue sur le TRAIN.
    3. Garde les top_k selon `metric`, les revalide sur le TEST (hors
       échantillon) — c'est CE classement-là qui fait foi, pas le train.

    Retourne (finalistes classés par perf de test, tous les essais du train,
    date de coupure train/test).
    """
    search_space = search_space or DEFAULT_SEARCH_SPACE
    rng = random.Random(seed)
    train_data, test_data, split_date = split_train_test(per_ticker, train_frac)

    seen = set()
    train_results = []
    trial, attempts, max_attempts = 0, 0, n_trials * 4

    while trial < n_trials and attempts < max_attempts:
        attempts += 1
        cfg = _sample_params(rng, search_space)
        key = tuple(sorted(cfg.items()))
        if key in seen:
            continue
        seen.add(key)
        trial += 1

        metrics = evaluate(train_data, StrategyParams(**cfg))
        if metrics:
            metrics.update(cfg)
            train_results.append(metrics)

        if progress_callback:
            progress_callback(0.75 * trial / n_trials, f"Essai {trial}/{n_trials} (entraînement)")

    train_df = pd.DataFrame(train_results)
    if train_df.empty:
        raise RuntimeError(
            "Aucun des essais n'a produit de trade exploitable sur la période "
            "d'entraînement. Essaie d'augmenter le nombre d'essais."
        )

    train_df = train_df.sort_values(metric, ascending=False).reset_index(drop=True)
    finalists = train_df.head(top_k)

    test_rows = []
    for i, (_, row) in enumerate(finalists.iterrows()):
        cfg = _clean_cfg_from_row(row)
        metrics = evaluate(test_data, StrategyParams(**cfg))
        if metrics:
            metrics.update(cfg)
            metrics[f"train_{metric}"] = float(row[metric])
            test_rows.append(metrics)
        if progress_callback:
            progress_callback(0.75 + 0.25 * (i + 1) / len(finalists), f"Validation hors échantillon {i+1}/{len(finalists)}")

    test_df = pd.DataFrame(test_rows).sort_values(metric, ascending=False).reset_index(drop=True)
    return test_df, train_df, split_date


def params_from_row(row: pd.Series) -> StrategyParams:
    """Reconstruit un StrategyParams complet et propre (types Python natifs)
    à partir d'une ligne de résultat de l'optimiseur."""
    return StrategyParams(**_clean_cfg_from_row(row))
