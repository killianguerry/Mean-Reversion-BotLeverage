"""
portfolio.py
------------
Lance la stratégie mean reversion (backtester.py) sur un ensemble de tickers
(l'univers Nasdaq-100) et agrège les résultats : résumé par ticker + courbe
d'équity d'un portefeuille équipondéré. Utilisé à la fois par main.py (CLI)
et streamlit_app.py.
"""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from backtester import StrategyParams, run_backtest


def run_universe(per_ticker: Dict[str, pd.DataFrame], params: StrategyParams) -> Tuple[pd.DataFrame, pd.Series, pd.Series, Dict]:
    """
    Lance le backtest sur chaque ticker et agrège les résultats.

    Retourne :
    - summary_df : une ligne par ticker (perf, trades, win rate, max drawdown)
    - portfolio_curve : courbe d'équity moyenne d'un portefeuille équipondéré (base 100)
    - portfolio_buy_hold : courbe moyenne buy & hold équipondérée (base 100)
    - details : dict {ticker: {"result": BacktestResult, "df": DataFrame, "buy_hold": Series}}
      pour permettre l'affichage détaillé par action (ex: graphiques Streamlit)
    """
    summaries = []
    equity_curves = {}
    bh_curves = {}
    details = {}

    for ticker, df in per_ticker.items():
        try:
            result = run_backtest(df, params)
        except Exception:
            continue

        buy_hold = 10_000 * df["Close"] / df["Close"].iloc[0]
        long_trades = sum(1 for t in result.trades if t.direction == "long")
        short_trades = sum(1 for t in result.trades if t.direction == "short")

        summaries.append({
            "Ticker": ticker,
            "Performance (%)": result.total_return_pct,
            "Win rate (%)": result.win_rate_pct,
            "Nb trades": result.num_trades,
            "Trades longs": long_trades,
            "Trades shorts": short_trades,
            "Max drawdown (%)": result.max_drawdown_pct,
        })

        equity_curves[ticker] = result.equity_curve / result.equity_curve.iloc[0] * 100
        bh_curves[ticker] = buy_hold / buy_hold.iloc[0] * 100
        details[ticker] = {"result": result, "df": df, "buy_hold": buy_hold}

    if not summaries:
        raise RuntimeError("Aucun backtest n'a pu être exécuté sur l'univers fourni.")

    summary_df = pd.DataFrame(summaries).sort_values("Performance (%)", ascending=False).reset_index(drop=True)
    portfolio_curve = pd.DataFrame(equity_curves).mean(axis=1, skipna=True)
    portfolio_buy_hold = pd.DataFrame(bh_curves).mean(axis=1, skipna=True)

    return summary_df, portfolio_curve, portfolio_buy_hold, details
