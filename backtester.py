"""
backtester.py
-------------
Stratégie de mean reversion "haut winrate + levier", adaptée d'une vidéo
YouTube (Bollinger Bands + RSI multi-timeframe + ADX + stop ATR),
transposée d'un usage crypto 1h/4h vers un usage actions/ETFs daily/weekly.

Règles d'entrée LONG (toutes doivent être vraies) :
  - RSI hebdomadaire (semaine précédente clôturée) > rsi_long_threshold
  - (optionnel) RSI daily < rsi_daily_threshold, confluence multi-timeframe
  - ADX daily > adx_daily_threshold ET ADX weekly (semaine précédente) > adx_weekly_threshold
  - Clôture daily < bande de Bollinger basse, confirmée sur confirm_bars
    clôtures consécutives (pas un simple touch isolé)
  -> Entrée à l'ouverture du jour suivant.

Règles de sortie LONG (mode unique : mean reversion pure, pensé pour
maximiser le winrate — petits gains fréquents plutôt que laisser courir) :
  - Stop-loss : plus bas de la bougie signal - atr_mult * ATR(14), statique
  - Take profit : clôture > bande de Bollinger haute

Le short est symétrique et optionnel (désactivé par défaut : le short sur
actions a des contraintes réglementaires/coûts que le long n'a pas).

Le P&L de chaque trade peut être amplifié par un levier (params.leverage),
avec coût d'emprunt et liquidation simulée — voir StrategyParams.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from indicators import (
    bollinger_bands, rsi, atr, adx, resample_weekly, map_weekly_to_daily,
    consecutive_below, consecutive_above,
)


@dataclass
class StrategyParams:
    bb_period: int = 20
    bb_std: float = 1.5
    rsi_period: int = 14
    rsi_long_threshold: float = 50.0
    rsi_short_threshold: float = 45.0
    adx_period: int = 14
    adx_daily_threshold: float = 15.0
    adx_weekly_threshold: float = 15.0
    atr_period: int = 14
    atr_mult: float = 2.5
    enable_shorts: bool = False
    initial_equity: float = 10_000.0
    # --- Filtres "haute conviction" (winrate) ---
    # Nombre de clôtures CONSÉCUTIVES requises hors de la bande de Bollinger
    # avant d'autoriser une entrée (1 = comportement d'origine, un simple
    # dépassement suffit). Un signal qui persiste sur plusieurs jours est
    # une confluence plus forte qu'un touch isolé -> moins de trades, mais
    # signal plus propre.
    confirm_bars: int = 1
    # Filtre RSI additionnel sur le RSI DAILY (en plus du RSI hebdo déjà en
    # place). None = désactivé (comportement d'origine). Ex: 40 -> exige en
    # plus que le RSI daily soit lui aussi en zone de faiblesse pour un long,
    # ce qui ajoute une confluence multi-timeframe stricte.
    rsi_daily_threshold: Optional[float] = None
    # --- Levier ---
    # Multiplicateur appliqué au P&L de chaque trade. 1.0 = pas de levier.
    # ATTENTION : le levier n'améliore jamais l'edge d'une stratégie, il
    # amplifie ses résultats (gains ET pertes) et introduit un risque de
    # liquidation qu'une stratégie non-levier n'a pas. Voir liquidation
    # ci-dessous.
    leverage: float = 1.0
    # Taux d'emprunt ANNUEL (en fraction, ex 0.06 = 6%/an) appliqué au
    # capital emprunté (leverage - 1) x jours de détention / 365. C'est un
    # coût réel du levier sur marge, souvent oublié dans les backtests
    # "jouets". 0 = pas de coût (irréaliste au-delà de leverage=1).
    leverage_borrow_rate_annual: float = 0.06
    # Si la perte sur un trade, une fois le levier appliqué, atteint ou
    # dépasse ce seuil (fraction du capital du trade, ex 0.9 = -90%), le
    # trade est considéré LIQUIDÉ : sortie forcée à -liquidation_threshold,
    # quel que soit le stop-loss théorique (simule un appel de marge /
    # liquidation qui intervient avant que le stop ne se déclenche, en cas
    # de gap ou de mouvement violent intra-journalier).
    liquidation_threshold: float = 0.9


@dataclass
class Trade:
    direction: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_pct: Optional[float] = None  # P&L du trade APRÈS levier et coût d'emprunt
    raw_pnl_pct: Optional[float] = None  # P&L brut du mouvement de prix, SANS levier


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: List[Trade] = field(default_factory=list)

    @property
    def total_return_pct(self) -> float:
        return (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1) * 100

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate_pct(self) -> float:
        closed = [t for t in self.trades if t.pnl_pct is not None]
        if not closed:
            return 0.0
        wins = sum(1 for t in closed if t.pnl_pct > 0)
        return wins / len(closed) * 100

    @property
    def max_drawdown_pct(self) -> float:
        running_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve - running_max) / running_max
        return drawdown.min() * 100

    def summary(self) -> str:
        return (
            f"Performance totale     : {self.total_return_pct:+.2f}%\n"
            f"Nombre de trades       : {self.num_trades}\n"
            f"Win rate               : {self.win_rate_pct:.1f}%\n"
            f"Max drawdown           : {self.max_drawdown_pct:.2f}%\n"
        )


def run_backtest(df: pd.DataFrame, params: StrategyParams) -> BacktestResult:
    """
    df : DataFrame daily avec colonnes Open, High, Low, Close, indexé par Date.
    params : StrategyParams
    """
    df = df.copy()

    mid, upper, lower = bollinger_bands(df["Close"], params.bb_period, params.bb_std)
    df["bb_mid"], df["bb_upper"], df["bb_lower"] = mid, upper, lower
    df["adx_daily"] = adx(df, params.adx_period)
    df["atr"] = atr(df, params.atr_period)

    weekly = resample_weekly(df)
    weekly_rsi = rsi(weekly["Close"], params.rsi_period)
    weekly_adx = adx(weekly, params.adx_period)

    df["rsi_weekly"] = map_weekly_to_daily(df.index, weekly_rsi)
    df["adx_weekly"] = map_weekly_to_daily(df.index, weekly_adx)
    df["rsi_daily"] = rsi(df["Close"], params.rsi_period)

    # Confirmation multi-bougies : nb de clôtures consécutives sous/sur la bande
    df["below_streak"] = consecutive_below(df["Close"], df["bb_lower"])
    df["above_streak"] = consecutive_above(df["Close"], df["bb_upper"])

    df = df.dropna(subset=["bb_mid", "adx_daily", "atr", "rsi_weekly", "adx_weekly", "rsi_daily"])

    dates = df.index
    opens, highs, lows, closes = (
        df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values,
    )
    bb_upper, bb_lower = df["bb_upper"].values, df["bb_lower"].values
    adx_d, adx_w = df["adx_daily"].values, df["adx_weekly"].values
    rsi_w, rsi_d = df["rsi_weekly"].values, df["rsi_daily"].values
    atr_v = df["atr"].values
    below_streak, above_streak = df["below_streak"].values, df["above_streak"].values

    n = len(df)
    equity = params.initial_equity
    equity_curve = np.full(n, equity)
    trades: List[Trade] = []
    position = None  # dict avec direction, entry_idx, entry_price, stop

    for i in range(n - 1):
        if position is None:
            long_signal = (
                rsi_w[i] > params.rsi_long_threshold
                and adx_d[i] > params.adx_daily_threshold
                and adx_w[i] > params.adx_weekly_threshold
                and closes[i] < bb_lower[i]
                and below_streak[i] >= params.confirm_bars
                and (params.rsi_daily_threshold is None or rsi_d[i] < params.rsi_daily_threshold)
            )
            short_signal = (
                params.enable_shorts
                and rsi_w[i] < params.rsi_short_threshold
                and adx_d[i] > params.adx_daily_threshold
                and adx_w[i] > params.adx_weekly_threshold
                and closes[i] > bb_upper[i]
                and above_streak[i] >= params.confirm_bars
                and (params.rsi_daily_threshold is None or rsi_d[i] > (100 - params.rsi_daily_threshold))
            )
            if long_signal:
                entry_price = opens[i + 1]
                stop = lows[i] - params.atr_mult * atr_v[i]
                position = {"dir": "long", "entry_idx": i + 1, "entry_price": entry_price, "stop": stop}
                trades.append(Trade("long", dates[i + 1], entry_price))
            elif short_signal:
                entry_price = opens[i + 1]
                stop = highs[i] + params.atr_mult * atr_v[i]
                position = {"dir": "short", "entry_idx": i + 1, "entry_price": entry_price, "stop": stop}
                trades.append(Trade("short", dates[i + 1], entry_price))
            equity_curve[i + 1] = equity
        else:
            j = i + 1
            exit_price, exit_reason = None, None

            # Sortie mean reversion pure : stop-loss ATR statique (fixé à
            # l'entrée) ou take-profit dès que le prix touche la bande de
            # Bollinger opposée. Petits gains fréquents -> vise le winrate
            # le plus élevé, quitte à couper les gagnants tôt.
            if position["dir"] == "long":
                if lows[j] <= position["stop"]:
                    exit_price = min(opens[j], position["stop"])
                    exit_reason = "stop_loss"
                elif closes[j] > bb_upper[j]:
                    exit_price = closes[j]
                    exit_reason = "take_profit"
            else:
                if highs[j] >= position["stop"]:
                    exit_price = max(opens[j], position["stop"])
                    exit_reason = "stop_loss"
                elif closes[j] < bb_lower[j]:
                    exit_price = closes[j]
                    exit_reason = "take_profit"

            if exit_price is not None:
                raw_ret = (
                    (exit_price - position["entry_price"]) / position["entry_price"]
                    if position["dir"] == "long"
                    else (position["entry_price"] - exit_price) / position["entry_price"]
                )
                days_held = max(j - position["entry_idx"], 0)

                # --- Levier ---
                lev_ret = raw_ret * params.leverage
                # Coût d'emprunt sur la portion empruntée (leverage - 1), au
                # prorata du temps de détention. Ne s'applique que si on
                # emprunte réellement (leverage > 1).
                if params.leverage > 1.0:
                    borrow_cost = (params.leverage - 1.0) * params.leverage_borrow_rate_annual * (days_held / 365.0)
                    lev_ret -= borrow_cost
                # --- Liquidation ---
                # Si la perte levier dépasse le seuil de liquidation, le
                # trade est coupé à ce seuil (perte plafonnée mais actée),
                # peu importe où le stop "théorique" se situait : au-delà
                # de ce niveau, en réalité le courtier liquide la position
                # avant que ta logique de stop n'ait la main.
                if lev_ret <= -params.liquidation_threshold:
                    lev_ret = -params.liquidation_threshold
                    exit_reason = "liquidation"

                equity *= (1 + lev_ret)
                trade = trades[-1]
                trade.exit_date, trade.exit_price = dates[j], exit_price
                trade.exit_reason = exit_reason
                trade.pnl_pct = lev_ret * 100
                trade.raw_pnl_pct = raw_ret * 100
                position = None
                equity_curve[j] = equity
            else:
                mark_raw_ret = (
                    (closes[j] - position["entry_price"]) / position["entry_price"]
                    if position["dir"] == "long"
                    else (position["entry_price"] - closes[j]) / position["entry_price"]
                )
                mark_lev_ret = mark_raw_ret * params.leverage
                mark_lev_ret = max(mark_lev_ret, -params.liquidation_threshold)
                equity_curve[j] = equity * (1 + mark_lev_ret)

    return BacktestResult(
        equity_curve=pd.Series(equity_curve, index=dates),
        trades=trades,
    )
