"""
portfolio_sim.py
-----------------
Simulation de portefeuille "réaliste" : une SEULE enveloppe de capital en
EUR, taille de chaque position = 2% de l'équity courante, 20 positions
maximum en simultané, frais modélisés sur la structure Interactive Brokers
(IBKR Pro, tarification fixe pour les actions US) + conversion de change
EUR/USD.

Contrairement à portfolio.py (qui fait tourner CHAQUE ticker isolément avec
son propre capital plein, puis moyenne les courbes d'équity — un raccourci
pédagogique pratique mais irréaliste pour juger une performance en argent
réel), ce module simule un SEUL portefeuille partagé qui alloue son capital
entre les tickers au fil du temps, dans l'ordre chronologique réel des
signaux, avec les contraintes de capacité et de trésorerie qu'un vrai
compte de courtage impose : si 20 positions sont déjà ouvertes ou s'il ne
reste pas assez de cash, un signal est ignoré, exactement comme il le
serait en réel.

Hypothèses de frais (Interactive Brokers, IBKR Pro, tarification FIXE pour
les actions US — vérifié sur la documentation IBKR) :
  - Commission par action : 0.005 $/action, minimum 1.00 $/ordre, plafonnée
    à 1% de la valeur de l'ordre.
  - Conversion de change (EUR -> USD à l'achat, USD -> EUR à la vente) :
    0.03% du montant converti, minimum 2.00 $ (frais de conversion
    AUTOMATIQUE IBKR ; la conversion manuelle via IdealPro est moins chère
    mais demande de gérer soi-même sa trésorerie en devises, non modélisée
    ici par souci de simplicité).

Simplifications assumées (à garder en tête en interprétant les résultats) :
  - Chaque trade convertit intégralement EUR<->USD à l'entrée ET à la
    sortie (pas de trésorerie USD conservée entre deux trades) : c'est le
    scénario le PLUS PÉNALISANT en frais de change, donc plutôt
    conservateur (un vrai trader actif minimiserait ses conversions en
    gardant du cash en USD).
  - Positions en nombre d'actions ENTIER (pas de fractionnaire, même si
    IBKR le permet), arrondi à l'inférieur -> une position peut être un
    peu plus petite que 2% pile.
  - Le levier (StrategyParams.leverage) n'est PAS appliqué ici : la taille
    de position à 2% de l'équity est elle-même le contrôle de risque de ce
    module. Combiner les deux (levier + sizing réaliste) est possible mais
    n'est pas fait par défaut, pour ne pas mélanger deux sources de risque
    différentes dans un seul chiffre.
  - Pas de modélisation des frais de financement/marge sur les positions
    short (qui ne sont de toute façon pas activées par défaut), ni des
    dividendes, ni du slippage (exécution supposée au prix exact retenu
    par le backtester mono-actif).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import pandas as pd

from backtester import StrategyParams, run_backtest

ProgressCallback = Optional[Callable[[float, str], None]]


@dataclass
class PortfolioSimParams:
    initial_capital_eur: float = 100_000.0
    max_positions: int = 20
    position_pct: float = 0.02  # 2% de l'équity courante par position
    # --- Frais IBKR (actions US, tarification FIXE) ---
    commission_per_share_usd: float = 0.005
    commission_min_usd: float = 1.0
    commission_max_pct_of_trade: float = 0.01  # plafond 1% de la valeur de l'ordre
    # --- Conversion de change EUR/USD (auto-conversion IBKR) ---
    fx_fee_pct: float = 0.0003  # 0.03%
    fx_fee_min_usd: float = 2.0
    # Taux utilisé UNIQUEMENT si aucun historique EUR/USD n'a pu être chargé
    eur_usd_fallback_rate: float = 1.08


@dataclass
class ClosedTrade:
    ticker: str
    direction: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    shares: int
    entry_price_usd: float
    exit_price_usd: float
    cost_basis_eur: float   # capital notionnel engagé à l'entrée (hors frais)
    fees_eur: float          # commissions + frais de change, entrée + sortie cumulés
    pnl_eur: float
    pnl_pct: float            # sur le cost_basis_eur
    exit_reason: Optional[str]


@dataclass
class PortfolioSimResult:
    equity_curve: pd.Series  # en €, au fil du temps
    closed_trades: List[ClosedTrade] = field(default_factory=list)
    n_skipped_capacity: int = 0  # signaux ignorés car max_positions déjà atteint
    n_skipped_cash: int = 0       # signaux ignorés par manque de trésorerie
    n_skipped_lot: int = 0        # signaux ignorés car 2% de l'équity < prix d'1 action
    total_fees_eur: float = 0.0
    used_live_fx: bool = True     # False si repli sur taux EUR/USD constant

    @property
    def final_equity_eur(self) -> float:
        return float(self.equity_curve.iloc[-1])

    @property
    def total_return_pct(self) -> float:
        return (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1) * 100

    @property
    def max_drawdown_pct(self) -> float:
        running_max = self.equity_curve.cummax()
        dd = (self.equity_curve - running_max) / running_max
        return dd.min() * 100

    @property
    def win_rate_pct(self) -> float:
        if not self.closed_trades:
            return 0.0
        wins = sum(1 for t in self.closed_trades if t.pnl_eur > 0)
        return wins / len(self.closed_trades) * 100

    def summary(self) -> str:
        n_signals = (len(self.closed_trades) + self.n_skipped_capacity
                     + self.n_skipped_cash + self.n_skipped_lot)
        fx_note = "" if self.used_live_fx else "  (⚠️ taux EUR/USD constant, historique indisponible)"
        return (
            f"Capital initial            : {self.equity_curve.iloc[0]:,.0f} €\n"
            f"Capital final               : {self.final_equity_eur:,.0f} €{fx_note}\n"
            f"Performance totale          : {self.total_return_pct:+.2f}%\n"
            f"Max drawdown                : {self.max_drawdown_pct:.2f}%\n"
            f"Trades exécutés             : {len(self.closed_trades)}\n"
            f"Win rate                    : {self.win_rate_pct:.1f}%\n"
            f"Frais totaux (commissions + change) : {self.total_fees_eur:,.0f} €\n"
            f"Signaux ignorés (20 positions déjà ouvertes) : {self.n_skipped_capacity}\n"
            f"Signaux ignorés (trésorerie insuffisante)    : {self.n_skipped_cash}\n"
            f"Signaux ignorés (position < 1 action)        : {self.n_skipped_lot}\n"
            f"({n_signals} signaux générés au total par la stratégie)"
        )


def _commission_usd(shares: int, price_usd: float, p: PortfolioSimParams) -> float:
    """Commission IBKR Pro, tarification fixe : 0.005$/action, minimum
    1$/ordre, plafonnée à 1% de la valeur de l'ordre (le plafond l'emporte
    sur le minimum si les deux s'appliquent, conformément à la règle IBKR)."""
    trade_value = shares * price_usd
    fee = max(shares * p.commission_per_share_usd, p.commission_min_usd)
    fee = min(fee, p.commission_max_pct_of_trade * trade_value)
    return fee


def _fx_fee_usd(trade_value_usd: float, p: PortfolioSimParams) -> float:
    """Frais de conversion de change (approximé sur la valeur du trade en
    USD ; en réalité appliqué sur le montant EUR converti, du même ordre de
    grandeur)."""
    return max(p.fx_fee_pct * trade_value_usd, p.fx_fee_min_usd)


def run_portfolio_simulation(
    per_ticker: Dict[str, pd.DataFrame],
    strategy_params: StrategyParams,
    sim_params: PortfolioSimParams,
    eur_usd_rate: Optional[pd.Series] = None,
    used_live_fx: bool = True,
    progress_callback: ProgressCallback = None,
) -> PortfolioSimResult:
    """
    Simule un portefeuille unique en EUR sur l'ensemble de per_ticker.

    eur_usd_rate : Series {date -> USD par EUR}. Si None, utilise le taux
    constant sim_params.eur_usd_fallback_rate.
    """
    # 1. Génère les trades candidats (entrée/sortie) ticker par ticker, en
    #    réutilisant EXACTEMENT la même logique de signal que le backtester
    #    mono-actif (run_backtest). On ne récupère que les dates/prix
    #    d'entrée et de sortie : son équity interne suppose un capital plein
    #    et indépendant par ticker, ce qui n'a pas de sens ici.
    tickers = list(per_ticker.keys())
    candidates = []
    for idx, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(0.05 + 0.45 * idx / max(len(tickers), 1), f"Signaux {ticker}...")
        df = per_ticker[ticker]
        try:
            result = run_backtest(df, strategy_params)
        except Exception:
            continue
        for t in result.trades:
            candidates.append({
                "ticker": ticker, "direction": t.direction,
                "entry_date": t.entry_date, "entry_price": t.entry_price,
                "exit_date": t.exit_date, "exit_price": t.exit_price,
                "exit_reason": t.exit_reason,
            })

    if not candidates:
        raise RuntimeError("Aucun signal généré sur l'univers fourni avec ces réglages.")

    # 2. Calendrier unifié = union de toutes les dates présentes dans
    #    l'univers de données.
    all_dates = pd.DatetimeIndex(sorted(set().union(*(df.index for df in per_ticker.values()))))

    # 3. Taux de change EUR/USD aligné sur le calendrier de la simulation.
    if eur_usd_rate is not None:
        fx = eur_usd_rate.reindex(all_dates).ffill().bfill()
    else:
        fx = pd.Series(sim_params.eur_usd_fallback_rate, index=all_dates)
        used_live_fx = False

    close_dict = {t: per_ticker[t]["Close"] for t in tickers}

    entries_by_date: Dict[pd.Timestamp, list] = {}
    exits_by_date: Dict[pd.Timestamp, list] = {}
    for ct in candidates:
        ct["_id"] = (ct["ticker"], ct["entry_date"])
        entries_by_date.setdefault(ct["entry_date"], []).append(ct)
        if ct["exit_date"] is not None:
            exits_by_date.setdefault(ct["exit_date"], []).append(ct)

    cash_eur = sim_params.initial_capital_eur
    open_positions: Dict[tuple, dict] = {}
    closed_trades: List[ClosedTrade] = []
    equity_points: Dict[pd.Timestamp, float] = {}
    total_fees_eur = 0.0
    n_skip_capacity = n_skip_cash = n_skip_lot = 0
    last_price_cache: Dict[str, float] = {}

    def mark_to_market(date: pd.Timestamp) -> float:
        """Valeur nette en USD des positions ouvertes : + valeur de marché
        pour un long, - coût de rachat pour un short (le cash de la vente à
        découvert a déjà été crédité à l'entrée, donc le passif restant est
        ce qu'il en coûterait de racheter maintenant)."""
        total = 0.0
        for pos in open_positions.values():
            price = close_dict[pos["ticker"]].get(date)
            if price is None or pd.isna(price):
                price = last_price_cache.get(pos["ticker"], pos["entry_price_usd"])
            else:
                last_price_cache[pos["ticker"]] = price
            total += pos["shares"] * price if pos["direction"] == "long" else -pos["shares"] * price
        return total

    n_days = len(all_dates)
    for day_idx, date in enumerate(all_dates):
        if progress_callback and day_idx % 60 == 0:
            progress_callback(0.5 + 0.5 * day_idx / max(n_days, 1), f"Simulation portefeuille... {date.date()}")

        rate = float(fx.loc[date])  # USD pour 1 EUR

        # --- Sorties d'abord : libère capacité et trésorerie avant les entrées du jour ---
        for ct in exits_by_date.get(date, []):
            pos = open_positions.pop(ct["_id"], None)
            if pos is None:
                continue  # l'entrée correspondante avait été ignorée (capacité/cash)

            exit_price = ct["exit_price"]
            trade_value_usd = pos["shares"] * exit_price
            fees_usd = _commission_usd(pos["shares"], exit_price, sim_params) + _fx_fee_usd(trade_value_usd, sim_params)

            # long : on vend -> encaisse. short : on rachète -> décaisse.
            exit_cash_usd = (trade_value_usd - fees_usd) if pos["direction"] == "long" else -(trade_value_usd + fees_usd)
            exit_cash_eur = exit_cash_usd / rate
            cash_eur += exit_cash_eur

            fees_eur_exit = fees_usd / rate
            total_fees_eur += fees_eur_exit

            pnl_eur = pos["entry_cash_flow_eur"] + exit_cash_eur  # somme des deux flux = P&L net (marche pour long et short)
            pnl_pct = (pnl_eur / pos["cost_basis_eur"] * 100) if pos["cost_basis_eur"] else 0.0

            closed_trades.append(ClosedTrade(
                ticker=pos["ticker"], direction=pos["direction"],
                entry_date=pos["entry_date"], exit_date=date,
                shares=pos["shares"], entry_price_usd=pos["entry_price_usd"], exit_price_usd=exit_price,
                cost_basis_eur=pos["cost_basis_eur"],
                fees_eur=pos["entry_fees_eur"] + fees_eur_exit,
                pnl_eur=pnl_eur, pnl_pct=pnl_pct, exit_reason=ct["exit_reason"],
            ))

        # --- Entrées du jour (ordre alphabétique = déterministe) ---
        for ct in sorted(entries_by_date.get(date, []), key=lambda c: c["ticker"]):
            if len(open_positions) >= sim_params.max_positions:
                n_skip_capacity += 1
                continue

            current_equity_eur = cash_eur + mark_to_market(date)
            alloc_usd = current_equity_eur * sim_params.position_pct * rate
            entry_price = ct["entry_price"]
            shares = int(alloc_usd // entry_price)
            if shares < 1:
                n_skip_lot += 1
                continue

            trade_value_usd = shares * entry_price
            fees_usd = _commission_usd(shares, entry_price, sim_params) + _fx_fee_usd(trade_value_usd, sim_params)

            # long : on achète -> décaisse. short : on vend à découvert -> encaisse.
            entry_cash_usd = -(trade_value_usd + fees_usd) if ct["direction"] == "long" else (trade_value_usd - fees_usd)
            entry_cash_eur = entry_cash_usd / rate
            fees_eur_entry = fees_usd / rate

            # Vérif de trésorerie : uniquement pour les longs (un short encaisse
            # du cash à l'entrée ; la marge nécessaire pour le couvrir n'est pas
            # modélisée ici, cf. limitations en tête de fichier).
            if ct["direction"] == "long" and -entry_cash_eur > cash_eur:
                n_skip_cash += 1
                continue

            cash_eur += entry_cash_eur
            total_fees_eur += fees_eur_entry

            open_positions[ct["_id"]] = {
                "ticker": ct["ticker"], "direction": ct["direction"],
                "shares": shares, "entry_price_usd": entry_price,
                "entry_date": date, "entry_cash_flow_eur": entry_cash_eur,
                "entry_fees_eur": fees_eur_entry,
                "cost_basis_eur": trade_value_usd / rate,
            }

        equity_points[date] = cash_eur + mark_to_market(date)

    equity_curve = pd.Series(equity_points).sort_index()

    return PortfolioSimResult(
        equity_curve=equity_curve, closed_trades=closed_trades,
        n_skipped_capacity=n_skip_capacity, n_skipped_cash=n_skip_cash, n_skipped_lot=n_skip_lot,
        total_fees_eur=total_fees_eur, used_live_fx=used_live_fx,
    )
