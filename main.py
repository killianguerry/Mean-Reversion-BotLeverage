"""
main.py
-------
Point d'entrée en ligne de commande du bot mean reversion "haut winrate + levier".

Par défaut, lance le backtest sur les 100 actions du Nasdaq-100, en
récupérant et mettant en cache les données AUTOMATIQUEMENT (Wikipedia +
Yahoo Finance) — aucun CSV à fournir.

Exemples :

    # Backtest sur tout l'univers Nasdaq-100 (auto-fetch + cache)
    python main.py

    # Forcer un nouveau téléchargement même si le cache est frais
    python main.py --refresh-data

    # Backtest sur un seul ticker (toujours automatique, via yfinance)
    python main.py --ticker QQQ --period 3y

    # Backtest sur un seul ticker à partir d'un CSV local (mode avancé/hors-ligne)
    python main.py --csv data/mon_fichier.csv

    # Resserrer les entrées pour viser un winrate plus élevé
    python main.py --confirm-bars 2 --rsi-daily 40

    # Ajouter du levier (avec coût d'emprunt et liquidation simulée)
    python main.py --leverage 2 --confirm-bars 2 --rsi-daily 40

    # Simulation réaliste : capital unique en EUR, frais IBKR, 20 positions
    # max à 2% de l'équity chacune
    python main.py --realistic-sim --capital 100000
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from backtester import StrategyParams, run_backtest
from data_loader import load_csv, load_yfinance
from fx_data import get_eurusd_series_safe, DEFAULT_CACHE_PATH as DEFAULT_FX_CACHE_PATH
from nasdaq100_data import get_universe, DEFAULT_CACHE_PATH, DEFAULT_MAX_AGE_DAYS
from optimizer import optimize, params_from_row, METRIC_LABELS
from portfolio import run_universe
from portfolio_sim import PortfolioSimParams, run_portfolio_simulation


def parse_args():
    p = argparse.ArgumentParser(description="Backtester mean reversion (Bollinger + RSI + ADX + ATR)")

    src = p.add_mutually_exclusive_group()
    src.add_argument("--ticker", type=str, help="Backtest sur un seul ticker (téléchargé via yfinance)")
    src.add_argument("--csv", type=str, help="Backtest sur un seul ticker à partir d'un CSV local (mode avancé)")

    p.add_argument("--period", type=str, default="5y", help="Période d'historique (défaut : 5y)")
    p.add_argument("--refresh-data", action="store_true", help="Force le retéléchargement des données Nasdaq-100")
    p.add_argument("--max-age-days", type=float, default=DEFAULT_MAX_AGE_DAYS,
                    help="Âge max du cache en jours avant retéléchargement automatique (défaut : 1)")
    p.add_argument("--data-cache", type=str, default=DEFAULT_CACHE_PATH)

    p.add_argument("--bb-period", type=int, default=20)
    p.add_argument("--bb-std", type=float, default=1.5)
    p.add_argument("--rsi-long", type=float, default=50.0)
    p.add_argument("--rsi-short", type=float, default=45.0)
    p.add_argument("--adx-daily", type=float, default=15.0)
    p.add_argument("--adx-weekly", type=float, default=15.0)
    p.add_argument("--atr-mult", type=float, default=2.5)
    p.add_argument("--shorts", action="store_true", help="Activer les positions short")
    p.add_argument("--initial-equity", type=float, default=10_000.0)

    p.add_argument("--confirm-bars", type=int, default=1,
                    help="Nombre de clôtures consécutives requises hors bande de Bollinger "
                         "avant d'autoriser une entrée (défaut 1 = comportement d'origine, "
                         "simple touch). Augmenter (ex: 2-3) réduit le nombre de trades mais "
                         "vise un winrate plus élevé.")
    p.add_argument("--rsi-daily", type=float, default=None,
                    help="Filtre RSI daily additionnel (en plus du RSI hebdo déjà en place). "
                         "None = désactivé. Ex: 40 -> exige aussi RSI daily < 40 pour un long "
                         "(et > 60 pour un short), confluence multi-timeframe plus stricte.")
    p.add_argument("--leverage", type=float, default=1.0,
                    help="Multiplicateur de levier appliqué au P&L de chaque trade (1.0 = pas "
                         "de levier). ATTENTION : amplifie gains ET pertes, introduit un risque "
                         "de liquidation. Voir --liquidation-threshold et --borrow-rate.")
    p.add_argument("--borrow-rate", type=float, default=0.06,
                    help="Taux d'emprunt annuel (fraction, ex 0.06 = 6%%/an) appliqué au capital "
                         "emprunté si --leverage > 1. Coût réel du levier sur marge.")
    p.add_argument("--liquidation-threshold", type=float, default=0.9,
                    help="Perte (fraction du capital du trade, après levier) au-delà de laquelle "
                         "le trade est considéré liquidé de force (défaut 0.9 = -90%%).")

    p.add_argument("--optimize", action="store_true",
                    help="Lance l'optimiseur de réglages (recherche aléatoire + validation hors "
                         "échantillon) au lieu d'un backtest simple. Ignore les autres --bb-*, --rsi-*, "
                         "--adx-*, --atr-mult, --confirm-bars, --rsi-daily.")
    p.add_argument("--n-trials", type=int, default=60,
                    help="Nombre d'essais pour l'optimiseur (--optimize). Compte ~1-2s par essai sur "
                         "les 100 actions.")
    p.add_argument("--top-k", type=int, default=10,
                    help="Nombre de finalistes revalidés hors échantillon (--optimize)")
    p.add_argument("--opt-metric", type=str, default="strat_perf",
                    choices=["strat_perf", "perf_risk_ratio", "win_rate"],
                    help="Métrique à optimiser (--optimize) : strat_perf (performance totale), "
                         "perf_risk_ratio (performance/drawdown) ou win_rate.")

    p.add_argument("--out", type=str, default="output", help="Dossier de sortie")

    p.add_argument("--realistic-sim", action="store_true",
                    help="Lance la simulation de portefeuille réaliste (capital unique en EUR, "
                         "frais IBKR, conversion de change, max 20 positions simultanées à 2%% "
                         "de l'équity chacune) au lieu du mode 'moyenne par ticker' par défaut. "
                         "Ne s'applique qu'à l'univers complet (pas --ticker ni --csv).")
    p.add_argument("--capital", type=float, default=100_000.0,
                    help="Capital initial en EUR (--realistic-sim)")
    p.add_argument("--max-positions", type=int, default=20,
                    help="Nombre maximum de positions simultanées (--realistic-sim)")
    p.add_argument("--position-pct", type=float, default=2.0,
                    help="Taille de chaque position, en %% de l'équity courante (--realistic-sim)")
    p.add_argument("--commission-per-share", type=float, default=0.005,
                    help="Commission IBKR par action en USD (--realistic-sim)")
    p.add_argument("--commission-min", type=float, default=1.0,
                    help="Commission minimum par ordre en USD (--realistic-sim)")
    p.add_argument("--commission-max-pct", type=float, default=1.0,
                    help="Plafond de commission, en %% de la valeur de l'ordre (--realistic-sim)")
    p.add_argument("--fx-fee-pct", type=float, default=0.03,
                    help="Frais de conversion de change, en %% du montant converti (--realistic-sim)")
    p.add_argument("--fx-fee-min", type=float, default=2.0,
                    help="Frais de conversion de change minimum en USD (--realistic-sim)")
    p.add_argument("--eur-usd-rate", type=float, default=1.08,
                    help="Taux EUR/USD de repli si l'historique n'est pas disponible (--realistic-sim)")
    p.add_argument("--no-fx-history", action="store_true",
                    help="N'essaie pas de télécharger l'historique EUR/USD réel, utilise "
                         "directement le taux constant --eur-usd-rate (--realistic-sim)")

    return p.parse_args()


def build_params(args) -> StrategyParams:
    return StrategyParams(
        bb_period=args.bb_period,
        bb_std=args.bb_std,
        rsi_long_threshold=args.rsi_long,
        rsi_short_threshold=args.rsi_short,
        adx_daily_threshold=args.adx_daily,
        adx_weekly_threshold=args.adx_weekly,
        atr_mult=args.atr_mult,
        enable_shorts=args.shorts,
        initial_equity=args.initial_equity,
        confirm_bars=args.confirm_bars,
        rsi_daily_threshold=args.rsi_daily,
        leverage=args.leverage,
        leverage_borrow_rate_annual=args.borrow_rate,
        liquidation_threshold=args.liquidation_threshold,
    )


def run_single(df: pd.DataFrame, params: StrategyParams, out_dir: str, label: str):
    result = run_backtest(df, params)
    print("\n=== Résultats du backtest ===")
    print(result.summary())

    os.makedirs(out_dir, exist_ok=True)
    trades_df = pd.DataFrame([t.__dict__ for t in result.trades])
    trades_df.to_csv(os.path.join(out_dir, "trades.csv"), index=False)

    buy_hold = params.initial_equity * df["Close"] / df["Close"].iloc[0]
    fig, ax = plt.subplots(figsize=(10, 5))
    result.equity_curve.plot(ax=ax, label="Stratégie", linewidth=2)
    buy_hold.reindex(result.equity_curve.index).plot(ax=ax, label="Buy & Hold", linestyle="--")
    ax.set_title(f"Backtest Mean Reversion — {label}")
    ax.set_ylabel("Équity ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(out_dir, "equity_curve.png"), dpi=150, bbox_inches="tight")
    print(f"Résultats sauvegardés dans : {out_dir}/")


def run_optimize(args):
    print("Récupération des données Nasdaq-100 (cache automatique si disponible)...")
    try:
        universe = get_universe(
            cache_path=args.data_cache,
            max_age_days=args.max_age_days,
            force_refresh=args.refresh_data,
            period=args.period,
        )
    except RuntimeError as exc:
        print(f"\nErreur : {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"{len(universe)} tickers prêts.\n")
    print(f"Optimisation : {args.n_trials} essais, métrique = {METRIC_LABELS.get(args.opt_metric, args.opt_metric)}")
    print("(recherche sur les 70% premiers jours, puis validation sur les 30% restants, jamais vus)\n")

    def progress(frac, msg):
        print(f"\r[{frac * 100:5.1f}%] {msg:60s}", end="", flush=True)

    try:
        test_df, train_df, split_date = optimize(
            universe, n_trials=args.n_trials, top_k=args.top_k, metric=args.opt_metric,
            progress_callback=progress,
        )
    except RuntimeError as exc:
        print(f"\nErreur : {exc}", file=sys.stderr)
        sys.exit(1)
    print("\n")

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "optimizer_results.csv")
    test_df.to_csv(out_path, index=False)

    print(f"=== {len(test_df)} configurations validées (entraînement avant le {split_date.date()}, test après) ===\n")
    display_cols = [c for c in ["strat_perf", f"train_{args.opt_metric}", "bh_perf", "win_rate",
                                 "max_dd", "n_trades", "confirm_bars", "rsi_daily_threshold"] if c in test_df.columns]
    print(test_df[display_cols].to_string(index=False))
    print(f"\nRésultats complets : {out_path}")

    best = params_from_row(test_df.iloc[0])
    print("\n=== Meilleure configuration (classée sur la performance de TEST, hors échantillon) ===")
    print(best)
    print(
        f"\nPour relancer un backtest avec cette configuration :\n"
        f"  python main.py --bb-period {best.bb_period} --bb-std {best.bb_std} "
        f"--rsi-long {best.rsi_long_threshold} --adx-daily {best.adx_daily_threshold} "
        f"--adx-weekly {best.adx_weekly_threshold} --atr-mult {best.atr_mult} "
        f"--confirm-bars {best.confirm_bars} --rsi-daily {best.rsi_daily_threshold}"
    )


def run_full_universe(args, params: StrategyParams):
    print("Récupération des données Nasdaq-100 (cache automatique si disponible)...")
    try:
        universe = get_universe(
            cache_path=args.data_cache,
            max_age_days=args.max_age_days,
            force_refresh=args.refresh_data,
            period=args.period,
        )
    except RuntimeError as exc:
        print(f"\nErreur : {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"{len(universe)} tickers prêts pour le backtest.\n")

    summary_df, portfolio_curve, portfolio_bh, details = run_universe(universe, params)

    os.makedirs(args.out, exist_ok=True)
    summary_path = os.path.join(args.out, "summary_par_ticker.csv")
    summary_df.to_csv(summary_path, index=False)

    print("=== Résultats agrégés — univers Nasdaq-100 ===")
    print(f"Tickers testés          : {len(summary_df)}")
    print(f"Performance moyenne     : {summary_df['Performance (%)'].mean():+.2f}%")
    print(f"Performance médiane     : {summary_df['Performance (%)'].median():+.2f}%")
    print(f"Buy & hold (référence)  : {(portfolio_bh.iloc[-1] / portfolio_bh.iloc[0] - 1) * 100:+.2f}%")
    print(f"Total trades            : {summary_df['Nb trades'].sum()}")
    print(f"Win rate moyen          : {summary_df['Win rate (%)'].mean():.1f}%")
    print(f"\nTop 5 :\n{summary_df.head(5).to_string(index=False)}")
    print(f"\nMoins bons :\n{summary_df.tail(5).to_string(index=False)}")
    print(f"\nRésumé complet : {summary_path}")

    fig, ax = plt.subplots(figsize=(10, 5))
    portfolio_curve.plot(ax=ax, label="Stratégie", linewidth=2)
    portfolio_bh.plot(ax=ax, label="Buy & Hold (moyenne)", linestyle="--")
    ax.set_title("Portefeuille équipondéré Nasdaq-100 (base 100)")
    ax.set_ylabel("Valeur (base 100)")
    ax.legend()
    ax.grid(alpha=0.3)
    plot_path = os.path.join(args.out, "portfolio_equity_curve.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Graphique du portefeuille : {plot_path}")


def build_sim_params(args) -> PortfolioSimParams:
    return PortfolioSimParams(
        initial_capital_eur=args.capital,
        max_positions=args.max_positions,
        position_pct=args.position_pct / 100,
        commission_per_share_usd=args.commission_per_share,
        commission_min_usd=args.commission_min,
        commission_max_pct_of_trade=args.commission_max_pct / 100,
        fx_fee_pct=args.fx_fee_pct / 100,
        fx_fee_min_usd=args.fx_fee_min,
        eur_usd_fallback_rate=args.eur_usd_rate,
    )


def run_realistic_sim(args, params: StrategyParams):
    print("Récupération des données Nasdaq-100 (cache automatique si disponible)...")
    try:
        universe = get_universe(
            cache_path=args.data_cache,
            max_age_days=args.max_age_days,
            force_refresh=args.refresh_data,
            period=args.period,
        )
    except RuntimeError as exc:
        print(f"\nErreur : {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"{len(universe)} tickers prêts.\n")

    sim_params = build_sim_params(args)

    eur_usd_rate, used_live_fx = None, False
    if not args.no_fx_history:
        print("Récupération de l'historique EUR/USD (cache automatique si disponible)...")
        eur_usd_rate, used_live_fx = get_eurusd_series_safe(
            period=args.period, cache_path=DEFAULT_FX_CACHE_PATH,
            fallback_rate=args.eur_usd_rate,
        )
        if not used_live_fx:
            print(f"  -> historique indisponible, taux constant {args.eur_usd_rate} utilisé.\n")
        else:
            print(f"  -> {len(eur_usd_rate)} points chargés.\n")
    else:
        used_live_fx = False

    def progress(frac, msg):
        print(f"\r[{frac * 100:5.1f}%] {msg:60s}", end="", flush=True)

    result = run_portfolio_simulation(
        universe, params, sim_params,
        eur_usd_rate=eur_usd_rate if not args.no_fx_history else None,
        used_live_fx=used_live_fx,
        progress_callback=progress,
    )
    print("\n")

    print("=== Simulation de portefeuille réaliste (capital unique, frais IBKR, EUR) ===")
    print(result.summary())

    os.makedirs(args.out, exist_ok=True)
    trades_df = pd.DataFrame([t.__dict__ for t in result.closed_trades])
    trades_path = os.path.join(args.out, "trades_realistic_sim.csv")
    trades_df.to_csv(trades_path, index=False)
    print(f"\nTrades détaillés : {trades_path}")

    fig, ax = plt.subplots(figsize=(10, 5))
    result.equity_curve.plot(ax=ax, label="Portefeuille (capital réel)", linewidth=2, color="#2563eb")
    ax.set_title("Simulation de portefeuille réaliste — capital unique en EUR")
    ax.set_ylabel("Équity (€)")
    ax.legend()
    ax.grid(alpha=0.3)
    plot_path = os.path.join(args.out, "portfolio_sim_equity_curve.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Graphique : {plot_path}")


def main():
    args = parse_args()

    if args.optimize:
        run_optimize(args)
        return

    params = build_params(args)

    if args.csv:
        df = load_csv(args.csv)
        print(f"Données chargées (CSV) : {len(df)} bougies, de {df.index.min().date()} à {df.index.max().date()}")
        run_single(df, params, args.out, label=os.path.basename(args.csv))
    elif args.ticker:
        df = load_yfinance(args.ticker, period=args.period)
        print(f"Données chargées ({args.ticker}) : {len(df)} bougies, de {df.index.min().date()} à {df.index.max().date()}")
        run_single(df, params, args.out, label=args.ticker)
    elif args.realistic_sim:
        run_realistic_sim(args, params)
    else:
        run_full_universe(args, params)


if __name__ == "__main__":
    sys.exit(main())
