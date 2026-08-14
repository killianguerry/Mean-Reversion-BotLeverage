"""
streamlit_app.py
------------------
Interface web du bot de mean reversion sur le Nasdaq-100.

Les données (liste des tickers + historique de prix) sont récupérées et
mises en cache AUTOMATIQUEMENT au premier lancement (Wikipedia + Yahoo
Finance). Aucun CSV à fournir, aucune action manuelle requise. Le cache est
réutilisé tant qu'il a moins de 24h ; un bouton permet de forcer un
rafraîchissement.

Lancement local :
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from backtester import StrategyParams, run_backtest
from fx_data import get_eurusd_series_safe
from indicators import bollinger_bands
from nasdaq100_data import get_universe, DEFAULT_CACHE_PATH, DEFAULT_MAX_AGE_DAYS
from optimizer import optimize, params_from_row, METRIC_LABELS
from portfolio import run_universe
from portfolio_sim import PortfolioSimParams, run_portfolio_simulation

st.set_page_config(page_title="Mean Reversion Bot — Haut Winrate + Levier — Nasdaq-100", layout="wide")

# ---------------------------------------------------------------------------
# Style : cartes de métriques
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .metric-label {font-size:0.78rem; color:#6b7280; text-transform:uppercase;
                   letter-spacing:0.03em; margin-bottom:0.15rem;}
    .metric-value {font-size:1.7rem; font-weight:700; line-height:1.2;}
    .metric-value.positive {color:#16a34a;}
    .metric-value.negative {color:#dc2626;}
    .metric-value.neutral {color:#111827;}
    </style>
    """,
    unsafe_allow_html=True,
)


def tone_for(value: float) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "neutral"


def metric_card(col, label: str, value_str: str, tone: str = "neutral"):
    col.markdown(
        f"""<div class="metric-label">{label}</div>
        <div class="metric-value {tone}">{value_str}</div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Profils de réglages : sauvegarde/chargement dans un fichier JSON local.
# NB : comme pour le cache de données, ce fichier vit sur le disque de
# l'instance en cours. Il survit aux rechargements de page et aux sessions,
# mais peut être réinitialisé si l'app est redéployée/rebootée sur Streamlit
# Cloud — d'où le bouton d'export/import JSON pour en garder une copie.
# ---------------------------------------------------------------------------
PROFILES_PATH = Path("data/profiles.json")

# Clés de widgets (via key=) qui composent un profil complet
PROFILE_KEYS = [
    "bb_period", "bb_std", "rsi_long", "rsi_short", "adx_daily", "adx_weekly",
    "atr_mult", "enable_shorts", "confirm_bars", "rsi_daily_enabled", "rsi_daily",
    "leverage", "borrow_rate", "liquidation_threshold",
]


def load_profiles() -> dict:
    if PROFILES_PATH.exists():
        try:
            return json.loads(PROFILES_PATH.read_text())
        except Exception:
            return {}
    return {}


def save_profiles(profiles: dict) -> None:
    PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILES_PATH.write_text(json.dumps(profiles, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Récupération des données (auto-fetch + cache, sans CSV)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_universe_cached(cache_path: str, max_age_days: float, force_refresh: bool, period: str):
    """Enveloppe get_universe() dans le cache Streamlit. force_refresh est
    inclus dans la clé de cache : cliquer sur 'rafraîchir' invalide bien le
    cache Streamlit et déclenche un nouveau téléchargement."""
    return get_universe(
        cache_path=cache_path,
        max_age_days=max_age_days,
        force_refresh=force_refresh,
        period=period,
    )


@st.cache_data(show_spinner=False)
def backtest_ticker(ticker: str, df: pd.DataFrame, params_dict: dict):
    params = StrategyParams(**params_dict)
    result = run_backtest(df, params)
    mid, upper, lower = bollinger_bands(df["Close"], params.bb_period, params.bb_std)
    return result, mid, upper, lower


def make_equity_fig(result, buy_hold: pd.Series, title: str):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    result.equity_curve.plot(ax=ax, label="Stratégie", linewidth=2, color="#2563eb")
    buy_hold.reindex(result.equity_curve.index).plot(ax=ax, label="Buy & Hold", linestyle="--", color="#16a34a")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Équity ($)")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def equity_chart_png(ticker: str, result, buy_hold: pd.Series, title: str) -> bytes:
    """Même logique de cache que price_chart_png : évite de redessiner ce
    graphique à chaque interaction (le code d'un expander s'exécute même
    quand il est visuellement fermé)."""
    fig = make_equity_fig(result, buy_hold, title)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def make_price_fig(df: pd.DataFrame, upper, lower, result):
    """Prix + bandes de Bollinger sur tout l'historique, avec le point
    d'entrée et de sortie de CHAQUE trade, reliés par une ligne verte
    (trade gagnant) ou rouge (trade perdant)."""
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df.index, df["Close"], label="Prix (close)", color="#2563eb", linewidth=1.1, zorder=2)
    ax.plot(upper.index, upper, label="Bande haute (écart-type +)", color="#dc2626", linewidth=0.8, alpha=0.6, zorder=1)
    ax.plot(lower.index, lower, label="Bande basse (écart-type -)", color="#16a34a", linewidth=0.8, alpha=0.6, zorder=1)

    # relie chaque entrée à sa sortie, coloré selon le résultat du trade
    for t in result.trades:
        if t.exit_date is not None:
            color = "#16a34a" if (t.pnl_pct or 0) >= 0 else "#dc2626"
            ax.plot([t.entry_date, t.exit_date], [t.entry_price, t.exit_price],
                    linestyle=":", color=color, linewidth=1.1, alpha=0.7, zorder=4)

    entries = [(t.entry_date, t.entry_price) for t in result.trades]
    exits = [(t.exit_date, t.exit_price) for t in result.trades if t.exit_date is not None]

    if entries:
        ex, ey = zip(*entries)
        ax.scatter(ex, ey, marker="^", color="#16a34a", s=70, label="Entrée", zorder=5, edgecolor="white", linewidth=0.6)
    if exits:
        xx, xy = zip(*exits)
        ax.scatter(xx, xy, marker="D", color="#dc2626", s=50, label="Sortie", zorder=5, edgecolor="white", linewidth=0.6)

    ax.legend(loc="upper left", frameon=False, fontsize=8, ncol=3)
    ax.grid(alpha=0.25)
    ax.set_title(f"Historique complet — {len(result.trades)} trade(s)", fontsize=8, color="#6b7280")
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def price_chart_png(ticker: str, df: pd.DataFrame, upper: pd.Series, lower: pd.Series, result) -> bytes:
    """Rend le graphique prix/Bollinger/entrées-sorties en PNG et met le
    résultat en cache. Sans ce cache, chaque interaction dans l'app (taper
    dans la recherche, bouger un curseur) redessinerait les 100 graphiques
    à chaque fois — avec le cache, seul le premier rendu par ticker+réglages
    coûte du temps, les reruns suivants sont quasi instantanés."""
    fig = make_price_fig(df, upper, lower, result)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Sidebar : paramètres + gestion des données
# ---------------------------------------------------------------------------
st.title("📉 Mean Reversion Bot — Haut Winrate + Levier — Nasdaq-100")
st.caption("Bollinger Bands + RSI multi-timeframe + ADX, sortie mean reversion pure (winrate maximisé), levier optionnel.")

st.sidebar.header("Données")
if st.sidebar.button("🔄 Forcer le rafraîchissement des données"):
    st.cache_data.clear()
    force_refresh = True
else:
    force_refresh = False

# ---------------------------------------------------------------------------
# Profils de réglages
# ---------------------------------------------------------------------------
if "profiles" not in st.session_state:
    st.session_state.profiles = load_profiles()

st.sidebar.header("Profils de réglages")

profile_names = sorted(st.session_state.profiles.keys())
chosen = st.sidebar.selectbox(
    "Profil enregistré", ["(aucun)"] + profile_names, key="profile_picker",
    help="Choisis un profil précédemment sauvegardé, puis clique sur Charger "
         "pour appliquer tous ses réglages (curseurs + mode de sortie) d'un coup.",
)
col_load, col_del = st.sidebar.columns(2)
if col_load.button("📂 Charger", disabled=(chosen == "(aucun)"), width="stretch"):
    for k, v in st.session_state.profiles[chosen].items():
        st.session_state[k] = v
    st.sidebar.success(f"Profil « {chosen} » chargé.")
    st.rerun()
if col_del.button("🗑️ Supprimer", disabled=(chosen == "(aucun)"), width="stretch"):
    del st.session_state.profiles[chosen]
    save_profiles(st.session_state.profiles)
    st.rerun()

with st.sidebar.expander("💾 Sauvegarder / renommer / importer"):
    new_name = st.text_input("Nom du nouveau profil", key="new_profile_name")
    if st.button("💾 Sauvegarder les réglages actuels", width="stretch"):
        name = new_name.strip()
        if not name:
            st.warning("Donne un nom au profil avant de sauvegarder.")
        else:
            st.session_state.profiles[name] = {
                k: st.session_state[k] for k in PROFILE_KEYS if k in st.session_state
            }
            save_profiles(st.session_state.profiles)
            st.success(f"Profil « {name} » sauvegardé.")
            st.rerun()

    if profile_names:
        st.divider()
        rename_target = st.selectbox("Renommer un profil", profile_names, key="rename_target")
        rename_new = st.text_input("Nouveau nom", key="rename_new_name")
        if st.button("✏️ Renommer", width="stretch"):
            new_clean = rename_new.strip()
            if not new_clean:
                st.warning("Indique le nouveau nom.")
            else:
                st.session_state.profiles[new_clean] = st.session_state.profiles.pop(rename_target)
                save_profiles(st.session_state.profiles)
                st.success(f"« {rename_target} » renommé en « {new_clean} ».")
                st.rerun()

    st.divider()
    st.caption(
        "⚠️ Les profils sont stockés sur le serveur de l'app et peuvent être "
        "réinitialisés si elle est redéployée. Exporte-les régulièrement pour "
        "garder une sauvegarde."
    )
    st.download_button(
        "⬇️ Exporter tous les profils (JSON)",
        data=json.dumps(st.session_state.profiles, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name="profils_bot.json", mime="application/json",
        width="stretch", disabled=not profile_names,
    )
    uploaded_profiles = st.file_uploader("⬆️ Importer des profils (JSON)", type=["json"])
    if uploaded_profiles is not None:
        try:
            imported = json.loads(uploaded_profiles.read())
            st.session_state.profiles.update(imported)
            save_profiles(st.session_state.profiles)
            st.success(f"{len(imported)} profil(s) importé(s).")
            st.rerun()
        except Exception as exc:
            st.error(f"Fichier invalide : {exc}")

_WIDGET_DEFAULTS = {
    "bb_period": 20, "bb_std": 1.5, "rsi_long": 50, "rsi_short": 45,
    "adx_daily": 15, "adx_weekly": 15, "atr_mult": 2.5, "enable_shorts": False,
    "confirm_bars": 1, "rsi_daily_enabled": False, "rsi_daily": 40,
    "leverage": 1.0, "borrow_rate": 6.0, "liquidation_threshold": 90,
}
for _k, _v in _WIDGET_DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

# Applique une config choisie dans l'optimiseur, si l'utilisateur vient de
# cliquer sur "Appliquer". DOIT s'exécuter ici, avant que les curseurs ne
# soient instanciés plus bas — Streamlit interdit de modifier session_state
# pour un widget déjà créé dans le run courant.
if "_pending_apply" in st.session_state:
    for _k, _v in st.session_state.pop("_pending_apply").items():
        st.session_state[_k] = _v

st.sidebar.header("Paramètres de la stratégie")
bb_period = st.sidebar.slider(
    "Période Bollinger (jours)", 10, 40, key="bb_period",
    help="Nombre de jours utilisés pour calculer la moyenne et l'écart-type du "
         "prix (les bandes de Bollinger). Une valeur courte rend les bandes plus "
         "réactives aux mouvements récents ; une valeur longue les rend plus lisses "
         "et plus stables.",
)
bb_std = st.sidebar.slider(
    "Écart-type Bollinger", 1.0, 3.0, step=0.5, key="bb_std",
    help="Largeur des bandes de Bollinger, exprimée en nombre d'écarts-types autour "
         "du prix moyen. Plus la valeur est élevée, plus les bandes sont larges : les "
         "signaux (prix qui sort des bandes) deviennent plus rares mais plus extrêmes.",
)
rsi_long = st.sidebar.slider(
    "RSI hebdo seuil long (>)", 45, 65, key="rsi_long",
    help="Le RSI hebdomadaire (indicateur de force de la tendance sur une échelle de "
         "0 à 100) doit être AU-DESSUS de ce seuil pour autoriser un achat. Sert à ne "
         "prendre des positions longues que si la tendance de fond est haussière.",
)
rsi_short = st.sidebar.slider(
    "RSI hebdo seuil short (<)", 30, 50, key="rsi_short",
    help="Le RSI hebdomadaire doit être EN-DESSOUS de ce seuil pour autoriser une "
         "vente à découvert (short). Sert à ne shorter que si la tendance de fond est "
         "baissière. N'a d'effet que si les positions short sont activées ci-dessous.",
)
adx_daily = st.sidebar.slider(
    "ADX daily seuil", 10, 35, key="adx_daily",
    help="L'ADX mesure la force d'une tendance (peu importe sa direction), sur une "
         "échelle de 0 à 100. Ce seuil, appliqué aux données journalières, filtre les "
         "marchés qui stagnent en range plat : plus il est élevé, plus on exige une "
         "tendance marquée avant d'entrer en position.",
)
adx_weekly = st.sidebar.slider(
    "ADX hebdo seuil", 10, 35, key="adx_weekly",
    help="Même principe que l'ADX daily, mais calculé sur les données hebdomadaires. "
         "Permet de confirmer qu'il y a bien une tendance de fond, pas seulement un "
         "mouvement de court terme.",
)
atr_mult = st.sidebar.slider(
    "Multiplicateur ATR (stop-loss initial)", 1.0, 6.0, step=0.5, key="atr_mult",
    help="L'ATR mesure la volatilité récente du prix. Ce multiplicateur détermine la "
         "distance du stop-loss initial par rapport au prix d'entrée, en multiples de "
         "l'ATR. Une valeur élevée place le stop plus loin (moins de sorties "
         "prématurées, mais perte potentielle plus grande si le trade tourne mal).",
)
enable_shorts = st.sidebar.checkbox(
    "Activer les positions short", key="enable_shorts",
    help="Si activé, le bot peut aussi parier à la baisse (vendre à découvert), en "
         "plus des achats classiques. Désactivé par défaut car le short comporte des "
         "coûts et des contraintes réglementaires que l'achat n'a pas.",
)

st.sidebar.header("Confluence haut winrate")
confirm_bars = st.sidebar.slider(
    "Clôtures consécutives hors bande requises", 1, 4, key="confirm_bars",
    help="Nombre de clôtures CONSÉCUTIVES exigées hors de la bande de Bollinger "
         "avant d'autoriser une entrée (1 = un simple touch suffit, comportement "
         "de base). Augmenter réduit le nombre de trades mais vise un winrate "
         "plus élevé (signal plus persistant, moins de faux départs).",
)
rsi_daily_enabled = st.sidebar.checkbox(
    "Filtre RSI daily additionnel", key="rsi_daily_enabled",
    help="Ajoute une confluence multi-timeframe : en plus du RSI hebdomadaire, "
         "exige aussi que le RSI daily soit en zone de faiblesse (long) ou de "
         "force (short).",
)
rsi_daily_threshold = None
if rsi_daily_enabled:
    rsi_daily_threshold = st.sidebar.slider(
        "Seuil RSI daily", 20, 50, key="rsi_daily",
        help="Pour un long : exige RSI daily < ce seuil. Pour un short : exige "
             "RSI daily > (100 - ce seuil). Plus la valeur est basse, plus le "
             "filtre est strict.",
    )

st.sidebar.header("Levier")
leverage = st.sidebar.slider(
    "Multiplicateur de levier", 1.0, 5.0, step=0.5, key="leverage",
    help="Multiplie le P&L de chaque trade. ATTENTION : n'améliore jamais "
         "l'edge de la stratégie, amplifie gains ET pertes, et introduit un "
         "risque de liquidation qu'une position sans levier n'a pas. 1.0 = "
         "pas de levier.",
)
borrow_rate = st.sidebar.slider(
    "Taux d'emprunt annuel (%)", 0.0, 15.0, step=0.5, key="borrow_rate",
    help="Coût annuel appliqué au capital emprunté (levier - 1), au prorata "
         "du temps de détention de chaque trade. Un backtest sans ce coût "
         "surestime la performance réelle du levier.",
)
liquidation_threshold_pct = st.sidebar.slider(
    "Seuil de liquidation (%)", 50, 99, key="liquidation_threshold",
    help="Si la perte sur un trade, une fois le levier appliqué, atteint ce "
         "pourcentage du capital du trade, la position est considérée "
         "liquidée de force (appel de marge), quel que soit le stop-loss "
         "théorique.",
)

params_dict = dict(
    bb_period=bb_period, bb_std=bb_std,
    rsi_long_threshold=rsi_long, rsi_short_threshold=rsi_short,
    adx_daily_threshold=adx_daily, adx_weekly_threshold=adx_weekly,
    atr_mult=atr_mult, enable_shorts=enable_shorts,
    confirm_bars=confirm_bars, rsi_daily_threshold=rsi_daily_threshold,
    leverage=leverage, leverage_borrow_rate_annual=borrow_rate / 100,
    liquidation_threshold=liquidation_threshold_pct / 100,
)

# --- Récupération automatique des données (spinner pendant le 1er téléchargement) ---
with st.spinner(
    "Récupération des données Nasdaq-100 (Wikipedia + Yahoo Finance)... "
    "peut prendre plusieurs minutes la première fois, instantané ensuite (cache)."
):
    try:
        per_ticker = load_universe_cached(DEFAULT_CACHE_PATH, DEFAULT_MAX_AGE_DAYS, force_refresh, "5y")
    except RuntimeError as exc:
        st.error(
            "Impossible de récupérer les données Nasdaq-100 automatiquement.\n\n"
            f"{exc}"
        )
        st.stop()

st.sidebar.success(f"{len(per_ticker)} tickers chargés (cache local, données réelles).")


@st.cache_data(show_spinner=False)
def build_export_csv(per_ticker: dict) -> bytes:
    """Reconstruit un CSV long (Date, Ticker, Open, High, Low, Close) à
    partir des données en mémoire, pour permettre de télécharger les
    données réelles directement depuis l'app (utile si le bot tourne sur
    Streamlit Cloud, où il n'y a pas d'accès direct au système de fichiers)."""
    frames = []
    for ticker, df in per_ticker.items():
        sub = df.reset_index()
        sub["Ticker"] = ticker
        frames.append(sub)
    long_df = pd.concat(frames, ignore_index=True)
    return long_df.to_csv(index=False).encode("utf-8")


st.sidebar.download_button(
    "📥 Télécharger les données (CSV)",
    data=build_export_csv(per_ticker),
    file_name="nasdaq100_universe.csv",
    mime="text/csv",
    help="Exporte les données réelles actuellement chargées (100 actions, "
         "format long Date/Ticker/OHLC) pour analyse externe.",
)

# ---------------------------------------------------------------------------
# Backtest de chaque ticker
# ---------------------------------------------------------------------------
if not per_ticker:
    st.error(
        "Aucune donnée disponible : le dictionnaire de tickers chargé est vide. "
        "Essaie de forcer le rafraîchissement des données dans la barre latérale."
    )
    st.stop()

tickers_sorted = sorted(per_ticker.keys())
progress = st.progress(0.0, text="Calcul des backtests...")

summaries = []
per_ticker_results = {}
backtest_errors = []  # (ticker, message) pour diagnostic si tout échoue

for i, ticker in enumerate(tickers_sorted, start=1):
    df = per_ticker[ticker]
    try:
        result, mid, upper, lower = backtest_ticker(ticker, df, params_dict)
    except Exception as exc:
        backtest_errors.append((ticker, str(exc)))
        progress.progress(i / len(tickers_sorted))
        continue

    buy_hold = 10_000 * df["Close"] / df["Close"].iloc[0]
    long_trades = sum(1 for t in result.trades if t.direction == "long")
    short_trades = sum(1 for t in result.trades if t.direction == "short")

    summaries.append({
        "Ticker": ticker,
        "Performance (%)": result.total_return_pct,
        "Win rate (%)": result.win_rate_pct,
        "Nb trades": result.num_trades,
        "Max drawdown (%)": result.max_drawdown_pct,
    })
    per_ticker_results[ticker] = dict(
        df=df, result=result, upper=upper, lower=lower,
        buy_hold=buy_hold, long_trades=long_trades, short_trades=short_trades,
    )
    progress.progress(i / len(tickers_sorted), text=f"Backtest {ticker} ({i}/{len(tickers_sorted)})")

progress.empty()
summary_df = pd.DataFrame(summaries)

if summary_df.empty:
    st.error(
        f"Le backtest a échoué sur les {len(tickers_sorted)} tickers chargés "
        f"({len(backtest_errors)} erreur(s)). Détail des premières erreurs :"
    )
    for ticker, msg in backtest_errors[:5]:
        st.code(f"{ticker} : {msg}")
    st.stop()
elif backtest_errors:
    st.warning(
        f"{len(backtest_errors)} ticker(s) sur {len(tickers_sorted)} ont échoué "
        f"et sont exclus des résultats (ex: {backtest_errors[0][0]} — {backtest_errors[0][1]})."
    )

# ---------------------------------------------------------------------------
# Section 1 : résultats agrégés (portefeuille équipondéré)
# ---------------------------------------------------------------------------
st.subheader(f"Résultats agrégés ({len(summary_df)} actions, données réelles Nasdaq-100)")

normalized_curves = {t: r["result"].equity_curve / r["result"].equity_curve.iloc[0] * 100
                     for t, r in per_ticker_results.items()}
portfolio_curve = pd.DataFrame(normalized_curves).mean(axis=1, skipna=True)

normalized_bh = {t: r["buy_hold"] / r["buy_hold"].iloc[0] * 100 for t, r in per_ticker_results.items()}
portfolio_bh = pd.DataFrame(normalized_bh).mean(axis=1, skipna=True)

avg_perf = summary_df["Performance (%)"].mean()
avg_bh_perf = (portfolio_bh.iloc[-1] / portfolio_bh.iloc[0] - 1) * 100
avg_win_rate = summary_df["Win rate (%)"].mean()
total_trades = int(summary_df["Nb trades"].sum())
avg_dd = summary_df["Max drawdown (%)"].mean()
total_long = sum(r["long_trades"] for r in per_ticker_results.values())
total_short = sum(r["short_trades"] for r in per_ticker_results.values())

c1, c2, c3, c4 = st.columns(4)
metric_card(c1, "Performance moyenne", f"{avg_perf:+.1f}%", tone_for(avg_perf))
metric_card(c2, "Buy & hold (référence)", f"{avg_bh_perf:+.1f}%", tone_for(avg_bh_perf))
metric_card(c3, "Win rate moyen", f"{avg_win_rate:.0f}%")
metric_card(c4, "Nb. de trades (total)", f"{total_trades}")

c5, c6, c7 = st.columns(3)
metric_card(c5, "Max drawdown moyen", f"{avg_dd:.1f}%", "negative")
metric_card(c6, "Trades longs", f"{total_long}")
metric_card(c7, "Trades shorts", f"{total_short}")

st.markdown("**Portefeuille équipondéré (base 100) : Stratégie vs Buy & Hold**")
fig, ax = plt.subplots(figsize=(11, 3.5))
portfolio_curve.plot(ax=ax, label="Stratégie", linewidth=2, color="#2563eb")
portfolio_bh.plot(ax=ax, label="Buy & Hold Nasdaq-100 (moyenne des 100 actions)", linestyle="--", color="#16a34a")
ax.set_ylabel("Valeur (base 100)")
ax.legend(loc="upper left", frameon=False)
ax.grid(alpha=0.25)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.markdown("**Classement des actions par performance**")
st.dataframe(
    summary_df.sort_values("Performance (%)", ascending=False).style.format({
        "Performance (%)": "{:+.2f}", "Win rate (%)": "{:.1f}", "Max drawdown (%)": "{:.2f}",
    }),
    width="stretch",
    height=300,
)

# ---------------------------------------------------------------------------
# Section Optimiseur : recherche automatique de réglages, avec validation
# train/test intégrée (même protocole que celui suivi manuellement tout au
# long du projet, pour éviter le sur-ajustement).
# ---------------------------------------------------------------------------
def strategy_params_to_ui_dict(p: StrategyParams) -> dict:
    """Traduit un StrategyParams en dict {clé de widget: valeur}. Ne touche
    PAS session_state directement — Streamlit interdit de modifier
    session_state pour un widget déjà instancié dans le run courant. Le
    résultat est stocké dans une clé tampon ('_pending_apply') et appliqué
    au tout début du script suivant, avant que les curseurs ne soient créés."""
    return {
        "bb_period": p.bb_period,
        "bb_std": p.bb_std,
        "rsi_long": p.rsi_long_threshold,
        "rsi_short": p.rsi_short_threshold,
        "adx_daily": p.adx_daily_threshold,
        "adx_weekly": p.adx_weekly_threshold,
        "atr_mult": p.atr_mult,
        "enable_shorts": p.enable_shorts,
        "confirm_bars": p.confirm_bars,
        "rsi_daily_enabled": p.rsi_daily_threshold is not None,
        "rsi_daily": p.rsi_daily_threshold if p.rsi_daily_threshold is not None else 40,
    }


st.subheader("🔍 Optimiseur de réglages")
with st.expander("Rechercher automatiquement de meilleurs réglages", expanded=False):
    st.caption(
        "Teste un grand nombre de combinaisons au hasard : optimise sur les 70% "
        "premiers jours de données (entraînement), puis revalide les meilleurs "
        "candidats sur les 30% restants, jamais vus pendant la recherche. "
        "**Seule la colonne « Perf. test » doit orienter ton choix** — la colonne "
        "« Perf. train » peut être optimiste (sur-ajustement)."
    )
    oc1, oc2 = st.columns(2)
    opt_n_trials = oc1.slider(
        "Nombre d'essais", 20, 300, 60, 10,
        help="Plus d'essais = recherche plus large mais plus longue. Chaque "
             "lancement recalcule aussi tout le tableau de bord — compte "
             "environ 1 à 3 minutes pour 60 essais sur les 100 actions, "
             "proportionnellement plus pour un nombre d'essais plus élevé.",
    )
    opt_metric = oc2.selectbox(
        "Optimiser pour", list(METRIC_LABELS.keys()),
        format_func=lambda k: METRIC_LABELS[k],
        help="Performance totale : maximise le rendement brut. Ratio "
             "performance/drawdown : privilégie un meilleur compromis "
             "risque/rendement, quitte à avoir un rendement brut plus faible. "
             "Win rate : maximise le taux de trades gagnants (attention, "
             "souvent au prix d'une performance globale plus faible, voir nos "
             "tests précédents).",
    )

    if st.button("🚀 Lancer l'optimisation"):
        opt_bar = st.progress(0.0, text="Initialisation...")
        try:
            test_results, train_results, split_date = optimize(
                per_ticker, n_trials=opt_n_trials, top_k=10, metric=opt_metric,
                progress_callback=lambda frac, msg: opt_bar.progress(frac, text=msg),
            )
            st.session_state["optimizer_results"] = test_results
            st.session_state["optimizer_split_date"] = split_date
            st.session_state["optimizer_metric"] = opt_metric
        except RuntimeError as exc:
            st.error(str(exc))
        opt_bar.empty()

    if "optimizer_results" in st.session_state:
        opt_results = st.session_state["optimizer_results"]
        opt_split = st.session_state["optimizer_split_date"]
        opt_metric_used = st.session_state["optimizer_metric"]
        st.success(
            f"{len(opt_results)} configurations validées — entraînement avant le "
            f"{opt_split.date()}, test après (jamais vu pendant la recherche)."
        )

        display = opt_results.rename(columns={
            "strat_perf": "Perf. test (%)",
            f"train_{opt_metric_used}": "Score train",
            "bh_perf": "Buy & hold (%)",
            "win_rate": "Win rate (%)",
            "max_dd": "Max drawdown (%)",
            "n_trades": "Nb trades",
            "confirm_bars": "Confirm. bougies",
        })
        show_cols = [c for c in ["Perf. test (%)", "Score train", "Buy & hold (%)", "Win rate (%)",
                                   "Max drawdown (%)", "Nb trades", "Confirm. bougies"] if c in display.columns]
        st.dataframe(
            display[show_cols].style.format({
                "Perf. test (%)": "{:+.1f}", "Score train": "{:+.2f}", "Buy & hold (%)": "{:+.1f}",
                "Win rate (%)": "{:.1f}", "Max drawdown (%)": "{:.1f}",
            }),
            width="stretch", height=300,
        )

        st.markdown("**Appliquer une configuration**")
        opt_options = [
            f"#{i+1} — confirm_bars={row['confirm_bars']} — perf test {row['strat_perf']:+.1f}% "
            f"(win rate {row['win_rate']:.0f}%)"
            for i, row in opt_results.iterrows()
        ]
        opt_choice = st.selectbox("Choisis un candidat", range(len(opt_options)), format_func=lambda i: opt_options[i])

        oc3, oc4 = st.columns(2)
        if oc3.button("✅ Appliquer ces réglages", width="stretch"):
            st.session_state["_pending_apply"] = strategy_params_to_ui_dict(
                params_from_row(opt_results.iloc[opt_choice])
            )
            st.rerun()
        if oc4.button("💾 Appliquer et sauvegarder comme profil", width="stretch"):
            ui_dict = strategy_params_to_ui_dict(params_from_row(opt_results.iloc[opt_choice]))
            st.session_state["_pending_apply"] = ui_dict
            profile_name = f"Optimisé {pd.Timestamp.now():%Y-%m-%d %H:%M}"
            st.session_state.profiles[profile_name] = ui_dict
            save_profiles(st.session_state.profiles)
            st.rerun()

# ---------------------------------------------------------------------------
# Section : Simulation de portefeuille réaliste (capital unique en EUR,
# frais IBKR, conversion de change, 20 positions max à 2% de l'équity)
# ---------------------------------------------------------------------------
st.subheader("💶 Simulation de portefeuille réaliste (EUR, frais IBKR)")
with st.expander(
    "Simuler l'évolution d'un capital réel en euros (20 positions max, 2% chacune, frais IBKR)",
    expanded=False,
):
    st.caption(
        "Contrairement au tableau de bord ci-dessus (qui teste chaque action isolément avec un "
        "capital plein, puis moyenne les résultats — pratique pour comparer les actions entre "
        "elles, mais irréaliste pour juger une performance en argent réel), cette section simule "
        "**un seul portefeuille partagé** : 20 positions maximum en simultané, chacune dimensionnée "
        "à 2% de l'équity du moment, avec les frais Interactive Brokers (commissions actions US + "
        "conversion de change EUR/USD) et les contraintes de trésorerie/capacité d'un vrai compte."
    )

    sc1, sc2, sc3 = st.columns(3)
    sim_capital = sc1.number_input("Capital initial (€)", 1_000, 10_000_000, 100_000, step=5_000)
    sim_max_positions = sc2.slider("Positions simultanées max", 1, 40, 20)
    sim_position_pct = sc3.slider("Taille par position (% équity)", 0.5, 10.0, 2.0, step=0.5)

    with st.expander("⚙️ Avancé : frais (par défaut = tarification IBKR Pro fixe, actions US)"):
        fc1, fc2, fc3 = st.columns(3)
        sim_comm_per_share = fc1.number_input("Commission par action ($)", 0.0, 0.05, 0.005, step=0.001, format="%.3f")
        sim_comm_min = fc2.number_input("Commission min. par ordre ($)", 0.0, 10.0, 1.0, step=0.5)
        sim_comm_max_pct = fc3.number_input("Plafond commission (% valeur ordre)", 0.1, 5.0, 1.0, step=0.1)
        fc4, fc5 = st.columns(2)
        sim_fx_fee_pct = fc4.number_input("Frais de change (% converti)", 0.0, 1.0, 0.03, step=0.01)
        sim_fx_fee_min = fc5.number_input("Frais de change min. ($)", 0.0, 20.0, 2.0, step=0.5)
        sim_fallback_rate = st.number_input("Taux EUR/USD de repli (si historique indisponible)", 0.8, 1.5, 1.08, step=0.01)

    if st.button("💶 Lancer la simulation réaliste"):
        sim_bar = st.progress(0.0, text="Initialisation...")
        eur_usd_rate, used_live_fx = get_eurusd_series_safe(
            period="5y", fallback_rate=sim_fallback_rate,
        )
        sim_params = PortfolioSimParams(
            initial_capital_eur=float(sim_capital),
            max_positions=sim_max_positions,
            position_pct=sim_position_pct / 100,
            commission_per_share_usd=sim_comm_per_share,
            commission_min_usd=sim_comm_min,
            commission_max_pct_of_trade=sim_comm_max_pct / 100,
            fx_fee_pct=sim_fx_fee_pct / 100,
            fx_fee_min_usd=sim_fx_fee_min,
            eur_usd_fallback_rate=sim_fallback_rate,
        )
        try:
            sim_result = run_portfolio_simulation(
                per_ticker, StrategyParams(**params_dict), sim_params,
                eur_usd_rate=eur_usd_rate, used_live_fx=used_live_fx,
                progress_callback=lambda frac, msg: sim_bar.progress(min(frac, 1.0), text=msg),
            )
            st.session_state["sim_result"] = sim_result
        except RuntimeError as exc:
            st.error(str(exc))
        sim_bar.empty()

    if "sim_result" in st.session_state:
        sim_result = st.session_state["sim_result"]
        if not sim_result.used_live_fx:
            st.warning(
                "⚠️ L'historique EUR/USD réel n'a pas pu être téléchargé — taux constant utilisé "
                "à la place. Moins réaliste : le taux de change bouge de plusieurs % sur la période."
            )

        m1, m2, m3, m4 = st.columns(4)
        metric_card(m1, "Capital final", f"{sim_result.final_equity_eur:,.0f} €", tone_for(sim_result.total_return_pct))
        metric_card(m2, "Performance totale", f"{sim_result.total_return_pct:+.1f}%", tone_for(sim_result.total_return_pct))
        metric_card(m3, "Max drawdown", f"{sim_result.max_drawdown_pct:.1f}%", "negative")
        metric_card(m4, "Win rate", f"{sim_result.win_rate_pct:.0f}%")

        m5, m6, m7 = st.columns(3)
        metric_card(m5, "Trades exécutés", f"{len(sim_result.closed_trades)}")
        metric_card(m6, "Frais totaux payés", f"{sim_result.total_fees_eur:,.0f} €", "negative")
        metric_card(
            m7, "Signaux ignorés",
            f"{sim_result.n_skipped_capacity + sim_result.n_skipped_cash + sim_result.n_skipped_lot}",
        )
        st.caption(
            f"Détail des signaux ignorés — capacité (20 positions déjà ouvertes) : "
            f"{sim_result.n_skipped_capacity} · trésorerie insuffisante : {sim_result.n_skipped_cash} · "
            f"position < 1 action : {sim_result.n_skipped_lot}"
        )

        fig_sim, ax_sim = plt.subplots(figsize=(11, 3.5))
        sim_result.equity_curve.plot(ax=ax_sim, linewidth=2, color="#2563eb")
        ax_sim.set_ylabel("Équity (€)")
        ax_sim.set_title("Portefeuille réaliste — capital unique en euros, net de frais", fontsize=10)
        ax_sim.grid(alpha=0.25)
        fig_sim.tight_layout()
        st.pyplot(fig_sim)
        plt.close(fig_sim)

        if sim_result.closed_trades:
            sim_trades_df = pd.DataFrame([t.__dict__ for t in sim_result.closed_trades])
            st.dataframe(sim_trades_df, width="stretch", height=250)
            st.download_button(
                "⬇️ Télécharger les trades (CSV)",
                data=sim_trades_df.to_csv(index=False).encode("utf-8"),
                file_name="trades_simulation_realiste.csv", mime="text/csv",
            )

# ---------------------------------------------------------------------------
# Section 2 : détail par action — un graphique entrée/sortie par action,
# affiché directement (empilés les uns au-dessus des autres, sans clic)
# ---------------------------------------------------------------------------
st.subheader("Détail par action — points d'entrée et de sortie")
search = st.text_input("🔍 Filtrer par ticker", "")
tickers_to_show = [t for t in tickers_sorted if search.upper() in t]
st.caption(f"{len(tickers_to_show)} action(s) affichée(s)")

for ticker in tickers_to_show:
    if ticker not in per_ticker_results:
        continue
    r = per_ticker_results[ticker]
    result = r["result"]
    bh_return = (r["buy_hold"].iloc[-1] / r["buy_hold"].iloc[0] - 1) * 100

    st.markdown(
        f"**{ticker}** — {result.total_return_pct:+.1f}% · "
        f"{result.num_trades} trade(s) · win rate {result.win_rate_pct:.0f}% · "
        f"max drawdown {result.max_drawdown_pct:.1f}%"
    )

    fig2_bytes = price_chart_png(ticker, r["df"], r["upper"], r["lower"], result)
    st.image(fig2_bytes, width="stretch")

    with st.expander(f"Détails {ticker} (métriques, courbe d'équity, trades)"):
        c1, c2, c3, c4 = st.columns(4)
        metric_card(c1, "Performance stratégie", f"{result.total_return_pct:+.1f}%", tone_for(result.total_return_pct))
        metric_card(c2, "Buy & hold (référence)", f"{bh_return:+.1f}%", tone_for(bh_return))
        metric_card(c3, "Win rate", f"{result.win_rate_pct:.0f}%")
        metric_card(c4, "Nb. de trades", f"{result.num_trades}")

        c5, c6, c7 = st.columns(3)
        metric_card(c5, "Max drawdown", f"{result.max_drawdown_pct:.1f}%", "negative")
        metric_card(c6, "Trades longs", f"{r['long_trades']}")
        metric_card(c7, "Trades shorts", f"{r['short_trades']}")

        fig1_bytes = equity_chart_png(ticker, result, r["buy_hold"], f"{ticker} — Stratégie vs Buy & Hold")
        st.image(fig1_bytes, width="stretch")

        if result.trades:
            trades_df = pd.DataFrame([t.__dict__ for t in result.trades])
            st.dataframe(trades_df, width="stretch", height=180)
        else:
            st.write("Aucun trade déclenché sur cette action avec ces paramètres.")

    st.divider()

st.caption(
    "⚠️ Backtest historique, à titre pédagogique. Les performances passées ne "
    "préjugent pas des performances futures."
)
