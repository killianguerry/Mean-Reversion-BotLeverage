# Mean Reversion Backtester — Nasdaq-100

Bot de backtesting d'une stratégie de **mean reversion** (Bollinger Bands +
RSI hebdomadaire multi-timeframe + ADX + stop ATR), testée automatiquement
sur les **100 actions du Nasdaq-100**.

**Aucune donnée à fournir manuellement.** Au premier lancement, le bot
récupère automatiquement :
- la liste actuelle des 100 tickers depuis Wikipedia
- 5 ans d'historique de prix pour chacun, via Yahoo Finance (`yfinance`)

et les met en cache localement (`data/nasdaq100_universe.parquet`). Le cache
est réutilisé tant qu'il a moins de 24h ; passé ce délai (ou avec
`--refresh-data` / le bouton de rafraîchissement dans l'app), les données
sont retéléchargées automatiquement.

## La stratégie

**Entrée LONG** (toutes les conditions doivent être vraies) :
- RSI hebdomadaire (semaine précédente clôturée) **> 55**
- ADX daily **> 20** ET ADX hebdomadaire **> 20**
- Clôture daily **sous la bande de Bollinger basse**

→ Entrée à l'ouverture du jour suivant.

**Sortie LONG :**
- Stop-loss : plus bas de la bougie signal − *multiplicateur* × ATR(14)
- Take profit : clôture **au-dessus** de la bande de Bollinger haute

Le **short** est symétrique mais désactivé par défaut.

> ⚠️ Outil pédagogique. Les performances passées ne préjugent pas des
> performances futures. Rien ici ne constitue un conseil en investissement.

## Installation

```bash
git clone https://github.com/TON_PSEUDO/mean-reversion-backtester.git
cd mean-reversion-backtester
python3 -m venv venv
source venv/bin/activate  # sous Windows : venv\Scripts\activate
pip install -r requirements.txt
```

## Utilisation

### Interface web (recommandé)

```bash
streamlit run streamlit_app.py
```

Au premier lancement, un message indique que les données sont en cours de
téléchargement (quelques minutes pour les 100 actions). Les lancements
suivants sont quasi instantanés grâce au cache.

### Ligne de commande

```bash
# Backtest sur les 100 actions du Nasdaq-100 (par défaut)
python main.py

# Forcer un nouveau téléchargement des données
python main.py --refresh-data

# Backtest sur un seul ticker (ex: l'ETF QQQ)
python main.py --ticker QQQ --period 3y

# Personnaliser les paramètres de la stratégie
python main.py --bb-period 25 --atr-mult 3 --shorts
```

Toutes les options : `python main.py --help`

### Résultats

Chaque run génère dans `output/` :
- `portfolio_equity_curve.png` — courbe d'équity du portefeuille équipondéré
- `summary_par_ticker.csv` — performance, trades, win rate et max drawdown pour chacune des 100 actions

## Structure du projet

```
mean_reversion_bot_leverage/
├── main.py                # point d'entrée CLI
├── streamlit_app.py       # interface web (auto-fetch, menu par action, optimiseur)
├── backtester.py          # logique de la stratégie + moteur de simulation
├── indicators.py          # Bollinger Bands, RSI, ATR, ADX, resample hebdo
├── data_loader.py         # chargement CSV / ticker unique (mode avancé)
├── nasdaq100_data.py      # récupération + cache auto (Wikipedia + yfinance)
├── portfolio.py           # agrégation des backtests sur l'univers complet
├── optimizer.py           # optimiseur de réglages (recherche + validation train/test)
├── requirements.txt
├── data/                  # cache généré automatiquement (non versionné)
└── output/                # résultats générés (non versionné)
```

## Optimiseur de réglages

Plutôt que de régler les curseurs à la main, l'optimiseur teste automatiquement
un grand nombre de combinaisons au hasard et cherche la meilleure performance :

- **Recherche** sur les 70% premiers jours de données (entraînement)
- **Validation** des meilleurs candidats sur les 30% restants, jamais vus
  pendant la recherche — évite le sur-ajustement. C'est cette performance de
  test, pas celle d'entraînement, qui doit orienter le choix.

**Interface web** : section "🔍 Optimiseur de réglages", avec un bouton pour
appliquer directement la configuration choisie (et la sauvegarder comme profil
si besoin).

**CLI** :
```bash
python main.py --optimize --n-trials 60
python main.py --optimize --n-trials 100 --opt-metric perf_risk_ratio
```
Résultats sauvegardés dans `output/optimizer_results.csv`, et la meilleure
configuration est affichée avec la commande prête à copier pour la relancer.

## Dépannage : la récupération des données échoue

Cause la plus fréquente : une version de `yfinance` trop ancienne (Yahoo
change régulièrement son API). Le bot fait un test de connexion avant de
lancer les 100 téléchargements et affiche un message clair en cas
d'échec. Dans l'ordre à essayer :

1. `pip install --upgrade yfinance`
2. Vérifier la connexion internet
3. Désactiver un VPN / changer de réseau si Yahoo Finance semble bloqué
4. Attendre quelques minutes (Yahoo limite parfois temporairement les requêtes)

## Prochaines étapes possibles

- Optimisation des paramètres (grid search / walk-forward)
- Connexion à un broker (Alpaca, Interactive Brokers) pour le live trading
- Filtres additionnels (volume, secteur, corrélation)
