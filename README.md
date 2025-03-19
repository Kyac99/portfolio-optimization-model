# Modèle d'Allocation d'Actifs Optimisé

Ce projet développe un modèle complet d'allocation d'actifs optimisé utilisant différentes méthodes d'optimisation de portefeuille, avec un focus sur les actions et les obligations.

## Objectifs du Projet

- Implémenter et comparer différentes méthodes d'optimisation de portefeuille:
  - Optimisation de Markowitz (Mean-Variance Optimization)
  - Modèle Black-Litterman
  - Approches Factor Investing
- Intégrer des contraintes spécifiques:
  - Liquidité
  - Drawdown maximal
  - Contraintes réglementaires
- Backtester les différentes stratégies
- Analyser la robustesse des modèles

## Structure du Projet

```
portfolio-optimization-model/
│
├── data/                      # Données brutes et traitées
│   ├── raw/                   # Données brutes téléchargées
│   └── processed/             # Données nettoyées et prêtes à l'emploi
│
├── src/                       # Code source
│   ├── data/                  # Scripts pour la collecte et préparation des données
│   │   ├── data_collection.py # Téléchargement des données de marché
│   │   └── data_processing.py # Nettoyage et préparation des données
│   │
│   ├── models/                # Implémentation des modèles d'optimisation
│   │   ├── markowitz.py       # Optimisation de Markowitz
│   │   ├── black_litterman.py # Modèle Black-Litterman
│   │   └── factor_investing.py# Approches Factor Investing
│   │
│   ├── optimization/          # Scripts d'optimisation avec contraintes
│   │   ├── constraints.py     # Implémentation des contraintes
│   │   └── optimizer.py       # Algorithmes d'optimisation
│   │
│   └── evaluation/            # Scripts d'évaluation et de backtest
│       ├── backtest.py        # Backtesting des stratégies
│       └── performance.py     # Calcul des métriques de performance
│
├── notebooks/                 # Jupyter notebooks pour analyses et visualisations
│   ├── 01_data_exploration.ipynb
│   ├── 02_markowitz_optimization.ipynb
│   ├── 03_black_litterman.ipynb
│   ├── 04_factor_investing.ipynb
│   └── 05_comparison.ipynb
│
├── results/                   # Résultats des tests et optimisations
│   ├── figures/               # Graphiques et visualisations
│   └── models/                # Modèles entraînés et leurs paramètres
│
├── requirements.txt           # Dépendances du projet
└── README.md                  # Documentation principale
```

## Installation et Utilisation

1. Cloner ce dépôt:
```bash
git clone https://github.com/Kyac99/portfolio-optimization-model.git
cd portfolio-optimization-model
```

2. Installer les dépendances:
```bash
pip install -r requirements.txt
```

3. Exécuter les notebooks dans l'ordre pour:
   - Explorer les données
   - Appliquer les différentes méthodes d'optimisation
   - Comparer les résultats

## Principales Bibliothèques Utilisées

- pandas & numpy: Manipulation des données
- scipy & cvxpy: Optimisation du portefeuille
- matplotlib & seaborn: Visualisation des résultats
- yfinance: Collecte des données de marché
- scikit-learn: Pour les approches factor investing
- PyPortfolioOpt: Implémentations des algorithmes d'optimisation de portefeuille

## Méthodologie

1. **Collecte des données**: Utilisation de yfinance pour obtenir les données historiques d'actions et d'obligations
2. **Optimisation de Markowitz**: Implémentation de l'optimisation moyenne-variance
3. **Modèle Black-Litterman**: Intégration d'opinions subjectives dans l'optimisation
4. **Factor Investing**: Construction de portefeuilles basés sur des facteurs de risque
5. **Contraintes**: Application de contraintes sur liquidité, drawdown et concentration
6. **Backtest**: Test des stratégies sur données historiques
7. **Analyse de robustesse**: Tests de sensibilité et simulation Monte Carlo

## Résultats et Performances

Les résultats des différentes optimisations et leurs performances en backtest sont disponibles dans le dossier `results/` et présentés dans les notebooks.