#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module pour l'optimisation de portefeuille basée sur les facteurs.
Ce module implémente des stratégies d'investissement factoriel (Factor Investing)
pour construire des portefeuilles optimisés.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco
import logging
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FactorInvestingOptimizer:
    """
    Classe implémentant l'optimisation de portefeuille basée sur les facteurs.
    """
    
    def __init__(self, returns_data, factor_data=None, risk_free_rate=0.0):
        """
        Initialise l'optimiseur basé sur les facteurs.
        
        Args:
            returns_data (pandas.DataFrame): Rendements historiques des actifs
            factor_data (pandas.DataFrame, optional): Données de facteurs pré-définis
            risk_free_rate (float): Taux sans risque (annualisé)
        """
        self.returns_data = returns_data
        self.factor_data = factor_data
        self.risk_free_rate = risk_free_rate
        self.assets = list(returns_data.columns)
        self.n_assets = len(self.assets)
        
        # Si aucune donnée de facteur n'est fournie, les créer à partir des rendements
        if factor_data is None:
            logger.info("Aucune donnée de facteur fournie, extraction des facteurs via PCA")
            self.extract_factors()
        
        logger.info(f"Initialisation de l'optimiseur factoriel avec {self.n_assets} actifs")
    
    def extract_factors(self, n_factors=5):
        """
        Extrait les facteurs à partir des rendements en utilisant PCA.
        
        Args:
            n_factors (int): Nombre de facteurs à extraire
            
        Note:
            Cette méthode modifie l'instance en ajoutant les attributs:
            - factors: Rendements des facteurs extraits
            - factor_loadings: Coefficients de sensibilité des actifs aux facteurs
        """
        logger.info(f"Extraction de {n_factors} facteurs via PCA")
        
        # Extraction des facteurs via PCA
        pca = PCA(n_components=n_factors)
        factor_returns = pca.fit_transform(self.returns_data)
        
        # Convertir en DataFrame
        self.factors = pd.DataFrame(
            factor_returns, 
            index=self.returns_data.index,
            columns=[f'Factor_{i+1}' for i in range(n_factors)]
        )
        
        # Calculer les loadings des facteurs
        self.factor_loadings = pd.DataFrame(
            pca.components_.T,
            index=self.assets,
            columns=self.factors.columns
        )
        
        # Calculer le pourcentage de variance expliquée
        self.explained_variance = pd.Series(
            pca.explained_variance_ratio_,
            index=self.factors.columns
        )
        
        logger.info(f"Facteurs extraits avec succès. Variance expliquée: {self.explained_variance.sum():.2%}")
        
        return self.factors, self.factor_loadings
    
    def compute_factor_exposures(self, method='regression'):
        """
        Calcule les expositions (betas) des actifs aux facteurs.
        
        Args:
            method (str): Méthode de calcul ('regression' ou 'covariance')
            
        Returns:
            pandas.DataFrame: Matrice d'expositions aux facteurs
        """
        logger.info(f"Calcul des expositions aux facteurs via {method}")
        
        if not hasattr(self, 'factors'):
            logger.error("Aucun facteur disponible. Exécutez extract_factors() d'abord.")
            return None
        
        if method == 'regression':
            # Calcul des expositions par régression linéaire
            exposures = pd.DataFrame(index=self.assets, columns=self.factors.columns)
            
            for asset in self.assets:
                model = LinearRegression()
                model.fit(self.factors, self.returns_data[asset])
                exposures.loc[asset] = model.coef_
            
            self.factor_exposures = exposures
        
        elif method == 'covariance':
            # Calcul des expositions par covariance/variance
            cov_matrix = pd.concat([self.returns_data, self.factors], axis=1).cov()
            factor_var = np.diag(cov_matrix.loc[self.factors.columns, self.factors.columns])
            
            exposures = pd.DataFrame(index=self.assets, columns=self.factors.columns)
            
            for asset in self.assets:
                asset_factor_cov = cov_matrix.loc[asset, self.factors.columns]
                exposures.loc[asset] = asset_factor_cov / factor_var
            
            self.factor_exposures = exposures
        
        else:
            logger.error(f"Méthode {method} non reconnue. Utilisez 'regression' ou 'covariance'.")
            return None
        
        logger.info("Expositions aux facteurs calculées avec succès")
        
        return self.factor_exposures
    
    def compute_factor_expected_returns(self, risk_premium=0.05, use_historical=True, n_periods=252):
        """
        Calcule les rendements attendus des facteurs.
        
        Args:
            risk_premium (float): Prime de risque pour l'ensemble des facteurs
            use_historical (bool): Utiliser les rendements historiques comme base
            n_periods (int): Nombre de périodes pour les rendements (252 jours = 1 an)
            
        Returns:
            pandas.Series: Rendements attendus des facteurs
        """
        logger.info("Calcul des rendements attendus des facteurs")
        
        if not hasattr(self, 'factors'):
            logger.error("Aucun facteur disponible. Exécutez extract_factors() d'abord.")
            return None
        
        if use_historical:
            # Utiliser les rendements historiques des facteurs
            factor_returns = self.factors.iloc[-n_periods:].mean() * n_periods
        else:
            # Distribuer la prime de risque selon l'importance des facteurs
            factor_returns = self.explained_variance * risk_premium
        
        self.factor_expected_returns = factor_returns
        
        logger.info("Rendements attendus des facteurs calculés avec succès")
        
        return self.factor_expected_returns
    
    def compute_asset_expected_returns(self):
        """
        Calcule les rendements attendus des actifs basés sur le modèle factoriel.
        
        Returns:
            pandas.Series: Rendements attendus des actifs
        """
        logger.info("Calcul des rendements attendus des actifs basés sur les facteurs")
        
        if not hasattr(self, 'factor_exposures') or not hasattr(self, 'factor_expected_returns'):
            logger.error("Expositions aux facteurs ou rendements attendus des facteurs non disponibles.")
            return None
        
        # E[R] = β * E[F] + α
        # Pour simplifier, on suppose α = 0 (pas de rendement anormal)
        expected_returns = pd.Series(np.dot(self.factor_exposures, self.factor_expected_returns), 
                                   index=self.assets)
        
        self.expected_returns = expected_returns
        
        logger.info("Rendements attendus des actifs calculés avec succès")
        
        return self.expected_returns
    
    def compute_factor_covariance(self, n_periods=252):
        """
        Calcule la matrice de covariance des facteurs.
        
        Args:
            n_periods (int): Nombre de périodes pour le calcul (252 jours = 1 an)
            
        Returns:
            pandas.DataFrame: Matrice de covariance des facteurs
        """
        logger.info("Calcul de la matrice de covariance des facteurs")
        
        if not hasattr(self, 'factors'):
            logger.error("Aucun facteur disponible. Exécutez extract_factors() d'abord.")
            return None
        
        # Utiliser les dernières n_periods pour le calcul de la covariance
        factor_cov = self.factors.iloc[-n_periods:].cov() * n_periods
        
        self.factor_covariance = factor_cov
        
        logger.info("Matrice de covariance des facteurs calculée avec succès")
        
        return self.factor_covariance
    
    def compute_asset_covariance(self):
        """
        Calcule la matrice de covariance des actifs basée sur le modèle factoriel.
        
        Returns:
            pandas.DataFrame: Matrice de covariance des actifs
        """
        logger.info("Calcul de la matrice de covariance des actifs basée sur les facteurs")
        
        if not hasattr(self, 'factor_exposures') or not hasattr(self, 'factor_covariance'):
            logger.error("Expositions aux facteurs ou covariance des facteurs non disponibles.")
            return None
        
        # Σ = B * F * B' + D
        # où B est la matrice des expositions aux facteurs, F est la covariance des facteurs
        # et D est la matrice diagonale des variances résiduelles
        
        # Calculer la partie factorielle de la covariance
        factor_part = np.dot(
            np.dot(self.factor_exposures, self.factor_covariance),
            self.factor_exposures.T
        )
        
        # Calculer les variances résiduelles (approximatif)
        # Pour une version plus précise, il faudrait utiliser les résidus des régressions
        residual_var = np.diag(self.returns_data.var() - np.diag(factor_part))
        
        # Combiner les deux parties
        asset_cov = factor_part + residual_var
        
        self.covariance_matrix = pd.DataFrame(asset_cov, index=self.assets, columns=self.assets)
        
        logger.info("Matrice de covariance des actifs calculée avec succès")
        
        return self.covariance_matrix
    
    def score_assets_by_factor(self, factor_name, ascending=False):
        """
        Note les actifs selon leur exposition à un facteur spécifique.
        
        Args:
            factor_name (str): Nom du facteur
            ascending (bool): Tri croissant (True) ou décroissant (False)
            
        Returns:
            pandas.Series: Scores des actifs pour le facteur
        """
        logger.info(f"Notation des actifs par exposition au facteur {factor_name}")
        
        if not hasattr(self, 'factor_exposures'):
            logger.error("Expositions aux facteurs non disponibles.")
            return None
        
        if factor_name not in self.factor_exposures.columns:
            logger.error(f"Facteur {factor_name} non trouvé.")
            return None
        
        # Récupérer les expositions au facteur et les trier
        factor_scores = self.factor_exposures[factor_name].sort_values(ascending=ascending)
        
        return factor_scores
    
    def score_assets_multi_factor(self, factor_weights=None):
        """
        Note les actifs selon une combinaison pondérée de facteurs.
        
        Args:
            factor_weights (dict): Poids des facteurs {factor_name: weight}
            
        Returns:
            pandas.Series: Scores des actifs pour la combinaison de facteurs
        """
        logger.info("Notation des actifs par exposition multi-factorielle")
        
        if not hasattr(self, 'factor_exposures'):
            logger.error("Expositions aux facteurs non disponibles.")
            return None
        
        # Si aucun poids n'est fourni, utiliser des poids égaux
        if factor_weights is None:
            factor_weights = {factor: 1.0 for factor in self.factor_exposures.columns}
        
        # Vérifier que tous les facteurs existent
        for factor in factor_weights:
            if factor not in self.factor_exposures.columns:
                logger.error(f"Facteur {factor} non trouvé.")
                return None
        
        # Calculer les scores pondérés
        weighted_scores = pd.Series(0, index=self.assets)
        for factor, weight in factor_weights.items():
            weighted_scores += self.factor_exposures[factor] * weight
        
        # Trier les scores
        weighted_scores = weighted_scores.sort_values(ascending=False)
        
        return weighted_scores
    
    def create_portfolio_by_factor(self, factor_name, n_assets=10, ascending=False):
        """
        Crée un portefeuille en sélectionnant les n meilleurs actifs selon un facteur.
        
        Args:
            factor_name (str): Nom du facteur
            n_assets (int): Nombre d'actifs à inclure dans le portefeuille
            ascending (bool): Tri croissant (True) ou décroissant (False)
            
        Returns:
            pandas.Series: Poids des actifs dans le portefeuille
        """
        logger.info(f"Création d'un portefeuille basé sur le facteur {factor_name}")
        
        # Noter les actifs selon le facteur
        factor_scores = self.score_assets_by_factor(factor_name, ascending)
        
        if factor_scores is None:
            return None
        
        # Sélectionner les n meilleurs actifs
        selected_assets = factor_scores.index[:n_assets]
        
        # Créer un portefeuille équipondéré avec les actifs sélectionnés
        weights = pd.Series(0, index=self.assets)
        weights[selected_assets] = 1.0 / n_assets
        
        return weights
    
    def create_portfolio_multi_factor(self, factor_weights=None, n_assets=10):
        """
        Crée un portefeuille en sélectionnant les n meilleurs actifs selon une combinaison de facteurs.
        
        Args:
            factor_weights (dict): Poids des facteurs {factor_name: weight}
            n_assets (int): Nombre d'actifs à inclure dans le portefeuille
            
        Returns:
            pandas.Series: Poids des actifs dans le portefeuille
        """
        logger.info("Création d'un portefeuille basé sur une combinaison de facteurs")
        
        # Noter les actifs selon la combinaison de facteurs
        multi_scores = self.score_assets_multi_factor(factor_weights)
        
        if multi_scores is None:
            return None
        
        # Sélectionner les n meilleurs actifs
        selected_assets = multi_scores.index[:n_assets]
        
        # Créer un portefeuille équipondéré avec les actifs sélectionnés
        weights = pd.Series(0, index=self.assets)
        weights[selected_assets] = 1.0 / n_assets
        
        return weights
    
    def optimize_factor_portfolio(self, target_exposures=None):
        """
        Optimise un portefeuille pour cibler des expositions factorielles spécifiques.
        
        Args:
            target_exposures (dict): Expositions cibles aux facteurs {factor_name: target_exposure}
            
        Returns:
            pandas.Series: Poids optimisés des actifs dans le portefeuille
        """
        logger.info("Optimisation d'un portefeuille avec des expositions factorielles cibles")
        
        if not hasattr(self, 'factor_exposures'):
            logger.error("Expositions aux facteurs non disponibles.")
            return None
        
        # Préparer les contraintes
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Somme des poids = 1
        
        if target_exposures is not None:
            for factor, target in target_exposures.items():
                if factor in self.factor_exposures.columns:
                    constraints.append({
                        'type': 'eq',
                        'fun': lambda x, f=factor, t=target: np.dot(x, self.factor_exposures[f]) - t
                    })
        
        # Bornes: poids entre 0 et 1 (pas de vente à découvert)
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Fonction objectif: minimiser la variance
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(self.covariance_matrix, weights))
        
        # Point de départ: poids égaux
        initial_weights = np.array([1.0 / self.n_assets] * self.n_assets)
        
        # Optimisation
        result = sco.minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Vérifier la convergence
        if not result['success']:
            logger.warning(f"L'optimisation n'a pas convergé: {result['message']}")
        
        # Extraire les poids optimaux
        optimal_weights = pd.Series(result['x'], index=self.assets)
        
        return optimal_weights
    
    def _portfolio_return(self, weights):
        """
        Calcule le rendement attendu d'un portefeuille.
        
        Args:
            weights (numpy.ndarray): Poids des actifs dans le portefeuille
            
        Returns:
            float: Rendement attendu du portefeuille
        """
        if not hasattr(self, 'expected_returns'):
            self.compute_asset_expected_returns()
        
        return np.sum(self.expected_returns * weights)
    
    def _portfolio_volatility(self, weights):
        """
        Calcule la volatilité d'un portefeuille.
        
        Args:
            weights (numpy.ndarray): Poids des actifs dans le portefeuille
            
        Returns:
            float: Volatilité du portefeuille
        """
        if not hasattr(self, 'covariance_matrix'):
            self.compute_asset_covariance()
        
        return np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
    
    def _portfolio_sharpe_ratio(self, weights):
        """
        Calcule le ratio de Sharpe d'un portefeuille.
        
        Args:
            weights (numpy.ndarray): Poids des actifs dans le portefeuille
            
        Returns:
            float: Ratio de Sharpe du portefeuille
        """
        return (self._portfolio_return(weights) - self.risk_free_rate) / self._portfolio_volatility(weights)
    
    def _negative_sharpe_ratio(self, weights):
        """
        Calcule l'opposé du ratio de Sharpe (pour minimisation).
        
        Args:
            weights (numpy.ndarray): Poids des actifs dans le portefeuille
            
        Returns:
            float: Opposé du ratio de Sharpe
        """
        return -self._portfolio_sharpe_ratio(weights)
    
    def evaluate_portfolio(self, weights):
        """
        Évalue les caractéristiques d'un portefeuille.
        
        Args:
            weights (numpy.ndarray or pandas.Series): Poids des actifs dans le portefeuille
            
        Returns:
            dict: Caractéristiques du portefeuille
        """
        logger.info("Évaluation des caractéristiques du portefeuille")
        
        if isinstance(weights, pd.Series):
            weights = weights.values
        
        # Calculer les métriques du portefeuille
        portfolio = {
            'weights': pd.Series(weights, index=self.assets),
            'return': self._portfolio_return(weights),
            'volatility': self._portfolio_volatility(weights),
            'sharpe_ratio': self._portfolio_sharpe_ratio(weights)
        }
        
        # Calculer les expositions factorielles du portefeuille
        if hasattr(self, 'factor_exposures'):
            factor_exposures = {}
            for factor in self.factor_exposures.columns:
                factor_exposures[factor] = np.dot(weights, self.factor_exposures[factor])
            
            portfolio['factor_exposures'] = pd.Series(factor_exposures)
        
        return portfolio
    
    def plot_factor_loadings(self, n_top=10, figsize=(12, 8)):
        """
        Trace les loadings des facteurs pour les n actifs les plus importants.
        
        Args:
            n_top (int): Nombre d'actifs à afficher
            figsize (tuple): Dimensions de la figure
            
        Returns:
            matplotlib.figure.Figure: Figure matplotlib
        """
        if not hasattr(self, 'factor_loadings'):
            logger.error("Les loadings des facteurs ne sont pas disponibles.")
            return None
        
        n_factors = len(self.factor_loadings.columns)
        
        # Sélectionner les n actifs avec les plus grands loadings moyens
        mean_loadings = self.factor_loadings.abs().mean(axis=1).sort_values(ascending=False)
        top_assets = mean_loadings.index[:n_top]
        
        # Créer la figure
        fig, axes = plt.subplots(n_factors, 1, figsize=figsize, sharex=True)
        
        for i, factor in enumerate(self.factor_loadings.columns):
            # Trier les loadings pour ce facteur
            sorted_loadings = self.factor_loadings.loc[top_assets, factor].sort_values()
            
            # Tracer le graphique
            ax = axes[i] if n_factors > 1 else axes
            ax.barh(range(len(sorted_loadings)), sorted_loadings, align='center')
            ax.set_yticks(range(len(sorted_loadings)))
            ax.set_yticklabels(sorted_loadings.index)
            ax.set_title(f'Loadings pour {factor}')
            ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def plot_factor_returns(self, figsize=(12, 8)):
        """
        Trace les rendements cumulés des facteurs.
        
        Args:
            figsize (tuple): Dimensions de la figure
            
        Returns:
            matplotlib.figure.Figure: Figure matplotlib
        """
        if not hasattr(self, 'factors'):
            logger.error("Les rendements des facteurs ne sont pas disponibles.")
            return None
        
        # Calculer les rendements cumulés
        cumulative_returns = (1 + self.factors).cumprod()
        
        # Créer la figure
        fig, ax = plt.subplots(figsize=figsize)
        
        for factor in self.factors.columns:
            ax.plot(cumulative_returns.index, cumulative_returns[factor], label=factor)
        
        ax.set_title('Rendements cumulés des facteurs')
        ax.set_xlabel('Date')
        ax.set_ylabel('Rendement cumulé')
        ax.legend()
        ax.grid(True)
        
        return fig

def main():
    """
    Fonction principale pour tester l'optimiseur factoriel.
    """
    # Exemple d'utilisation
    import os
    import sys
    sys.path.append(os.path.abspath("../data"))
    from data_processing import load_data
    
    # Charger les données
    input_dir = "../../data/processed/"
    returns_data = load_data(input_dir + "optimization_clean_returns.pkl")
    
    if returns_data is None:
        logger.error("Impossible de charger les données. Arrêt du traitement.")
        return
    
    # Créer l'optimiseur
    risk_free_rate = 0.02  # 2% annualisé
    optimizer = FactorInvestingOptimizer(returns_data, risk_free_rate=risk_free_rate)
    
    # Extraire les facteurs
    factors, loadings = optimizer.extract_factors(n_factors=3)
    
    # Calculer les expositions aux facteurs
    exposures = optimizer.compute_factor_exposures()
    
    # Calculer les rendements attendus des facteurs et des actifs
    optimizer.compute_factor_expected_returns()
    optimizer.compute_asset_expected_returns()
    
    # Calculer la covariance des facteurs et des actifs
    optimizer.compute_factor_covariance()
    optimizer.compute_asset_covariance()
    
    # Créer un portefeuille basé sur un facteur
    factor_portfolio = optimizer.create_portfolio_by_factor('Factor_1', n_assets=5)
    
    # Créer un portefeuille basé sur une combinaison de facteurs
    factor_weights = {'Factor_1': 0.5, 'Factor_2': 0.3, 'Factor_3': 0.2}
    multi_factor_portfolio = optimizer.create_portfolio_multi_factor(factor_weights, n_assets=5)
    
    # Évaluer les portefeuilles
    factor_eval = optimizer.evaluate_portfolio(factor_portfolio)
    multi_eval = optimizer.evaluate_portfolio(multi_factor_portfolio)
    
    # Afficher les résultats
    print("\nPortefeuille basé sur le facteur 1:")
    print(f"Rendement: {factor_eval['return']:.4f}")
    print(f"Volatilité: {factor_eval['volatility']:.4f}")
    print(f"Ratio de Sharpe: {factor_eval['sharpe_ratio']:.4f}")
    print("\nExpositions aux facteurs:")
    print(factor_eval['factor_exposures'].to_string())
    
    print("\nPortefeuille multi-factoriel:")
    print(f"Rendement: {multi_eval['return']:.4f}")
    print(f"Volatilité: {multi_eval['volatility']:.4f}")
    print(f"Ratio de Sharpe: {multi_eval['sharpe_ratio']:.4f}")
    print("\nExpositions aux facteurs:")
    print(multi_eval['factor_exposures'].to_string())
    
    # Tracer les loadings des facteurs
    fig1 = optimizer.plot_factor_loadings()
    fig1.savefig("../../results/figures/factor_loadings.png")
    
    # Tracer les rendements des facteurs
    fig2 = optimizer.plot_factor_returns()
    fig2.savefig("../../results/figures/factor_returns.png")
    
    logger.info("Test de l'optimiseur factoriel terminé avec succès.")

if __name__ == "__main__":
    main()
