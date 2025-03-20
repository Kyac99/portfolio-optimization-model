#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module pour l'optimisation de portefeuille selon la théorie de Markowitz.
Ce module implémente l'optimisation moyenne-variance de Markowitz
pour trouver des portefeuilles efficients.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarkowitzOptimizer:
    """
    Classe implémentant l'optimisation de Markowitz.
    """
    
    def __init__(self, expected_returns, cov_matrix, risk_free_rate=0.0):
        """
        Initialise l'optimiseur de Markowitz.
        
        Args:
            expected_returns (pandas.Series): Rendements attendus des actifs
            cov_matrix (pandas.DataFrame): Matrice de covariance des actifs
            risk_free_rate (float): Taux sans risque (annualisé)
        """
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.assets = list(expected_returns.index)
        self.n_assets = len(self.assets)
        
        logger.info(f"Initialisation de l'optimiseur de Markowitz avec {self.n_assets} actifs")
    
    def _portfolio_return(self, weights):
        """
        Calcule le rendement attendu d'un portefeuille.
        
        Args:
            weights (numpy.ndarray): Poids des actifs dans le portefeuille
            
        Returns:
            float: Rendement attendu du portefeuille
        """
        return np.sum(self.expected_returns * weights)
    
    def _portfolio_volatility(self, weights):
        """
        Calcule la volatilité d'un portefeuille.
        
        Args:
            weights (numpy.ndarray): Poids des actifs dans le portefeuille
            
        Returns:
            float: Volatilité du portefeuille
        """
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
    
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
    
    def _minimize_volatility(self, target_return):
        """
        Minimise la volatilité pour un rendement cible.
        
        Args:
            target_return (float): Rendement cible
            
        Returns:
            numpy.ndarray: Poids optimisés
        """
        # Contraintes: somme des poids = 1, rendement = target_return
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: self._portfolio_return(x) - target_return}
        )
        
        # Bornes: poids entre 0 et 1 (pas de vente à découvert)
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Point de départ: poids égaux
        initial_weights = np.array([1.0 / self.n_assets] * self.n_assets)
        
        # Optimisation
        result = sco.minimize(
            self._portfolio_volatility,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result['x']
    
    def optimize_sharpe_ratio(self, allow_short=False):
        """
        Optimise le ratio de Sharpe du portefeuille.
        
        Args:
            allow_short (bool): Autoriser les ventes à découvert
            
        Returns:
            dict: Informations sur le portefeuille optimisé
        """
        logger.info("Optimisation du ratio de Sharpe")
        
        # Contraintes: somme des poids = 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bornes: poids entre 0 et 1 (pas de vente à découvert) ou entre -1 et 1 (vente à découvert)
        if allow_short:
            bounds = tuple((-1, 1) for _ in range(self.n_assets))
        else:
            bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Point de départ: poids égaux
        initial_weights = np.array([1.0 / self.n_assets] * self.n_assets)
        
        # Optimisation
        result = sco.minimize(
            self._negative_sharpe_ratio,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Vérifier la convergence
        if not result['success']:
            logger.warning(f"L'optimisation n'a pas convergé: {result['message']}")
        
        # Extraire les poids optimaux
        optimal_weights = result['x']
        
        # Calculer les métriques du portefeuille optimal
        optimal_portfolio = {
            'weights': pd.Series(optimal_weights, index=self.assets),
            'return': self._portfolio_return(optimal_weights),
            'volatility': self._portfolio_volatility(optimal_weights),
            'sharpe_ratio': self._portfolio_sharpe_ratio(optimal_weights)
        }
        
        logger.info(f"Portefeuille optimal trouvé avec Sharpe ratio: {optimal_portfolio['sharpe_ratio']:.4f}")
        
        return optimal_portfolio
    
    def optimize_min_volatility(self):
        """
        Trouve le portefeuille de variance minimale.
        
        Returns:
            dict: Informations sur le portefeuille à variance minimale
        """
        logger.info("Recherche du portefeuille à variance minimale")
        
        # Contraintes: somme des poids = 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bornes: poids entre 0 et 1 (pas de vente à découvert)
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Point de départ: poids égaux
        initial_weights = np.array([1.0 / self.n_assets] * self.n_assets)
        
        # Optimisation
        result = sco.minimize(
            self._portfolio_volatility,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Vérifier la convergence
        if not result['success']:
            logger.warning(f"L'optimisation n'a pas convergé: {result['message']}")
        
        # Extraire les poids optimaux
        optimal_weights = result['x']
        
        # Calculer les métriques du portefeuille optimal
        min_vol_portfolio = {
            'weights': pd.Series(optimal_weights, index=self.assets),
            'return': self._portfolio_return(optimal_weights),
            'volatility': self._portfolio_volatility(optimal_weights),
            'sharpe_ratio': self._portfolio_sharpe_ratio(optimal_weights)
        }
        
        logger.info(f"Portefeuille à variance minimale trouvé avec volatilité: {min_vol_portfolio['volatility']:.4f}")
        
        return min_vol_portfolio
    
    def optimize_for_target_return(self, target_return):
        """
        Optimise le portefeuille pour un rendement cible.
        
        Args:
            target_return (float): Rendement cible
            
        Returns:
            dict: Informations sur le portefeuille optimisé
        """
        logger.info(f"Optimisation pour un rendement cible de {target_return:.4f}")
        
        # Vérifier que le rendement cible est atteignable
        max_return = np.max(self.expected_returns)
        min_return = np.min(self.expected_returns)
        
        if target_return > max_return:
            logger.warning(f"Rendement cible {target_return:.4f} supérieur au rendement maximal {max_return:.4f}")
            target_return = max_return
        elif target_return < min_return:
            logger.warning(f"Rendement cible {target_return:.4f} inférieur au rendement minimal {min_return:.4f}")
            target_return = min_return
        
        # Optimisation
        optimal_weights = self._minimize_volatility(target_return)
        
        # Calculer les métriques du portefeuille optimal
        target_portfolio = {
            'weights': pd.Series(optimal_weights, index=self.assets),
            'return': self._portfolio_return(optimal_weights),
            'volatility': self._portfolio_volatility(optimal_weights),
            'sharpe_ratio': self._portfolio_sharpe_ratio(optimal_weights)
        }
        
        logger.info(f"Portefeuille trouvé avec volatilité: {target_portfolio['volatility']:.4f}")
        
        return target_portfolio
    
    def optimize_for_target_risk(self, target_risk):
        """
        Optimise le portefeuille pour une volatilité cible.
        
        Args:
            target_risk (float): Volatilité cible
            
        Returns:
            dict: Informations sur le portefeuille optimisé
        """
        logger.info(f"Optimisation pour une volatilité cible de {target_risk:.4f}")
        
        # Trouver portefeuilles à variance minimale et max Sharpe Ratio
        min_vol_portfolio = self.optimize_min_volatility()
        max_sharpe_portfolio = self.optimize_sharpe_ratio()
        
        min_risk = min_vol_portfolio['volatility']
        max_risk = max_sharpe_portfolio['volatility']
        
        # Vérifier que la volatilité cible est atteignable
        if target_risk < min_risk:
            logger.warning(f"Volatilité cible {target_risk:.4f} inférieure à la volatilité minimale {min_risk:.4f}")
            return min_vol_portfolio
        
        # Optimisation
        # On va chercher parmi les portefeuilles efficients celui qui a la volatilité la plus proche de la cible
        # On utilise une approche binaire pour trouver le rendement qui donne la volatilité cible
        
        # Plage de rendements possibles
        returns_range = np.linspace(min_vol_portfolio['return'], max_sharpe_portfolio['return'], 100)
        
        # Trouver le portefeuille avec la volatilité la plus proche de la cible
        closest_portfolio = None
        min_diff = float('inf')
        
        for target_return in returns_range:
            portfolio = self.optimize_for_target_return(target_return)
            diff = abs(portfolio['volatility'] - target_risk)
            
            if diff < min_diff:
                min_diff = diff
                closest_portfolio = portfolio
            
            # Si on est assez proche, on s'arrête
            if diff < 1e-4:
                break
        
        logger.info(f"Portefeuille trouvé avec volatilité: {closest_portfolio['volatility']:.4f}")
        
        return closest_portfolio
    
    def generate_efficient_frontier(self, n_points=100):
        """
        Génère la frontière efficiente.
        
        Args:
            n_points (int): Nombre de points sur la frontière
            
        Returns:
            dict: Informations sur la frontière efficiente
        """
        logger.info(f"Génération de la frontière efficiente avec {n_points} points")
        
        # Trouver le portefeuille à variance minimale
        min_vol_portfolio = self.optimize_min_volatility()
        
        # Trouver le portefeuille avec le rendement maximal
        max_return_idx = np.argmax(self.expected_returns)
        max_return = self.expected_returns.iloc[max_return_idx]
        max_return_asset = self.assets[max_return_idx]
        
        # Générer une plage de rendements
        returns_range = np.linspace(min_vol_portfolio['return'], max_return, n_points)
        
        # Optimiser pour chaque rendement cible
        efficient_portfolios = []
        
        for target_return in returns_range:
            portfolio = self.optimize_for_target_return(target_return)
            efficient_portfolios.append(portfolio)
        
        # Extraire les rendements et volatilités
        returns = [p['return'] for p in efficient_portfolios]
        volatilities = [p['volatility'] for p in efficient_portfolios]
        sharpes = [p['sharpe_ratio'] for p in efficient_portfolios]
        
        frontier = {
            'returns': returns,
            'volatilities': volatilities,
            'sharpe_ratios': sharpes,
            'portfolios': efficient_portfolios
        }
        
        logger.info("Frontière efficiente générée avec succès")
        
        return frontier
    
    def add_constraints(self, constraints_dict=None):
        """
        Ajoute des contraintes personnalisées à l'optimisation.
        
        Args:
            constraints_dict (dict): Dictionnaire de contraintes
                Clés possibles:
                - 'max_weight': Poids maximum par actif
                - 'min_weight': Poids minimum par actif
                - 'asset_classes': Dictionnaire des classes d'actifs et leurs contraintes
                
        Note:
            Cette méthode modifie l'instance.
        """
        if constraints_dict is None:
            return
        
        logger.info("Ajout de contraintes personnalisées")
        
        self.constraints_dict = constraints_dict
        
        # Traiter les contraintes ici (à implémenter dans les méthodes d'optimisation)
    
    def plot_efficient_frontier(self, show_assets=True, show_sharpe=True, 
                              show_min_vol=True, risk_free_line=True, 
                              figsize=(12, 8)):
        """
        Trace la frontière efficiente et éventuellement les actifs individuels.
        
        Args:
            show_assets (bool): Afficher les actifs individuels
            show_sharpe (bool): Afficher le portefeuille de Sharpe optimal
            show_min_vol (bool): Afficher le portefeuille à variance minimale
            risk_free_line (bool): Afficher la ligne du taux sans risque
            figsize (tuple): Dimensions de la figure
            
        Returns:
            matplotlib.figure.Figure: Figure matplotlib
        """
        logger.info("Traçage de la frontière efficiente")
        
        # Générer la frontière efficiente
        frontier = self.generate_efficient_frontier()
        
        # Créer la figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Tracer la frontière efficiente
        ax.plot(frontier['volatilities'], frontier['returns'], 'b-', linewidth=3, label='Frontière Efficiente')
        
        # Afficher les actifs individuels
        if show_assets:
            for i, asset in enumerate(self.assets):
                risk = np.sqrt(self.cov_matrix.iloc[i, i])
                ret = self.expected_returns.iloc[i]
                ax.scatter(risk, ret, s=100, alpha=0.6, label=asset)
        
        # Afficher le portefeuille à variance minimale
        if show_min_vol:
            min_vol_portfolio = self.optimize_min_volatility()
            ax.scatter(min_vol_portfolio['volatility'], min_vol_portfolio['return'], 
                      s=200, color='g', marker='*', 
                      label='Minimum Variance')
        
        # Afficher le portefeuille avec le ratio de Sharpe optimal
        if show_sharpe:
            max_sharpe_portfolio = self.optimize_sharpe_ratio()
            ax.scatter(max_sharpe_portfolio['volatility'], max_sharpe_portfolio['return'], 
                      s=200, color='r', marker='*', 
                      label='Maximum Sharpe Ratio')
        
        # Afficher la ligne du taux sans risque
        if risk_free_line and show_sharpe:
            max_sharpe_portfolio = self.optimize_sharpe_ratio()
            x_vals = np.linspace(0, max(frontier['volatilities']) * 1.2, 100)
            y_vals = self.risk_free_rate + max_sharpe_portfolio['sharpe_ratio'] * x_vals
            ax.plot(x_vals, y_vals, 'r--', label='Capital Market Line')
        
        # Configurer le graphique
        ax.set_title('Frontière Efficiente de Markowitz', fontsize=16)
        ax.set_xlabel('Volatilité (Risque)', fontsize=14)
        ax.set_ylabel('Rendement Espéré', fontsize=14)
        ax.grid(True)
        ax.legend()
        
        return fig

def main():
    """
    Fonction principale pour tester l'optimiseur de Markowitz.
    """
    # Exemple d'utilisation
    import os
    import sys
    sys.path.append(os.path.abspath("../data"))
    from data_processing import load_data
    
    # Charger les données
    input_dir = "../../data/processed/"
    expected_returns = load_data(input_dir + "optimization_expected_returns.pkl")
    cov_matrix = load_data(input_dir + "optimization_covariance_matrix.pkl")
    risk_free_rate = 0.02  # 2% annualisé
    
    if expected_returns is None or cov_matrix is None:
        logger.error("Impossible de charger les données. Arrêt du traitement.")
        return
    
    # Créer l'optimiseur
    optimizer = MarkowitzOptimizer(expected_returns, cov_matrix, risk_free_rate)
    
    # Optimiser le portefeuille
    optimal_portfolio = optimizer.optimize_sharpe_ratio()
    min_vol_portfolio = optimizer.optimize_min_volatility()
    
    # Afficher les résultats
    print("\nPortefeuille optimal (Sharpe):")
    print(f"Rendement: {optimal_portfolio['return']:.4f}")
    print(f"Volatilité: {optimal_portfolio['volatility']:.4f}")
    print(f"Ratio de Sharpe: {optimal_portfolio['sharpe_ratio']:.4f}")
    print("\nAllocations:")
    print(optimal_portfolio['weights'].sort_values(ascending=False).to_string())
    
    print("\nPortefeuille à variance minimale:")
    print(f"Rendement: {min_vol_portfolio['return']:.4f}")
    print(f"Volatilité: {min_vol_portfolio['volatility']:.4f}")
    print(f"Ratio de Sharpe: {min_vol_portfolio['sharpe_ratio']:.4f}")
    print("\nAllocations:")
    print(min_vol_portfolio['weights'].sort_values(ascending=False).to_string())
    
    # Tracer la frontière efficiente
    fig = optimizer.plot_efficient_frontier()
    fig.savefig("../../results/figures/efficient_frontier.png")
    
    logger.info("Test de l'optimiseur de Markowitz terminé avec succès.")

if __name__ == "__main__":
    main()
