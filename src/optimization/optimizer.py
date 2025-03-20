#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module pour l'optimisation du portefeuille avec contraintes avancées.
Ce module fournit une classe générique pour l'optimisation de portefeuille
qui intègre différentes contraintes et méthodes d'optimisation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco
import logging
import sys
import os

# Ajouter le chemin du répertoire parent pour pouvoir importer les modules
sys.path.append(os.path.abspath(".."))
from optimization.constraints import PortfolioConstraints, create_long_only_bounds, create_risk_constraints

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """
    Classe générique pour l'optimisation de portefeuille avec diverses contraintes.
    """
    
    def __init__(self, expected_returns, cov_matrix, risk_free_rate=0.0):
        """
        Initialise l'optimiseur de portefeuille.
        
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
        
        logger.info(f"Initialisation de l'optimiseur avec {self.n_assets} actifs")
    
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
    
    def optimize_with_constraints(self, objective='sharpe', constraints=None, bounds=None, 
                                 initial_weights=None, target_return=None, 
                                 max_volatility=None):
        """
        Optimise le portefeuille avec des contraintes personnalisées.
        
        Args:
            objective (str): Objectif d'optimisation ('sharpe', 'volatility', 'return')
            constraints (list): Liste de contraintes pour l'optimisation
            bounds (tuple): Bornes pour les poids des actifs
            initial_weights (numpy.ndarray): Poids initiaux pour l'optimisation
            target_return (float): Rendement cible (pour objective='volatility')
            max_volatility (float): Volatilité maximale (pour objective='return')
            
        Returns:
            dict: Informations sur le portefeuille optimisé
        """
        logger.info(f"Optimisation du portefeuille avec objectif: {objective}")
        
        # Contraintes par défaut: somme des poids = 1
        if constraints is None:
            constraints = [{'type': 'eq', 'fun': PortfolioConstraints.sum_to_one_constraint}]
        
        # Bornes par défaut: 0 <= poids <= 1
        if bounds is None:
            bounds = create_long_only_bounds(self.n_assets)
        
        # Poids initiaux par défaut: poids égaux
        if initial_weights is None:
            initial_weights = np.array([1.0 / self.n_assets] * self.n_assets)
        
        # Fonction objectif selon l'objectif choisi
        if objective == 'sharpe':
            objective_function = self._negative_sharpe_ratio
        elif objective == 'volatility':
            if target_return is not None:
                # Ajouter une contrainte de rendement cible
                constraints.append({
                    'type': 'eq',
                    'fun': lambda w: self._portfolio_return(w) - target_return
                })
            objective_function = self._portfolio_volatility
        elif objective == 'return':
            if max_volatility is not None:
                # Ajouter une contrainte de volatilité maximale
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda w: max_volatility - self._portfolio_volatility(w)
                })
            objective_function = lambda w: -self._portfolio_return(w)  # Négatif pour maximisation
        else:
            raise ValueError(f"Objectif d'optimisation non reconnu: {objective}")
        
        # Optimisation
        result = sco.minimize(
            objective_function,
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
    
    def optimize_max_sharpe(self, constraints=None, bounds=None, initial_weights=None):
        """
        Optimise le portefeuille pour maximiser le ratio de Sharpe.
        
        Args:
            constraints (list): Liste de contraintes pour l'optimisation
            bounds (tuple): Bornes pour les poids des actifs
            initial_weights (numpy.ndarray): Poids initiaux pour l'optimisation
            
        Returns:
            dict: Informations sur le portefeuille optimisé
        """
        return self.optimize_with_constraints(
            objective='sharpe',
            constraints=constraints,
            bounds=bounds,
            initial_weights=initial_weights
        )
    
    def optimize_min_volatility(self, constraints=None, bounds=None, initial_weights=None, 
                              target_return=None):
        """
        Optimise le portefeuille pour minimiser la volatilité.
        
        Args:
            constraints (list): Liste de contraintes pour l'optimisation
            bounds (tuple): Bornes pour les poids des actifs
            initial_weights (numpy.ndarray): Poids initiaux pour l'optimisation
            target_return (float): Rendement cible (optionnel)
            
        Returns:
            dict: Informations sur le portefeuille optimisé
        """
        return self.optimize_with_constraints(
            objective='volatility',
            constraints=constraints,
            bounds=bounds,
            initial_weights=initial_weights,
            target_return=target_return
        )
    
    def optimize_max_return(self, constraints=None, bounds=None, initial_weights=None, 
                          max_volatility=None):
        """
        Optimise le portefeuille pour maximiser le rendement.
        
        Args:
            constraints (list): Liste de contraintes pour l'optimisation
            bounds (tuple): Bornes pour les poids des actifs
            initial_weights (numpy.ndarray): Poids initiaux pour l'optimisation
            max_volatility (float): Volatilité maximale (optionnel)
            
        Returns:
            dict: Informations sur le portefeuille optimisé
        """
        return self.optimize_with_constraints(
            objective='return',
            constraints=constraints,
            bounds=bounds,
            initial_weights=initial_weights,
            max_volatility=max_volatility
        )
    
    def generate_efficient_frontier(self, n_points=50, constraints=None, bounds=None):
        """
        Génère la frontière efficiente avec des contraintes personnalisées.
        
        Args:
            n_points (int): Nombre de points sur la frontière
            constraints (list): Liste de contraintes pour l'optimisation
            bounds (tuple): Bornes pour les poids des actifs
            
        Returns:
            dict: Informations sur la frontière efficiente
        """
        logger.info(f"Génération de la frontière efficiente avec {n_points} points")
        
        # Trouver le portefeuille à variance minimale
        min_vol_portfolio = self.optimize_min_volatility(constraints=constraints, bounds=bounds)
        
        # Trouver le portefeuille avec le rendement maximal
        max_return_portfolio = self.optimize_max_return(constraints=constraints, bounds=bounds)
        
        # Générer une plage de rendements
        target_returns = np.linspace(
            min_vol_portfolio['return'],
            max_return_portfolio['return'],
            n_points
        )
        
        # Optimiser pour chaque rendement cible
        efficient_portfolios = []
        
        for target_return in target_returns:
            portfolio = self.optimize_min_volatility(
                constraints=constraints,
                bounds=bounds,
                target_return=target_return
            )
            efficient_portfolios.append(portfolio)
        
        # Extraire les rendements et volatilités
        returns = [p['return'] for p in efficient_portfolios]
        volatilities = [p['volatility'] for p in efficient_portfolios]
        sharpes = [p['sharpe_ratio'] for p in efficient_portfolios]
        
        # Créer la frontière efficiente
        frontier = {
            'returns': returns,
            'volatilities': volatilities,
            'sharpe_ratios': sharpes,
            'portfolios': efficient_portfolios,
            'min_vol_portfolio': min_vol_portfolio,
            'max_return_portfolio': max_return_portfolio
        }
        
        # Trouver le portefeuille avec le meilleur ratio de Sharpe
        max_sharpe_idx = np.argmax(sharpes)
        frontier['max_sharpe_portfolio'] = efficient_portfolios[max_sharpe_idx]
        
        logger.info("Frontière efficiente générée avec succès")
        
        return frontier
    
    def generate_monte_carlo_portfolios(self, n_portfolios=10000, constraints=None, bounds=None):
        """
        Génère des portefeuilles aléatoires pour simuler la distribution.
        
        Args:
            n_portfolios (int): Nombre de portefeuilles à générer
            constraints (list): Liste de contraintes à vérifier
            bounds (tuple): Bornes pour les poids des actifs
            
        Returns:
            dict: Informations sur les portefeuilles générés
        """
        logger.info(f"Génération de {n_portfolios} portefeuilles aléatoires")
        
        # Initialiser les listes pour stocker les résultats
        returns = []
        volatilities = []
        sharpe_ratios = []
        weights_list = []
        
        # Générer des portefeuilles aléatoires
        valid_portfolios = 0
        
        while valid_portfolios < n_portfolios:
            # Générer des poids aléatoires
            weights = np.random.random(self.n_assets)
            weights /= np.sum(weights)  # Normaliser pour que la somme = 1
            
            # Vérifier les bornes
            if bounds is not None:
                valid_bounds = True
                for i, (lower, upper) in enumerate(bounds):
                    if weights[i] < lower or weights[i] > upper:
                        valid_bounds = False
                        break
                if not valid_bounds:
                    continue
            
            # Vérifier les contraintes
            if constraints is not None:
                valid_constraints = True
                for constraint in constraints:
                    if constraint['type'] == 'eq' and abs(constraint['fun'](weights)) > 1e-5:
                        valid_constraints = False
                        break
                    elif constraint['type'] == 'ineq' and constraint['fun'](weights) < 0:
                        valid_constraints = False
                        break
                if not valid_constraints:
                    continue
            
            # Calculer les métriques du portefeuille
            portfolio_return = self._portfolio_return(weights)
            portfolio_volatility = self._portfolio_volatility(weights)
            portfolio_sharpe = self._portfolio_sharpe_ratio(weights)
            
            # Stocker les résultats
            returns.append(portfolio_return)
            volatilities.append(portfolio_volatility)
            sharpe_ratios.append(portfolio_sharpe)
            weights_list.append(weights)
            
            valid_portfolios += 1
            if valid_portfolios % 1000 == 0:
                logger.info(f"Généré {valid_portfolios} portefeuilles valides")
        
        # Créer un DataFrame pour stocker les poids
        weights_df = pd.DataFrame(weights_list, columns=self.assets)
        
        # Créer un DataFrame pour les métriques
        portfolios = pd.DataFrame({
            'return': returns,
            'volatility': volatilities,
            'sharpe_ratio': sharpe_ratios
        })
        
        # Trouver les portefeuilles remarquables
        max_sharpe_idx = np.argmax(sharpe_ratios)
        min_vol_idx = np.argmin(volatilities)
        max_return_idx = np.argmax(returns)
        
        max_sharpe_portfolio = {
            'weights': pd.Series(weights_list[max_sharpe_idx], index=self.assets),
            'return': returns[max_sharpe_idx],
            'volatility': volatilities[max_sharpe_idx],
            'sharpe_ratio': sharpe_ratios[max_sharpe_idx]
        }
        
        min_vol_portfolio = {
            'weights': pd.Series(weights_list[min_vol_idx], index=self.assets),
            'return': returns[min_vol_idx],
            'volatility': volatilities[min_vol_idx],
            'sharpe_ratio': sharpe_ratios[min_vol_idx]
        }
        
        max_return_portfolio = {
            'weights': pd.Series(weights_list[max_return_idx], index=self.assets),
            'return': returns[max_return_idx],
            'volatility': volatilities[max_return_idx],
            'sharpe_ratio': sharpe_ratios[max_return_idx]
        }
        
        # Rassembler les résultats
        results = {
            'portfolios': portfolios,
            'weights': weights_df,
            'max_sharpe_portfolio': max_sharpe_portfolio,
            'min_vol_portfolio': min_vol_portfolio,
            'max_return_portfolio': max_return_portfolio
        }
        
        logger.info("Simulation Monte Carlo terminée avec succès")
        
        return results
    
    def plot_efficient_frontier(self, show_assets=True, show_mc=False, n_mc_portfolios=5000,
                              show_equal_weight=True, constraints=None, bounds=None,
                              figsize=(12, 8)):
        """
        Trace la frontière efficiente et éventuellement d'autres éléments.
        
        Args:
            show_assets (bool): Afficher les actifs individuels
            show_mc (bool): Afficher des portefeuilles aléatoires (Monte Carlo)
            n_mc_portfolios (int): Nombre de portefeuilles aléatoires
            show_equal_weight (bool): Afficher le portefeuille équipondéré
            constraints (list): Liste de contraintes pour l'optimisation
            bounds (tuple): Bornes pour les poids des actifs
            figsize (tuple): Dimensions de la figure
            
        Returns:
            matplotlib.figure.Figure: Figure matplotlib
        """
        logger.info("Traçage de la frontière efficiente")
        
        # Générer la frontière efficiente
        frontier = self.generate_efficient_frontier(
            n_points=50,
            constraints=constraints,
            bounds=bounds
        )
        
        # Créer la figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Tracer la frontière efficiente
        ax.plot(frontier['volatilities'], frontier['returns'], 'b-', linewidth=3, label='Frontière Efficiente')
        
        # Tracer le portefeuille à variance minimale
        min_vol = frontier['min_vol_portfolio']
        ax.scatter(min_vol['volatility'], min_vol['return'], 
                  s=200, color='g', marker='*', 
                  label='Minimum Variance')
        
        # Tracer le portefeuille avec le ratio de Sharpe optimal
        max_sharpe = frontier['max_sharpe_portfolio']
        ax.scatter(max_sharpe['volatility'], max_sharpe['return'], 
                  s=200, color='r', marker='*', 
                  label='Maximum Sharpe Ratio')
        
        # Tracer la ligne du taux sans risque (CML)
        x_vals = np.linspace(0, max(frontier['volatilities']) * 1.2, 100)
        y_vals = self.risk_free_rate + max_sharpe['sharpe_ratio'] * x_vals
        ax.plot(x_vals, y_vals, 'r--', label='Capital Market Line')
        
        # Tracer les actifs individuels
        if show_assets:
            for i, asset in enumerate(self.assets):
                risk = np.sqrt(self.cov_matrix.iloc[i, i])
                ret = self.expected_returns.iloc[i]
                ax.scatter(risk, ret, s=100, alpha=0.6, label=asset)
        
        # Tracer des portefeuilles aléatoires
        if show_mc:
            mc_results = self.generate_monte_carlo_portfolios(
                n_portfolios=n_mc_portfolios,
                constraints=constraints,
                bounds=bounds
            )
            
            ax.scatter(
                mc_results['portfolios']['volatility'],
                mc_results['portfolios']['return'],
                s=5, alpha=0.3, color='gray',
                label='Portefeuilles Aléatoires'
            )
        
        # Tracer le portefeuille équipondéré
        if show_equal_weight:
            equal_weights = np.array([1.0 / self.n_assets] * self.n_assets)
            equal_return = self._portfolio_return(equal_weights)
            equal_volatility = self._portfolio_volatility(equal_weights)
            
            ax.scatter(
                equal_volatility, equal_return,
                s=200, color='purple', marker='D',
                label='Equal Weight'
            )
        
        # Configurer le graphique
        ax.set_title('Frontière Efficiente avec Contraintes', fontsize=16)
        ax.set_xlabel('Volatilité (Risque)', fontsize=14)
        ax.set_ylabel('Rendement Espéré', fontsize=14)
        ax.grid(True)
        ax.legend()
        
        return fig
    
    def analyze_asset_allocation(self, portfolio):
        """
        Analyse l'allocation d'actifs d'un portefeuille.
        
        Args:
            portfolio (dict): Portefeuille à analyser (avec clé 'weights')
            
        Returns:
            pandas.DataFrame: Analyse de l'allocation
        """
        if 'weights' not in portfolio:
            raise ValueError("Le portefeuille doit contenir une clé 'weights'")
        
        weights = portfolio['weights']
        
        # Filtrer les poids significatifs
        significant_weights = weights[weights >= 0.01].sort_values(ascending=False)
        other_weight = weights[weights < 0.01].sum()
        
        # Ajouter une catégorie "Autres" si nécessaire
        if other_weight > 0:
            allocation = significant_weights.append(pd.Series(other_weight, index=['Autres']))
        else:
            allocation = significant_weights
        
        # Calculer des statistiques sur l'allocation
        stats = {
            'Nombre total d\'actifs': len(weights),
            'Nombre d\'actifs avec poids > 1%': len(significant_weights),
            'Poids maximum': weights.max(),
            'Actif le plus pondéré': weights.idxmax(),
            'Concentration (HHI)': np.sum(weights ** 2)
        }
        
        return {
            'allocation': allocation,
            'stats': pd.Series(stats)
        }

def main():
    """
    Fonction principale pour tester l'optimiseur.
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
    
    if expected_returns is None or cov_matrix is None:
        logger.error("Impossible de charger les données. Arrêt du traitement.")
        return
    
    # Créer l'optimiseur
    risk_free_rate = 0.02  # 2% annualisé
    optimizer = PortfolioOptimizer(expected_returns, cov_matrix, risk_free_rate)
    
    # Ajouter des contraintes de concentration
    max_weight = 0.3  # Poids maximum par actif
    bounds = create_long_only_bounds(optimizer.n_assets)
    
    constraints = [
        {'type': 'eq', 'fun': PortfolioConstraints.sum_to_one_constraint},
        {'type': 'ineq', 'fun': lambda w: max_weight - np.max(w)}  # Contrainte de concentration
    ]
    
    # Optimiser le portefeuille
    optimal_portfolio = optimizer.optimize_max_sharpe(constraints=constraints, bounds=bounds)
    min_vol_portfolio = optimizer.optimize_min_volatility(constraints=constraints, bounds=bounds)
    
    # Analyser l'allocation
    allocation_analysis = optimizer.analyze_asset_allocation(optimal_portfolio)
    
    # Afficher les résultats
    print("\nPortefeuille optimal (Sharpe):")
    print(f"Rendement: {optimal_portfolio['return']:.4f}")
    print(f"Volatilité: {optimal_portfolio['volatility']:.4f}")
    print(f"Ratio de Sharpe: {optimal_portfolio['sharpe_ratio']:.4f}")
    print("\nAllocations:")
    print(allocation_analysis['allocation'].to_string())
    print("\nStatistiques:")
    print(allocation_analysis['stats'].to_string())
    
    # Tracer la frontière efficiente
    fig = optimizer.plot_efficient_frontier(
        show_assets=True,
        show_mc=True,
        n_mc_portfolios=1000,
        constraints=constraints,
        bounds=bounds
    )
    fig.savefig("../../results/figures/efficient_frontier_constrained.png")
    
    logger.info("Test de l'optimiseur terminé avec succès.")

if __name__ == "__main__":
    main()
