#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module pour l'optimisation de portefeuille selon le modèle de Black-Litterman.
Ce module implémente le modèle Black-Litterman qui permet d'incorporer
des vues subjectives sur les rendements futurs dans le cadre d'une
optimisation de portefeuille.
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

class BlackLittermanOptimizer:
    """
    Classe implémentant l'optimisation de Black-Litterman.
    """
    
    def __init__(self, market_caps, cov_matrix, risk_free_rate=0.0, market_return=None, delta=2.5):
        """
        Initialise l'optimiseur de Black-Litterman.
        
        Args:
            market_caps (pandas.Series): Capitalisations boursières des actifs
            cov_matrix (pandas.DataFrame): Matrice de covariance des actifs
            risk_free_rate (float): Taux sans risque (annualisé)
            market_return (float): Rendement attendu du marché (annualisé)
            delta (float): Coefficient d'aversion au risque
        """
        self.market_caps = market_caps
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.assets = list(market_caps.index)
        self.n_assets = len(self.assets)
        
        # Calculer les poids implicites du marché
        self.market_weights = market_caps / market_caps.sum()
        
        # Calculer le rendement implicite du marché si non fourni
        if market_return is None:
            # Rendement du marché basé sur CAPM
            # E[R_m] = R_f + δ * σ²_m
            market_var = np.dot(self.market_weights.T, np.dot(self.cov_matrix, self.market_weights))
            market_return = risk_free_rate + delta * market_var
        
        self.market_return = market_return
        self.delta = delta
        
        # Calculer les rendements implicites selon CAPM
        self.pi = self._calculate_implied_returns()
        
        logger.info(f"Initialisation de l'optimiseur de Black-Litterman avec {self.n_assets} actifs")
        logger.info(f"Rendement implicite du marché: {self.market_return:.4f}")
    
    def _calculate_implied_returns(self):
        """
        Calcule les rendements implicites des actifs selon CAPM.
        
        Returns:
            pandas.Series: Rendements implicites
        """
        # π = δ * Σ * w
        pi = self.delta * np.dot(self.cov_matrix, self.market_weights)
        return pd.Series(pi, index=self.assets)
    
    def incorporate_views(self, P, Q, omega=None, tau=0.05):
        """
        Incorpore des vues subjectives dans le modèle.
        
        Args:
            P (numpy.ndarray ou pandas.DataFrame): Matrice des vues (k x n)
            Q (numpy.ndarray ou pandas.Series): Vecteur des rendements attendus des vues (k x 1)
            omega (numpy.ndarray, optional): Matrice de covariance des vues (k x k)
            tau (float, optional): Paramètre de confiance dans les priors (0 < tau < 1)
            
        Returns:
            pandas.Series: Rendements révisés selon Black-Litterman
        """
        logger.info("Incorporation des vues subjectives dans le modèle")
        
        # Convertir P et Q en arrays numpy si nécessaire
        if isinstance(P, pd.DataFrame):
            P = P.values
        if isinstance(Q, pd.Series):
            Q = Q.values
        
        # Calculer Omega si non fourni
        if omega is None:
            # Omega proportionnelle à la variance des vues
            # Omega = diag(P * (tau * Sigma) * P')
            omega = np.diag(np.dot(P, np.dot(tau * self.cov_matrix, P.T)))
        
        # Calculer les rendements révisés selon Black-Litterman
        # E[R] = [(tau * Sigma)^-1 + P' * Omega^-1 * P]^-1 * [(tau * Sigma)^-1 * pi + P' * Omega^-1 * Q]
        
        # Termes intermédiaires
        tau_sigma_inv = np.linalg.inv(tau * self.cov_matrix)
        omega_inv = np.linalg.inv(omega)
        
        # Premier terme : (tau * Sigma)^-1 * pi
        first_term = np.dot(tau_sigma_inv, self.pi)
        
        # Deuxième terme : P' * Omega^-1 * Q
        second_term = np.dot(P.T, np.dot(omega_inv, Q))
        
        # Terme pour l'inversion : [(tau * Sigma)^-1 + P' * Omega^-1 * P]
        coef_term = tau_sigma_inv + np.dot(P.T, np.dot(omega_inv, P))
        
        # Rendements révisés
        E_bl = np.dot(np.linalg.inv(coef_term), first_term + second_term)
        
        # Convertir en Series pandas
        bl_returns = pd.Series(E_bl, index=self.assets)
        
        # Covariance révisée (optionnelle)
        # M = [(tau * Sigma)^-1 + P' * Omega^-1 * P]^-1
        # Sigma_bl = Sigma + M
        M = np.linalg.inv(coef_term)
        bl_cov = self.cov_matrix + M
        
        self.bl_returns = bl_returns
        self.bl_cov = pd.DataFrame(bl_cov, index=self.assets, columns=self.assets)
        
        logger.info("Vues incorporées avec succès dans le modèle")
        
        return bl_returns
    
    def _portfolio_return(self, weights):
        """
        Calcule le rendement attendu d'un portefeuille.
        
        Args:
            weights (numpy.ndarray): Poids des actifs dans le portefeuille
            
        Returns:
            float: Rendement attendu du portefeuille
        """
        if hasattr(self, 'bl_returns'):
            return np.sum(self.bl_returns * weights)
        else:
            return np.sum(self.pi * weights)
    
    def _portfolio_volatility(self, weights):
        """
        Calcule la volatilité d'un portefeuille.
        
        Args:
            weights (numpy.ndarray): Poids des actifs dans le portefeuille
            
        Returns:
            float: Volatilité du portefeuille
        """
        if hasattr(self, 'bl_cov'):
            return np.sqrt(np.dot(weights.T, np.dot(self.bl_cov, weights)))
        else:
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
    
    def optimize_portfolio(self, allow_short=False, max_weight=None):
        """
        Optimise le portefeuille selon le modèle de Black-Litterman.
        
        Args:
            allow_short (bool): Autoriser les ventes à découvert
            max_weight (float, optional): Poids maximum par actif
            
        Returns:
            dict: Informations sur le portefeuille optimisé
        """
        logger.info("Optimisation du portefeuille selon Black-Litterman")
        
        # Contraintes: somme des poids = 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bornes selon les paramètres
        if allow_short:
            bounds = tuple((-1, 1) for _ in range(self.n_assets))
        elif max_weight is not None:
            bounds = tuple((0, max_weight) for _ in range(self.n_assets))
        else:
            bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Point de départ: poids du marché
        initial_weights = self.market_weights.values
        
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
        
        # Point de départ: poids du marché
        initial_weights = self.market_weights.values
        
        # Optimisation
        result = sco.minimize(
            self._portfolio_volatility,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result['x']
    
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
        
        # Point de départ: poids du marché
        initial_weights = self.market_weights.values
        
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
        if hasattr(self, 'bl_returns'):
            max_return_idx = np.argmax(self.bl_returns)
            max_return = self.bl_returns.iloc[max_return_idx]
        else:
            max_return_idx = np.argmax(self.pi)
            max_return = self.pi.iloc[max_return_idx]
        
        # Générer une plage de rendements
        returns_range = np.linspace(min_vol_portfolio['return'], max_return, n_points)
        
        # Optimiser pour chaque rendement cible
        efficient_portfolios = []
        
        for target_return in returns_range:
            weights = self._minimize_volatility(target_return)
            portfolio = {
                'weights': pd.Series(weights, index=self.assets),
                'return': self._portfolio_return(weights),
                'volatility': self._portfolio_volatility(weights),
                'sharpe_ratio': self._portfolio_sharpe_ratio(weights)
            }
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
    
    def plot_comparison(self, original_frontier=None, title="Comparaison des frontières efficientes", figsize=(12, 8)):
        """
        Trace la comparaison entre la frontière efficiente originale et celle de Black-Litterman.
        
        Args:
            original_frontier (dict, optional): Frontière efficiente originale
            title (str): Titre du graphique
            figsize (tuple): Dimensions de la figure
            
        Returns:
            matplotlib.figure.Figure: Figure matplotlib
        """
        # Générer la frontière efficiente de Black-Litterman
        bl_frontier = self.generate_efficient_frontier()
        
        # Créer la figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Tracer la frontière efficiente de Black-Litterman
        ax.plot(bl_frontier['volatilities'], bl_frontier['returns'], 'g-', linewidth=3, 
               label='Frontier - Black-Litterman')
        
        # Tracer le portefeuille optimal de Black-Litterman
        bl_portfolio = self.optimize_portfolio()
        ax.scatter(bl_portfolio['volatility'], bl_portfolio['return'], 
                  s=200, color='g', marker='*', 
                  label='Optimal - Black-Litterman')
        
        # Tracer la frontière efficiente originale si fournie
        if original_frontier is not None:
            ax.plot(original_frontier['volatilities'], original_frontier['returns'], 'b--', 
                   linewidth=2, label='Frontier - Original')
            
            # Trouver l'index du portefeuille avec le meilleur ratio de Sharpe
            best_idx = np.argmax(original_frontier['sharpe_ratios'])
            best_portfolio = original_frontier['portfolios'][best_idx]
            
            ax.scatter(best_portfolio['volatility'], best_portfolio['return'], 
                      s=200, color='b', marker='*', 
                      label='Optimal - Original')
        
        # Afficher les poids du marché
        market_return = self._portfolio_return(self.market_weights.values)
        market_vol = self._portfolio_volatility(self.market_weights.values)
        
        ax.scatter(market_vol, market_return, s=200, color='r', marker='o', label='Market Portfolio')
        
        # Configurer le graphique
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Volatilité (Risque)', fontsize=14)
        ax.set_ylabel('Rendement Espéré', fontsize=14)
        ax.grid(True)
        ax.legend()
        
        return fig
    
    def get_views_impact(self):
        """
        Analyse l'impact des vues subjectives sur les rendements attendus.
        
        Returns:
            pandas.DataFrame: Comparaison des rendements avant/après les vues
        """
        if not hasattr(self, 'bl_returns'):
            logger.warning("Aucune vue n'a été incorporée au modèle")
            return None
        
        # Comparer les rendements implicites et les rendements Black-Litterman
        comparison = pd.DataFrame({
            'CAPM Returns': self.pi,
            'BL Returns': self.bl_returns,
            'Difference': self.bl_returns - self.pi,
            'Percent Change': ((self.bl_returns - self.pi) / self.pi) * 100
        })
        
        return comparison.sort_values('Percent Change', ascending=False)

def create_view_matrix(assets, view_dict):
    """
    Crée une matrice de vues P et un vecteur de rendements Q.
    
    Args:
        assets (list): Liste des actifs du portefeuille
        view_dict (dict): Dictionnaire des vues
            Format: {'view1': {'ticker1': weight1, 'ticker2': weight2, ...}, 'return': expected_return}
            
    Returns:
        tuple: (P, Q)
    """
    n_assets = len(assets)
    n_views = len(view_dict)
    
    # Initialiser la matrice P et le vecteur Q
    P = np.zeros((n_views, n_assets))
    Q = np.zeros(n_views)
    
    # Remplir P et Q selon les vues
    for i, (view_name, view_content) in enumerate(view_dict.items()):
        for ticker, weight in view_content.items():
            if ticker == 'return':
                Q[i] = view_content['return']
            else:
                asset_idx = assets.index(ticker)
                P[i, asset_idx] = weight
    
    return P, Q

def main():
    """
    Fonction principale pour tester l'optimiseur de Black-Litterman.
    """
    # Exemple d'utilisation
    import os
    import sys
    import numpy as np
    sys.path.append(os.path.abspath("../data"))
    from data_processing import load_data
    
    # Charger les données
    input_dir = "../../data/processed/"
    cov_matrix = load_data(input_dir + "optimization_covariance_matrix.pkl")
    
    if cov_matrix is None:
        logger.error("Impossible de charger les données. Arrêt du traitement.")
        return
    
    # Créer des capitalisations boursières fictives pour l'exemple
    # En pratique, ces données devraient être chargées
    assets = list(cov_matrix.index)
    market_caps = pd.Series(np.random.uniform(1, 100, len(assets)), index=assets)
    
    # Créer l'optimiseur
    risk_free_rate = 0.02  # 2% annualisé
    optimizer = BlackLittermanOptimizer(market_caps, cov_matrix, risk_free_rate)
    
    # Rendements implicites
    print("\nRendements implicites (CAPM):")
    print(optimizer.pi.sort_values(ascending=False).to_string())
    
    # Définir des vues subjectives
    # Exemple: "L'actif A surperformera l'actif B de 5%"
    views = {
        'view1': {'AAPL': 1.0, 'MSFT': -1.0, 'return': 0.05},  # AAPL surperforme MSFT de 5%
        'view2': {'AMZN': 1.0, 'return': 0.15},  # AMZN a un rendement de 15%
    }
    
    # Créer la matrice de vues
    P, Q = create_view_matrix(assets, views)
    
    # Incorporer les vues
    bl_returns = optimizer.incorporate_views(P, Q)
    
    # Afficher l'impact des vues
    print("\nImpact des vues:")
    print(optimizer.get_views_impact().to_string())
    
    # Optimiser le portefeuille
    optimal_portfolio = optimizer.optimize_portfolio()
    
    # Afficher les résultats
    print("\nPortefeuille optimal Black-Litterman:")
    print(f"Rendement: {optimal_portfolio['return']:.4f}")
    print(f"Volatilité: {optimal_portfolio['volatility']:.4f}")
    print(f"Ratio de Sharpe: {optimal_portfolio['sharpe_ratio']:.4f}")
    print("\nAllocations:")
    print(optimal_portfolio['weights'].sort_values(ascending=False).to_string())
    
    # Comparer avec le portefeuille du marché
    market_return = optimizer._portfolio_return(optimizer.market_weights.values)
    market_volatility = optimizer._portfolio_volatility(optimizer.market_weights.values)
    market_sharpe = optimizer._portfolio_sharpe_ratio(optimizer.market_weights.values)
    
    print("\nPortefeuille du marché:")
    print(f"Rendement: {market_return:.4f}")
    print(f"Volatilité: {market_volatility:.4f}")
    print(f"Ratio de Sharpe: {market_sharpe:.4f}")
    
    # Tracer la comparaison
    fig = optimizer.plot_comparison()
    fig.savefig("../../results/figures/bl_comparison.png")
    
    logger.info("Test de l'optimiseur de Black-Litterman terminé avec succès.")

if __name__ == "__main__":
    main()
