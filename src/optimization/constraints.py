#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module pour la gestion des contraintes d'optimisation de portefeuille.
Ce module fournit des fonctions et classes pour définir et appliquer
différentes contraintes dans le processus d'optimisation.
"""

import numpy as np
import pandas as pd
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PortfolioConstraints:
    """
    Classe regroupant différentes contraintes pour l'optimisation de portefeuille.
    """
    
    @staticmethod
    def sum_to_one_constraint(weights):
        """
        Contrainte: la somme des poids doit être égale à 1.
        
        Args:
            weights (numpy.ndarray): Poids des actifs
            
        Returns:
            float: Écart par rapport à la contrainte (0 = respectée)
        """
        return np.sum(weights) - 1
    
    @staticmethod
    def long_only_constraint(weights):
        """
        Contrainte: tous les poids doivent être >= 0 (pas de vente à découvert).
        Cette contrainte est généralement appliquée via les bornes d'optimisation.
        
        Args:
            weights (numpy.ndarray): Poids des actifs
            
        Returns:
            bool: True si la contrainte est respectée, False sinon
        """
        return np.all(weights >= 0)
    
    @staticmethod
    def max_weight_constraint(weights, max_weight):
        """
        Contrainte: aucun poids ne doit dépasser max_weight.
        Cette contrainte est généralement appliquée via les bornes d'optimisation.
        
        Args:
            weights (numpy.ndarray): Poids des actifs
            max_weight (float): Poids maximum par actif
            
        Returns:
            bool: True si la contrainte est respectée, False sinon
        """
        return np.all(weights <= max_weight)
    
    @staticmethod
    def min_weight_constraint(weights, min_weight):
        """
        Contrainte: tous les poids doivent être >= min_weight.
        
        Args:
            weights (numpy.ndarray): Poids des actifs
            min_weight (float): Poids minimum par actif (pour les positions non nulles)
            
        Returns:
            bool: True si la contrainte est respectée, False sinon
        """
        # Vérifier uniquement les poids non nuls
        non_zero_weights = weights[weights > 0]
        return np.all(non_zero_weights >= min_weight)
    
    @staticmethod
    def asset_class_weight_constraint(weights, asset_classes, class_bounds):
        """
        Contrainte: le poids total de chaque classe d'actifs doit respecter des limites.
        
        Args:
            weights (numpy.ndarray): Poids des actifs
            asset_classes (dict): Dictionnaire associant chaque actif à sa classe
            class_bounds (dict): Limites min/max pour chaque classe {class: (min, max)}
            
        Returns:
            bool: True si la contrainte est respectée, False sinon
        """
        # Convertir weights en Series si nécessaire
        if isinstance(weights, np.ndarray):
            assets = list(asset_classes.keys())
            weights = pd.Series(weights, index=assets)
        
        # Vérifier les limites pour chaque classe d'actifs
        for asset_class, (min_weight, max_weight) in class_bounds.items():
            # Calculer le poids total de la classe
            class_weight = sum(weights[asset] for asset, cls in asset_classes.items() 
                               if cls == asset_class)
            
            # Vérifier les limites
            if class_weight < min_weight or class_weight > max_weight:
                return False
        
        return True
    
    @staticmethod
    def sector_exposure_constraint(weights, sector_exposures, sector_limits):
        """
        Contrainte: l'exposition à chaque secteur doit respecter des limites.
        
        Args:
            weights (numpy.ndarray): Poids des actifs
            sector_exposures (dict): Exposition de chaque actif aux secteurs {asset: {sector: exposure}}
            sector_limits (dict): Limites min/max pour chaque secteur {sector: (min, max)}
            
        Returns:
            bool: True si la contrainte est respectée, False sinon
        """
        # Convertir weights en Series si nécessaire
        if isinstance(weights, np.ndarray):
            assets = list(sector_exposures.keys())
            weights = pd.Series(weights, index=assets)
        
        # Calculer l'exposition totale pour chaque secteur
        sector_weights = {}
        for sector in sector_limits.keys():
            sector_weights[sector] = sum(weights[asset] * exposures.get(sector, 0) 
                                        for asset, exposures in sector_exposures.items())
        
        # Vérifier les limites pour chaque secteur
        for sector, (min_weight, max_weight) in sector_limits.items():
            sector_weight = sector_weights.get(sector, 0)
            if sector_weight < min_weight or sector_weight > max_weight:
                return False
        
        return True
    
    @staticmethod
    def target_return_constraint(weights, expected_returns, target_return):
        """
        Contrainte: le rendement attendu du portefeuille doit être égal à target_return.
        
        Args:
            weights (numpy.ndarray): Poids des actifs
            expected_returns (numpy.ndarray): Rendements attendus des actifs
            target_return (float): Rendement cible du portefeuille
            
        Returns:
            float: Écart par rapport à la contrainte (0 = respectée)
        """
        return np.dot(weights, expected_returns) - target_return
    
    @staticmethod
    def max_volatility_constraint(weights, cov_matrix, max_volatility):
        """
        Contrainte: la volatilité du portefeuille ne doit pas dépasser max_volatility.
        
        Args:
            weights (numpy.ndarray): Poids des actifs
            cov_matrix (numpy.ndarray): Matrice de covariance des actifs
            max_volatility (float): Volatilité maximale autorisée
            
        Returns:
            float: Écart par rapport à la contrainte (<=0 = respectée)
        """
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        return portfolio_volatility - max_volatility
    
    @staticmethod
    def turnover_constraint(weights, previous_weights, max_turnover):
        """
        Contrainte: le turnover (changement total des poids) ne doit pas dépasser max_turnover.
        
        Args:
            weights (numpy.ndarray): Nouveaux poids des actifs
            previous_weights (numpy.ndarray): Poids actuels des actifs
            max_turnover (float): Turnover maximal autorisé (0-1)
            
        Returns:
            float: Écart par rapport à la contrainte (<=0 = respectée)
        """
        turnover = np.sum(np.abs(weights - previous_weights)) / 2
        return turnover - max_turnover
    
    @staticmethod
    def tracking_error_constraint(weights, cov_matrix, benchmark_weights, max_tracking_error):
        """
        Contrainte: le tracking error par rapport à un benchmark ne doit pas dépasser max_tracking_error.
        
        Args:
            weights (numpy.ndarray): Poids des actifs
            cov_matrix (numpy.ndarray): Matrice de covariance des actifs
            benchmark_weights (numpy.ndarray): Poids du benchmark
            max_tracking_error (float): Tracking error maximal autorisé
            
        Returns:
            float: Écart par rapport à la contrainte (<=0 = respectée)
        """
        # Calculer les poids relatifs
        active_weights = weights - benchmark_weights
        
        # Calculer le tracking error
        tracking_var = np.dot(active_weights.T, np.dot(cov_matrix, active_weights))
        tracking_error = np.sqrt(tracking_var)
        
        return tracking_error - max_tracking_error
    
    @staticmethod
    def liquidity_constraint(weights, adv_values, portfolio_value, max_days_to_liquidate, participation_rate=0.1):
        """
        Contrainte: le portefeuille doit pouvoir être liquidé en max_days_to_liquidate jours.
        
        Args:
            weights (numpy.ndarray): Poids des actifs
            adv_values (numpy.ndarray): Volume quotidien moyen en unités monétaires pour chaque actif
            portfolio_value (float): Valeur totale du portefeuille
            max_days_to_liquidate (float): Nombre maximal de jours pour liquidation
            participation_rate (float): Taux de participation au volume quotidien (0-1)
            
        Returns:
            float: Écart par rapport à la contrainte (<=0 = respectée)
        """
        # Calculer la valeur monétaire de chaque position
        position_values = weights * portfolio_value
        
        # Calculer le montant quotidien liquidable pour chaque actif
        daily_liquidable = adv_values * participation_rate
        
        # Calculer le nombre de jours nécessaires pour liquidation
        days_to_liquidate = np.max(position_values / daily_liquidable)
        
        return days_to_liquidate - max_days_to_liquidate
    
    @staticmethod
    def max_drawdown_constraint(weights, returns_data, max_allowed_drawdown):
        """
        Contrainte: le drawdown maximal historique ne doit pas dépasser max_allowed_drawdown.
        
        Args:
            weights (numpy.ndarray): Poids des actifs
            returns_data (pandas.DataFrame): Rendements historiques des actifs
            max_allowed_drawdown (float): Drawdown maximal autorisé (0-1, positif)
            
        Returns:
            float: Écart par rapport à la contrainte (<=0 = respectée)
        """
        # Calculer les rendements historiques du portefeuille
        portfolio_returns = returns_data.dot(weights)
        
        # Calculer les rendements cumulés
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Calculer le drawdown à chaque période
        historical_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns / historical_max) - 1
        
        # Le drawdown maximal est le plus grand en valeur absolue
        max_drawdown = abs(drawdowns.min())
        
        return max_drawdown - max_allowed_drawdown
    
    @staticmethod
    def concentration_constraint(weights, max_concentration):
        """
        Contrainte: la concentration du portefeuille (Herfindahl-Hirschman Index) 
        ne doit pas dépasser max_concentration.
        
        Args:
            weights (numpy.ndarray): Poids des actifs
            max_concentration (float): Concentration maximale autorisée (0-1)
            
        Returns:
            float: Écart par rapport à la contrainte (<=0 = respectée)
        """
        # Calculer l'indice de concentration (HHI)
        # HHI = somme des carrés des poids
        concentration = np.sum(weights ** 2)
        
        return concentration - max_concentration

def create_long_only_bounds(n_assets):
    """
    Crée des bornes pour une optimisation sans vente à découvert.
    
    Args:
        n_assets (int): Nombre d'actifs
        
    Returns:
        tuple: Bornes pour l'optimisation
    """
    return tuple((0, 1) for _ in range(n_assets))

def create_max_weight_bounds(n_assets, max_weight):
    """
    Crée des bornes avec un poids maximum par actif.
    
    Args:
        n_assets (int): Nombre d'actifs
        max_weight (float): Poids maximum par actif
        
    Returns:
        tuple: Bornes pour l'optimisation
    """
    return tuple((0, max_weight) for _ in range(n_assets))

def create_individual_bounds(asset_bounds):
    """
    Crée des bornes personnalisées pour chaque actif.
    
    Args:
        asset_bounds (dict): Bornes pour chaque actif {asset: (min, max)}
        
    Returns:
        tuple: Bornes pour l'optimisation
    """
    return tuple(asset_bounds[asset] for asset in sorted(asset_bounds.keys()))

def create_asset_class_constraint(asset_classes, class_bounds):
    """
    Crée des contraintes pour les classes d'actifs.
    
    Args:
        asset_classes (dict): Dictionnaire associant chaque actif à sa classe
        class_bounds (dict): Limites min/max pour chaque classe {class: (min, max)}
        
    Returns:
        list: Liste de fonctions de contrainte
    """
    constraints = []
    
    for asset_class, (min_weight, max_weight) in class_bounds.items():
        # Indices des actifs de cette classe
        class_indices = [i for i, (asset, cls) in enumerate(asset_classes.items()) if cls == asset_class]
        
        # Contrainte de poids minimum pour la classe
        if min_weight > 0:
            constraints.append({
                'type': 'ineq',
                'fun': lambda w, indices=class_indices, min_w=min_weight: 
                      sum(w[i] for i in indices) - min_w
            })
        
        # Contrainte de poids maximum pour la classe
        if max_weight < 1:
            constraints.append({
                'type': 'ineq',
                'fun': lambda w, indices=class_indices, max_w=max_weight: 
                      max_w - sum(w[i] for i in indices)
            })
    
    return constraints

def create_risk_constraints(cov_matrix, max_volatility=None, target_return=None, expected_returns=None):
    """
    Crée des contraintes de risque pour l'optimisation.
    
    Args:
        cov_matrix (numpy.ndarray): Matrice de covariance des actifs
        max_volatility (float, optional): Volatilité maximale autorisée
        target_return (float, optional): Rendement cible du portefeuille
        expected_returns (numpy.ndarray, optional): Rendements attendus des actifs
        
    Returns:
        list: Liste de fonctions de contrainte
    """
    constraints = []
    
    # Contrainte de volatilité maximale
    if max_volatility is not None:
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: max_volatility**2 - np.dot(w.T, np.dot(cov_matrix, w))
        })
    
    # Contrainte de rendement cible
    if target_return is not None and expected_returns is not None:
        constraints.append({
            'type': 'eq',
            'fun': lambda w: np.dot(w, expected_returns) - target_return
        })
    
    return constraints

def main():
    """
    Fonction principale pour tester les contraintes.
    """
    # Exemple d'utilisation
    import numpy as np
    
    # Poids d'un portefeuille fictif
    weights = np.array([0.2, 0.3, 0.15, 0.35])
    expected_returns = np.array([0.05, 0.08, 0.1, 0.07])
    
    # Vérifier la contrainte de somme à 1
    sum_check = PortfolioConstraints.sum_to_one_constraint(weights)
    print(f"Contrainte de somme à 1: {sum_check} (devrait être proche de 0)")
    
    # Vérifier la contrainte de poids maximum
    max_weight_check = PortfolioConstraints.max_weight_constraint(weights, 0.4)
    print(f"Contrainte de poids maximum: {max_weight_check}")
    
    # Vérifier la contrainte de rendement cible
    target_return = 0.07
    return_check = PortfolioConstraints.target_return_constraint(weights, expected_returns, target_return)
    print(f"Contrainte de rendement cible: {return_check} (devrait être proche de 0 pour un rendement de 7%)")
    
    # Créer des bornes pour l'optimisation
    n_assets = 4
    bounds = create_max_weight_bounds(n_assets, 0.4)
    print(f"Bornes pour l'optimisation: {bounds}")
    
    logger.info("Test des contraintes terminé avec succès.")

if __name__ == "__main__":
    main()
