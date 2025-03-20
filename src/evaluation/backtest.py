#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module pour le backtesting de stratégies d'allocation d'actifs.
Ce module fournit des fonctions pour tester des stratégies d'allocation
sur des données historiques.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import sys
import os

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PortfolioBacktester:
    """
    Classe pour le backtesting de stratégies d'allocation d'actifs.
    """
    
    def __init__(self, returns_data, risk_free_rate=None):
        """
        Initialise le backtester.
        
        Args:
            returns_data (pandas.DataFrame): Rendements historiques des actifs
            risk_free_rate (pandas.Series, optional): Taux sans risque historique
        """
        self.returns_data = returns_data
        self.risk_free_rate = risk_free_rate
        self.assets = list(returns_data.columns)
        self.dates = returns_data.index
        
        logger.info(f"Initialisation du backtester avec {len(self.assets)} actifs sur {len(self.dates)} périodes")
    
    def backtest_fixed_weights(self, weights, start_date=None, end_date=None):
        """
        Teste une allocation fixe de poids sur des données historiques.
        
        Args:
            weights (pandas.Series): Poids fixes des actifs
            start_date (str or datetime, optional): Date de début du backtest
            end_date (str or datetime, optional): Date de fin du backtest
            
        Returns:
            dict: Résultats du backtest
        """
        logger.info("Backtesting avec des poids fixes")
        
        # Vérifier que les poids sont valides
        if not isinstance(weights, pd.Series):
            weights = pd.Series(weights, index=self.assets)
        
        # Filtrer les données selon les dates
        returns = self.returns_data.copy()
        if start_date is not None:
            returns = returns[returns.index >= start_date]
        if end_date is not None:
            returns = returns[returns.index <= end_date]
        
        # Calculer les rendements du portefeuille
        portfolio_returns = returns.dot(weights)
        
        # Calculer les rendements cumulés
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Calculer les métriques de performance
        performance = self.calculate_performance_metrics(portfolio_returns, self.risk_free_rate)
        
        # Résultats du backtest
        results = {
            'weights': weights,
            'returns': portfolio_returns,
            'cumulative_returns': cumulative_returns,
            'performance': performance
        }
        
        logger.info(f"Backtest terminé. Rendement total: {(cumulative_returns.iloc[-1] - 1) * 100:.2f}%")
        
        return results
    
    def backtest_rebalancing(self, weights_func, rebalancing_freq='M', lookback_periods=252,
                           start_date=None, end_date=None):
        """
        Teste une stratégie avec rééquilibrage périodique sur des données historiques.
        
        Args:
            weights_func (function): Fonction qui calcule les poids à partir des données historiques
            rebalancing_freq (str): Fréquence de rééquilibrage ('D', 'W', 'M', 'Q', 'Y')
            lookback_periods (int): Nombre de périodes pour le calcul des poids
            start_date (str or datetime, optional): Date de début du backtest
            end_date (str or datetime, optional): Date de fin du backtest
            
        Returns:
            dict: Résultats du backtest
        """
        logger.info(f"Backtesting avec rééquilibrage {rebalancing_freq}")
        
        # Filtrer les données selon les dates
        returns = self.returns_data.copy()
        if start_date is not None:
            returns = returns[returns.index >= start_date]
        if end_date is not None:
            returns = returns[returns.index <= end_date]
        
        # Ajouter une colonne pour le groupe de rééquilibrage
        returns['rebalance_group'] = returns.index.to_period(rebalancing_freq)
        
        # Identifier les dates de rééquilibrage (dernière date de chaque période)
        rebalance_dates = returns.groupby('rebalance_group').apply(lambda x: x.index[-1])
        
        # Initialiser les résultats
        all_weights = pd.DataFrame(index=returns.index, columns=self.assets)
        portfolio_returns = pd.Series(index=returns.index)
        
        # Calculer les poids et les rendements pour chaque période
        for i, rebalance_date in enumerate(rebalance_dates):
            # Déterminer la période d'historique pour le calcul des poids
            if i == 0:
                # Premier rééquilibrage: utiliser les données jusqu'à la date actuelle
                hist_end = rebalance_date
                hist_start = max(returns.index[0], hist_end - pd.Timedelta(days=lookback_periods))
                hist_data = returns[(returns.index >= hist_start) & (returns.index <= hist_end)]
                
                # Calculer les poids initiaux
                weights = weights_func(hist_data)
                
                # Définir la période d'application des poids
                next_rebalance = rebalance_dates[i+1] if i+1 < len(rebalance_dates) else returns.index[-1]
                application_period = returns[(returns.index > rebalance_date) & (returns.index <= next_rebalance)]
                
                # Stocker les poids et calculer les rendements
                for date in application_period.index:
                    all_weights.loc[date] = weights
                    portfolio_returns.loc[date] = application_period.loc[date, self.assets].dot(weights)
            else:
                # Rééquilibrages suivants
                hist_end = rebalance_date
                hist_start = max(returns.index[0], hist_end - pd.Timedelta(days=lookback_periods))
                hist_data = returns[(returns.index >= hist_start) & (returns.index <= hist_end)]
                
                # Calculer les poids pour cette période
                weights = weights_func(hist_data)
                
                # Définir la période d'application des poids
                next_rebalance = rebalance_dates[i+1] if i+1 < len(rebalance_dates) else returns.index[-1]
                application_period = returns[(returns.index > rebalance_date) & (returns.index <= next_rebalance)]
                
                # Stocker les poids et calculer les rendements
                for date in application_period.index:
                    all_weights.loc[date] = weights
                    portfolio_returns.loc[date] = application_period.loc[date, self.assets].dot(weights)
        
        # Supprimer les valeurs NA (première période)
        portfolio_returns = portfolio_returns.dropna()
        
        # Calculer les rendements cumulés
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Calculer les métriques de performance
        performance = self.calculate_performance_metrics(portfolio_returns, self.risk_free_rate)
        
        # Résultats du backtest
        results = {
            'weights': all_weights,
            'returns': portfolio_returns,
            'cumulative_returns': cumulative_returns,
            'performance': performance,
            'rebalance_dates': rebalance_dates
        }
        
        logger.info(f"Backtest terminé. Rendement total: {(cumulative_returns.iloc[-1] - 1) * 100:.2f}%")
        
        return results
    
    def calculate_performance_metrics(self, returns, risk_free_rate=None, periods_per_year=252):
        """
        Calcule les métriques de performance pour une série de rendements.
        
        Args:
            returns (pandas.Series): Rendements du portefeuille
            risk_free_rate (pandas.Series, optional): Taux sans risque
            periods_per_year (int): Nombre de périodes par an
            
        Returns:
            pandas.Series: Métriques de performance
        """
        # Calculer le rendement annualisé
        total_return = (1 + returns).prod() - 1
        num_periods = len(returns)
        annual_return = (1 + total_return) ** (periods_per_year / num_periods) - 1
        
        # Calculer la volatilité annualisée
        annual_volatility = returns.std() * np.sqrt(periods_per_year)
        
        # Calculer le ratio de Sharpe
        if risk_free_rate is not None:
            # Utiliser le taux sans risque moyen
            avg_rf = risk_free_rate.reindex(returns.index).mean() * periods_per_year
            sharpe_ratio = (annual_return - avg_rf) / annual_volatility
        else:
            sharpe_ratio = annual_return / annual_volatility
        
        # Calculer le ratio de Sortino (volatilité négative seulement)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_deviation = negative_returns.std() * np.sqrt(periods_per_year)
            sortino_ratio = annual_return / downside_deviation
        else:
            sortino_ratio = np.inf
        
        # Calculer le maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns / rolling_max - 1)
        max_drawdown = drawdowns.min()
        
        # Calculer la Value at Risk (VaR)
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)
        
        # Calculer l'Expected Shortfall (Conditional VaR)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Calculer le ratio de Calmar
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else np.inf
        
        # Calculer le ratio d'information (par rapport à un benchmark)
        # Note: Pas calculé ici car nécessite un benchmark
        
        # Métriques de performance
        metrics = {
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Annual Volatility': annual_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Max Drawdown': max_drawdown,
            'VaR 95%': var_95,
            'CVaR 95%': cvar_95,
            'VaR 99%': var_99,
            'CVaR 99%': cvar_99,
            'Calmar Ratio': calmar_ratio
        }
        
        return pd.Series(metrics)
    
    def compare_strategies(self, strategies, start_date=None, end_date=None):
        """
        Compare plusieurs stratégies d'allocation.
        
        Args:
            strategies (dict): Dictionnaire de stratégies {nom: weights}
            start_date (str or datetime, optional): Date de début du backtest
            end_date (str or datetime, optional): Date de fin du backtest
            
        Returns:
            dict: Résultats des comparaisons
        """
        logger.info(f"Comparaison de {len(strategies)} stratégies")
        
        # Initialiser les résultats
        results = {}
        
        # Tester chaque stratégie
        for name, weights in strategies.items():
            results[name] = self.backtest_fixed_weights(weights, start_date, end_date)
        
        # Comparer les performances
        performance_comparison = pd.DataFrame({
            name: results[name]['performance'] for name in strategies
        })
        
        # Comparer les rendements cumulés
        cumulative_returns = pd.DataFrame({
            name: results[name]['cumulative_returns'] for name in strategies
        })
        
        comparison = {
            'individual_results': results,
            'performance_comparison': performance_comparison,
            'cumulative_returns': cumulative_returns
        }
        
        logger.info("Comparaison terminée")
        
        return comparison
    
    def plot_cumulative_returns(self, cumulative_returns, title='Rendements Cumulés', 
                              figsize=(12, 8)):
        """
        Trace les rendements cumulés des stratégies.
        
        Args:
            cumulative_returns (pandas.DataFrame): Rendements cumulés des stratégies
            title (str): Titre du graphique
            figsize (tuple): Dimensions de la figure
            
        Returns:
            matplotlib.figure.Figure: Figure matplotlib
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        cumulative_returns.plot(ax=ax)
        
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Rendement Cumulé', fontsize=14)
        ax.grid(True)
        ax.legend()
        
        return fig
    
    def plot_drawdowns(self, returns, title='Drawdowns', figsize=(12, 8)):
        """
        Trace les drawdowns des stratégies.
        
        Args:
            returns (pandas.DataFrame): Rendements des stratégies
            title (str): Titre du graphique
            figsize (tuple): Dimensions de la figure
            
        Returns:
            matplotlib.figure.Figure: Figure matplotlib
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for column in returns.columns:
            # Calculer les drawdowns
            cumulative_returns = (1 + returns[column]).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns / rolling_max - 1)
            
            # Tracer les drawdowns
            drawdowns.plot(ax=ax, label=column)
        
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Drawdown', fontsize=14)
        ax.grid(True)
        ax.legend()
        
        return fig
    
    def stress_test(self, weights, crisis_periods, benchmark=None):
        """
        Effectue un stress test sur différentes périodes de crise.
        
        Args:
            weights (pandas.Series): Poids des actifs
            crisis_periods (dict): Périodes de crise {nom: (start_date, end_date)}
            benchmark (pandas.Series, optional): Rendements d'un benchmark
            
        Returns:
            pandas.DataFrame: Résultats du stress test
        """
        logger.info(f"Stress test sur {len(crisis_periods)} périodes de crise")
        
        # Initialiser les résultats
        results = {}
        
        # Tester sur chaque période de crise
        for crisis_name, (start_date, end_date) in crisis_periods.items():
            # Backtest sur la période de crise
            crisis_results = self.backtest_fixed_weights(weights, start_date, end_date)
            
            # Calculer les métriques pour le benchmark si fourni
            if benchmark is not None:
                benchmark_returns = benchmark[(benchmark.index >= start_date) & (benchmark.index <= end_date)]
                benchmark_metrics = self.calculate_performance_metrics(benchmark_returns, self.risk_free_rate)
                
                # Ajouter la surperformance
                crisis_results['performance']['Excess Return'] = crisis_results['performance']['Total Return'] - benchmark_metrics['Total Return']
            
            results[crisis_name] = crisis_results['performance']
        
        # Compiler les résultats
        stress_results = pd.DataFrame(results)
        
        logger.info("Stress test terminé")
        
        return stress_results
    
    def monte_carlo_simulation(self, weights, n_simulations=1000, horizon=252, 
                             confidence_level=0.95):
        """
        Effectue une simulation Monte Carlo pour estimer la distribution des rendements futurs.
        
        Args:
            weights (pandas.Series): Poids des actifs
            n_simulations (int): Nombre de simulations
            horizon (int): Horizon de prévision en jours
            confidence_level (float): Niveau de confiance pour les intervalles
            
        Returns:
            dict: Résultats de la simulation
        """
        logger.info(f"Simulation Monte Carlo avec {n_simulations} simulations sur un horizon de {horizon} jours")
        
        # Calculer les rendements moyens et la matrice de covariance
        mean_returns = self.returns_data.mean()
        cov_matrix = self.returns_data.cov()
        
        # Initialiser les résultats
        final_values = np.zeros(n_simulations)
        trajectories = np.zeros((horizon, n_simulations))
        
        # Générer les simulations
        for i in range(n_simulations):
            # Générer des rendements aléatoires
            returns = np.random.multivariate_normal(mean_returns, cov_matrix, horizon)
            
            # Calculer les rendements du portefeuille
            portfolio_returns = returns.dot(weights)
            
            # Calculer la valeur cumulée
            cumulative_returns = (1 + portfolio_returns).cumprod()
            
            # Stocker les résultats
            trajectories[:, i] = cumulative_returns
            final_values[i] = cumulative_returns[-1]
        
        # Calculer les statistiques
        mean_final = np.mean(final_values)
        median_final = np.median(final_values)
        std_final = np.std(final_values)
        
        # Calculer les intervalles de confiance
        alpha = (1 - confidence_level) / 2
        lower_bound = np.percentile(final_values, alpha * 100)
        upper_bound = np.percentile(final_values, (1 - alpha) * 100)
        
        # Calculer la Value at Risk
        var_95 = np.percentile(final_values, 5) - 1
        
        # Résultats de la simulation
        results = {
            'trajectories': trajectories,
            'final_values': final_values,
            'statistics': {
                'mean': mean_final,
                'median': median_final,
                'std': std_final,
                'confidence_interval': (lower_bound, upper_bound),
                'var_95': var_95
            }
        }
        
        logger.info("Simulation Monte Carlo terminée")
        
        return results
    
    def plot_monte_carlo(self, mc_results, title='Simulation Monte Carlo', figsize=(12, 8)):
        """
        Trace les résultats d'une simulation Monte Carlo.
        
        Args:
            mc_results (dict): Résultats de la simulation Monte Carlo
            title (str): Titre du graphique
            figsize (tuple): Dimensions de la figure
            
        Returns:
            matplotlib.figure.Figure: Figure matplotlib
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Tracer les trajectoires (échantillon aléatoire pour éviter d'encombrer le graphique)
        n_trajectories = mc_results['trajectories'].shape[1]
        sample_size = min(100, n_trajectories)
        sample_indices = np.random.choice(n_trajectories, sample_size, replace=False)
        
        for i in sample_indices:
            ax.plot(mc_results['trajectories'][:, i], color='blue', alpha=0.1)
        
        # Tracer la trajectoire moyenne
        mean_trajectory = np.mean(mc_results['trajectories'], axis=1)
        ax.plot(mean_trajectory, color='red', linewidth=2, label='Moyenne')
        
        # Tracer les intervalles de confiance
        lower_bound = np.percentile(mc_results['trajectories'], 2.5, axis=1)
        upper_bound = np.percentile(mc_results['trajectories'], 97.5, axis=1)
        
        ax.fill_between(range(len(mean_trajectory)), lower_bound, upper_bound, 
                       color='red', alpha=0.2, label='Intervalle de confiance 95%')
        
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Jours', fontsize=14)
        ax.set_ylabel('Valeur du portefeuille', fontsize=14)
        ax.grid(True)
        ax.legend()
        
        return fig

def main():
    """
    Fonction principale pour tester le backtester.
    """
    # Exemple d'utilisation
    import os
    import sys
    sys.path.append(os.path.abspath("../data"))
    sys.path.append(os.path.abspath("../models"))
    from data_processing import load_data
    from markowitz import MarkowitzOptimizer
    
    # Charger les données
    input_dir = "../../data/processed/"
    returns_data = load_data(input_dir + "optimization_clean_returns.pkl")
    expected_returns = load_data(input_dir + "optimization_expected_returns.pkl")
    cov_matrix = load_data(input_dir + "optimization_covariance_matrix.pkl")
    risk_free_rate = load_data(input_dir + "risk_free_rate.pkl")
    
    if returns_data is None or expected_returns is None or cov_matrix is None:
        logger.error("Impossible de charger les données. Arrêt du traitement.")
        return
    
    # Créer l'optimiseur
    optimizer = MarkowitzOptimizer(expected_returns, cov_matrix, 0.02)
    
    # Optimiser différents portefeuilles
    max_sharpe_portfolio = optimizer.optimize_sharpe_ratio()
    min_vol_portfolio = optimizer.optimize_min_volatility()
    
    # Créer un portefeuille équipondéré
    equal_weights = pd.Series(1/len(expected_returns), index=expected_returns.index)
    
    # Créer le backtester
    backtester = PortfolioBacktester(returns_data, risk_free_rate)
    
    # Comparer les stratégies
    strategies = {
        'Max Sharpe': max_sharpe_portfolio['weights'],
        'Min Volatility': min_vol_portfolio['weights'],
        'Equal Weight': equal_weights
    }
    
    comparison = backtester.compare_strategies(strategies)
    
    # Afficher les résultats
    print("\nComparaison des performances:")
    print(comparison['performance_comparison'].T)
    
    # Tracer les rendements cumulés
    fig1 = backtester.plot_cumulative_returns(comparison['cumulative_returns'])
    fig1.savefig("../../results/figures/backtest_comparison.png")
    
    # Tracer les drawdowns
    returns_df = pd.DataFrame({
        name: results['returns'] for name, results in comparison['individual_results'].items()
    })
    fig2 = backtester.plot_drawdowns(returns_df)
    fig2.savefig("../../results/figures/backtest_drawdowns.png")
    
    # Effectuer une simulation Monte Carlo
    mc_results = backtester.monte_carlo_simulation(max_sharpe_portfolio['weights'])
    fig3 = backtester.plot_monte_carlo(mc_results)
    fig3.savefig("../../results/figures/monte_carlo_simulation.png")
    
    logger.info("Test du backtester terminé avec succès.")

if __name__ == "__main__":
    main()
