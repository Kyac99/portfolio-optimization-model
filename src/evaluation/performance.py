#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module pour le calcul et l'analyse des métriques de performance.
Ce module fournit des fonctions pour calculer différentes métriques
de performance pour les portefeuilles.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """
    Classe pour l'analyse de performance des portefeuilles.
    """
    
    def __init__(self, returns_data=None, benchmark_returns=None, risk_free_rate=None):
        """
        Initialise l'analyseur de performance.
        
        Args:
            returns_data (pandas.DataFrame): Rendements des stratégies
            benchmark_returns (pandas.Series): Rendements du benchmark
            risk_free_rate (pandas.Series): Taux sans risque
        """
        self.returns_data = returns_data
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate
        
        if returns_data is not None:
            logger.info(f"Initialisation de l'analyseur de performance avec {returns_data.shape[1]} stratégies")
    
    def calculate_returns_metrics(self, returns, risk_free_rate=None, periods_per_year=252):
        """
        Calcule les métriques de rendement et de risque pour une série de rendements.
        
        Args:
            returns (pandas.Series): Rendements de la stratégie
            risk_free_rate (pandas.Series): Taux sans risque
            periods_per_year (int): Nombre de périodes par an
            
        Returns:
            pandas.Series: Métriques de rendement et de risque
        """
        # Calculer le rendement total
        total_return = (1 + returns).prod() - 1
        
        # Calculer le rendement annualisé
        num_periods = len(returns)
        annual_return = (1 + total_return) ** (periods_per_year / num_periods) - 1
        
        # Calculer la volatilité annualisée
        annual_volatility = returns.std() * np.sqrt(periods_per_year)
        
        # Calculer le ratio de Sharpe
        if risk_free_rate is not None:
            # Aligner et calculer le taux sans risque moyen
            if isinstance(risk_free_rate, pd.Series):
                aligned_rf = risk_free_rate.reindex(returns.index)
                avg_rf = aligned_rf.mean() * periods_per_year
            else:
                avg_rf = risk_free_rate
            sharpe_ratio = (annual_return - avg_rf) / annual_volatility
        else:
            sharpe_ratio = annual_return / annual_volatility
        
        # Calculer les rendements positifs et négatifs
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        # Calculer le ratio de Sortino (volatilité négative seulement)
        downside_deviation = negative_returns.std() * np.sqrt(periods_per_year) if len(negative_returns) > 0 else 0
        sortino_ratio = annual_return / downside_deviation if downside_deviation > 0 else np.inf
        
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
        
        # Calculer d'autres métriques
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Calculer le nombre de périodes positives et négatives
        num_positive = len(positive_returns)
        num_negative = len(negative_returns)
        win_rate = num_positive / (num_positive + num_negative) if (num_positive + num_negative) > 0 else 0
        
        # Calculer le gain moyen et la perte moyenne
        avg_gain = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
        
        # Calculer le ratio gain/perte
        gain_loss_ratio = abs(avg_gain / avg_loss) if avg_loss < 0 else np.inf
        
        # Calculer l'Omega (rendement cible par défaut = 0)
        returns_above_target = returns[returns > 0]
        returns_below_target = returns[returns <= 0]
        omega_ratio = (returns_above_target.sum() / abs(returns_below_target.sum())) if returns_below_target.sum() < 0 else np.inf
        
        # Métriques de rendement et de risque
        metrics = {
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Annual Volatility': annual_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Max Drawdown': max_drawdown,
            'Calmar Ratio': calmar_ratio,
            'VaR 95%': var_95,
            'CVaR 95%': cvar_95,
            'VaR 99%': var_99,
            'CVaR 99%': cvar_99,
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'Win Rate': win_rate,
            'Average Gain': avg_gain,
            'Average Loss': avg_loss,
            'Gain/Loss Ratio': gain_loss_ratio,
            'Omega Ratio': omega_ratio
        }
        
        return pd.Series(metrics)
    
    def calculate_relative_metrics(self, returns, benchmark_returns, risk_free_rate=None, periods_per_year=252):
        """
        Calcule les métriques de performance relatives à un benchmark.
        
        Args:
            returns (pandas.Series): Rendements de la stratégie
            benchmark_returns (pandas.Series): Rendements du benchmark
            risk_free_rate (pandas.Series): Taux sans risque
            periods_per_year (int): Nombre de périodes par an
            
        Returns:
            pandas.Series: Métriques de performance relatives
        """
        # Aligner les séries de rendements
        aligned_returns = pd.concat([returns, benchmark_returns], axis=1).dropna()
        returns_aligned = aligned_returns.iloc[:, 0]
        benchmark_aligned = aligned_returns.iloc[:, 1]
        
        # Calculer l'alpha et le beta
        cov_matrix = np.cov(returns_aligned, benchmark_aligned)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
        
        # Calculer le rendement annualisé
        total_return = (1 + returns_aligned).prod() - 1
        total_benchmark = (1 + benchmark_aligned).prod() - 1
        num_periods = len(returns_aligned)
        annual_return = (1 + total_return) ** (periods_per_year / num_periods) - 1
        annual_benchmark = (1 + total_benchmark) ** (periods_per_year / num_periods) - 1
        
        # Calculer l'alpha annualisé selon CAPM
        if risk_free_rate is not None:
            # Aligner et calculer le taux sans risque moyen
            if isinstance(risk_free_rate, pd.Series):
                aligned_rf = risk_free_rate.reindex(returns_aligned.index)
                avg_rf = aligned_rf.mean() * periods_per_year
            else:
                avg_rf = risk_free_rate
            alpha = annual_return - (avg_rf + beta * (annual_benchmark - avg_rf))
        else:
            alpha = annual_return - (beta * annual_benchmark)
        
        # Calculer le tracking error
        tracking_diff = returns_aligned - benchmark_aligned
        tracking_error = tracking_diff.std() * np.sqrt(periods_per_year)
        
        # Calculer le ratio d'information
        information_ratio = (annual_return - annual_benchmark) / tracking_error if tracking_error > 0 else np.inf
        
        # Calculer l'up-capture et down-capture
        up_months = benchmark_aligned > 0
        down_months = benchmark_aligned < 0
        
        up_capture = (returns_aligned[up_months].mean() / benchmark_aligned[up_months].mean()) if up_months.sum() > 0 else np.nan
        down_capture = (returns_aligned[down_months].mean() / benchmark_aligned[down_months].mean()) if down_months.sum() > 0 else np.nan
        
        # Calculer le M-squared (M²)
        if risk_free_rate is not None:
            volatility_aligned = returns_aligned.std() * np.sqrt(periods_per_year)
            volatility_benchmark = benchmark_aligned.std() * np.sqrt(periods_per_year)
            m_squared = avg_rf + (annual_return - avg_rf) * volatility_benchmark / volatility_aligned - annual_benchmark
        else:
            m_squared = np.nan
        
        # Calculer le R-squared (coefficient de détermination)
        correlation = np.corrcoef(returns_aligned, benchmark_aligned)[0, 1]
        r_squared = correlation ** 2
        
        # Calculer le ratio de Treynor
        treynor_ratio = (annual_return - avg_rf) / beta if beta > 0 else np.inf
        
        # Métriques relatives
        metrics = {
            'Alpha': alpha,
            'Beta': beta,
            'R-squared': r_squared,
            'Tracking Error': tracking_error,
            'Information Ratio': information_ratio,
            'Up Capture': up_capture,
            'Down Capture': down_capture,
            'M-squared': m_squared,
            'Treynor Ratio': treynor_ratio
        }
        
        return pd.Series(metrics)
    
    def analyze_all_strategies(self, periods_per_year=252):
        """
        Analyse toutes les stratégies et calcule leurs métriques de performance.
        
        Args:
            periods_per_year (int): Nombre de périodes par an
            
        Returns:
            pandas.DataFrame: Métriques de performance pour toutes les stratégies
        """
        if self.returns_data is None:
            logger.error("Aucune donnée de rendements disponible.")
            return None
        
        logger.info("Analyse de performance pour toutes les stratégies")
        
        # Initialiser le dictionnaire pour stocker les métriques
        metrics_dict = {}
        
        # Calculer les métriques pour chaque stratégie
        for column in self.returns_data.columns:
            strategy_returns = self.returns_data[column]
            
            # Calculer les métriques de rendement et de risque
            returns_metrics = self.calculate_returns_metrics(
                strategy_returns, 
                self.risk_free_rate, 
                periods_per_year
            )
            
            # Ajouter les métriques relatives si un benchmark est disponible
            if self.benchmark_returns is not None:
                relative_metrics = self.calculate_relative_metrics(
                    strategy_returns, 
                    self.benchmark_returns, 
                    self.risk_free_rate, 
                    periods_per_year
                )
                
                # Combiner les métriques
                combined_metrics = pd.concat([returns_metrics, relative_metrics])
                metrics_dict[column] = combined_metrics
            else:
                metrics_dict[column] = returns_metrics
        
        # Créer un DataFrame avec toutes les métriques
        metrics_df = pd.DataFrame(metrics_dict)
        
        return metrics_df
    
    def plot_cumulative_returns(self, figsize=(12, 8), title='Rendements Cumulés'):
        """
        Trace les rendements cumulés de toutes les stratégies.
        
        Args:
            figsize (tuple): Dimensions de la figure
            title (str): Titre du graphique
            
        Returns:
            matplotlib.figure.Figure: Figure matplotlib
        """
        if self.returns_data is None:
            logger.error("Aucune donnée de rendements disponible.")
            return None
        
        logger.info("Traçage des rendements cumulés")
        
        # Calculer les rendements cumulés
        cumulative_returns = (1 + self.returns_data).cumprod()
        
        # Ajouter le benchmark si disponible
        if self.benchmark_returns is not None:
            benchmark_cumulative = (1 + self.benchmark_returns).cumprod()
            cumulative_returns = pd.concat([cumulative_returns, benchmark_cumulative], axis=1)
            cumulative_returns.columns = list(cumulative_returns.columns[:-1]) + ['Benchmark']
        
        # Créer la figure
        fig, ax = plt.subplots(figsize=figsize)
        
        cumulative_returns.plot(ax=ax)
        
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Valeur (base 1)', fontsize=14)
        ax.grid(True)
        ax.legend()
        
        return fig
    
    def plot_drawdowns(self, figsize=(12, 8), title='Drawdowns'):
        """
        Trace les drawdowns de toutes les stratégies.
        
        Args:
            figsize (tuple): Dimensions de la figure
            title (str): Titre du graphique
            
        Returns:
            matplotlib.figure.Figure: Figure matplotlib
        """
        if self.returns_data is None:
            logger.error("Aucune donnée de rendements disponible.")
            return None
        
        logger.info("Traçage des drawdowns")
        
        # Initialiser le DataFrame pour stocker les drawdowns
        drawdowns_df = pd.DataFrame(index=self.returns_data.index)
        
        # Calculer les drawdowns pour chaque stratégie
        for column in self.returns_data.columns:
            cumulative_returns = (1 + self.returns_data[column]).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns / rolling_max - 1)
            drawdowns_df[column] = drawdowns
        
        # Ajouter le benchmark si disponible
        if self.benchmark_returns is not None:
            benchmark_cumulative = (1 + self.benchmark_returns).cumprod()
            rolling_max = benchmark_cumulative.expanding().max()
            benchmark_drawdowns = (benchmark_cumulative / rolling_max - 1)
            drawdowns_df['Benchmark'] = benchmark_drawdowns
        
        # Créer la figure
        fig, ax = plt.subplots(figsize=figsize)
        
        drawdowns_df.plot(ax=ax)
        
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Drawdown (%)', fontsize=14)
        ax.grid(True)
        ax.legend()
        
        return fig
    
    def plot_rolling_metrics(self, window=21, metrics=['return', 'volatility', 'sharpe'],
                           figsize=(15, 12), title='Métriques glissantes'):
        """
        Trace les métriques glissantes de toutes les stratégies.
        
        Args:
            window (int): Taille de la fenêtre glissante
            metrics (list): Liste des métriques à tracer
            figsize (tuple): Dimensions de la figure
            title (str): Titre du graphique
            
        Returns:
            matplotlib.figure.Figure: Figure matplotlib
        """
        if self.returns_data is None:
            logger.error("Aucune donnée de rendements disponible.")
            return None
        
        logger.info(f"Traçage des métriques glissantes avec fenêtre de {window} jours")
        
        # Nombre de métriques à tracer
        n_metrics = len(metrics)
        
        # Créer la figure avec plusieurs sous-graphiques
        fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)
        
        # Si une seule métrique, convertir axes en liste
        if n_metrics == 1:
            axes = [axes]
        
        # Calculer et tracer les métriques glissantes
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            if metric == 'return':
                # Rendement annualisé glissant
                rolling_returns = self.returns_data.rolling(window=window).apply(
                    lambda x: (1 + x).prod() ** (252 / window) - 1
                )
                
                if self.benchmark_returns is not None:
                    benchmark_rolling = self.benchmark_returns.rolling(window=window).apply(
                        lambda x: (1 + x).prod() ** (252 / window) - 1
                    )
                    rolling_returns = pd.concat([rolling_returns, benchmark_rolling], axis=1)
                    rolling_returns.columns = list(rolling_returns.columns[:-1]) + ['Benchmark']
                
                rolling_returns.plot(ax=ax)
                ax.set_title('Rendement annualisé glissant', fontsize=14)
                ax.set_ylabel('Rendement (%)', fontsize=12)
            
            elif metric == 'volatility':
                # Volatilité annualisée glissante
                rolling_vol = self.returns_data.rolling(window=window).std() * np.sqrt(252)
                
                if self.benchmark_returns is not None:
                    benchmark_rolling = self.benchmark_returns.rolling(window=window).std() * np.sqrt(252)
                    rolling_vol = pd.concat([rolling_vol, benchmark_rolling], axis=1)
                    rolling_vol.columns = list(rolling_vol.columns[:-1]) + ['Benchmark']
                
                rolling_vol.plot(ax=ax)
                ax.set_title('Volatilité annualisée glissante', fontsize=14)
                ax.set_ylabel('Volatilité (%)', fontsize=12)
            
            elif metric == 'sharpe':
                # Ratio de Sharpe glissant
                rolling_sharpe = pd.DataFrame(index=self.returns_data.index, columns=self.returns_data.columns)
                
                for column in self.returns_data.columns:
                    rolling_returns = self.returns_data[column].rolling(window=window).apply(
                        lambda x: (1 + x).prod() ** (252 / window) - 1
                    )
                    rolling_vol = self.returns_data[column].rolling(window=window).std() * np.sqrt(252)
                    rolling_sharpe[column] = rolling_returns / rolling_vol
                
                if self.benchmark_returns is not None:
                    benchmark_returns = self.benchmark_returns.rolling(window=window).apply(
                        lambda x: (1 + x).prod() ** (252 / window) - 1
                    )
                    benchmark_vol = self.benchmark_returns.rolling(window=window).std() * np.sqrt(252)
                    benchmark_sharpe = benchmark_returns / benchmark_vol
                    rolling_sharpe = pd.concat([rolling_sharpe, benchmark_sharpe], axis=1)
                    rolling_sharpe.columns = list(rolling_sharpe.columns[:-1]) + ['Benchmark']
                
                rolling_sharpe.plot(ax=ax)
                ax.set_title('Ratio de Sharpe glissant', fontsize=14)
                ax.set_ylabel('Sharpe', fontsize=12)
            
            else:
                logger.warning(f"Métrique {metric} non reconnue.")
            
            ax.grid(True)
            ax.legend()
        
        plt.tight_layout()
        fig.suptitle(title, fontsize=16, y=1.02)
        
        return fig
    
    def plot_monthly_returns_heatmap(self, strategy=None, cmap='coolwarm', figsize=(12, 8),
                                   title='Rendements mensuels'):
        """
        Trace une heatmap des rendements mensuels d'une stratégie.
        
        Args:
            strategy (str): Nom de la stratégie (si None, utilise la première)
            cmap (str): Colormap matplotlib
            figsize (tuple): Dimensions de la figure
            title (str): Titre du graphique
            
        Returns:
            matplotlib.figure.Figure: Figure matplotlib
        """
        if self.returns_data is None:
            logger.error("Aucune donnée de rendements disponible.")
            return None
        
        # Sélectionner la stratégie
        if strategy is None:
            strategy = self.returns_data.columns[0]
        
        if strategy not in self.returns_data.columns:
            logger.error(f"Stratégie {strategy} non trouvée.")
            return None
        
        logger.info(f"Traçage de la heatmap des rendements mensuels pour {strategy}")
        
        # Extraire les rendements de la stratégie
        returns = self.returns_data[strategy]
        
        # Convertir en rendements mensuels
        returns.index = pd.to_datetime(returns.index)
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Créer un DataFrame avec années en lignes et mois en colonnes
        monthly_returns.index = monthly_returns.index.to_period('M')
        monthly_pivot = monthly_returns.to_timestamp().pivot_table(
            index=monthly_returns.index.year,
            columns=monthly_returns.index.month,
            values=0
        )
        
        # Renommer les colonnes avec les noms des mois
        month_names = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
        monthly_pivot.columns = [month_names[i-1] for i in monthly_pivot.columns]
        
        # Créer la figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Créer la heatmap
        sns.heatmap(monthly_pivot, cmap=cmap, annot=True, fmt='.2%', ax=ax)
        
        ax.set_title(f"{title} - {strategy}", fontsize=16)
        ax.set_ylabel('Année', fontsize=14)
        
        return fig
    
    def plot_correlation_matrix(self, figsize=(10, 8), title='Matrice de corrélation'):
        """
        Trace la matrice de corrélation entre les stratégies.
        
        Args:
            figsize (tuple): Dimensions de la figure
            title (str): Titre du graphique
            
        Returns:
            matplotlib.figure.Figure: Figure matplotlib
        """
        if self.returns_data is None:
            logger.error("Aucune donnée de rendements disponible.")
            return None
        
        logger.info("Traçage de la matrice de corrélation")
        
        # Calculer la matrice de corrélation
        corr_matrix = self.returns_data.corr()
        
        # Ajouter le benchmark si disponible
        if self.benchmark_returns is not None:
            all_returns = pd.concat([self.returns_data, self.benchmark_returns], axis=1)
            all_returns.columns = list(all_returns.columns[:-1]) + ['Benchmark']
            corr_matrix = all_returns.corr()
        
        # Créer la figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Créer la heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                   mask=mask, ax=ax, fmt='.2f')
        
        ax.set_title(title, fontsize=16)
        
        return fig

def main():
    """
    Fonction principale pour tester l'analyseur de performance.
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
    
    # Simuler les rendements historiques des portefeuilles
    portfolio_returns = pd.DataFrame(index=returns_data.index)
    portfolio_returns['Max Sharpe'] = returns_data.dot(max_sharpe_portfolio['weights'])
    portfolio_returns['Min Volatility'] = returns_data.dot(min_vol_portfolio['weights'])
    portfolio_returns['Equal Weight'] = returns_data.dot(equal_weights)
    
    # Créer le benchmark (S&P 500 fictif)
    benchmark = returns_data.iloc[:, 0]  # Utiliser la première colonne comme benchmark fictif
    
    # Créer l'analyseur de performance
    analyzer = PerformanceAnalyzer(portfolio_returns, benchmark, risk_free_rate)
    
    # Analyser toutes les stratégies
    metrics = analyzer.analyze_all_strategies()
    
    # Afficher les métriques
    print("\nMétriques de performance:")
    print(metrics.T)
    
    # Tracer les rendements cumulés
    fig1 = analyzer.plot_cumulative_returns()
    fig1.savefig("../../results/figures/performance_cumulative_returns.png")
    
    # Tracer les drawdowns
    fig2 = analyzer.plot_drawdowns()
    fig2.savefig("../../results/figures/performance_drawdowns.png")
    
    # Tracer les métriques glissantes
    fig3 = analyzer.plot_rolling_metrics()
    fig3.savefig("../../results/figures/performance_rolling_metrics.png")
    
    # Tracer la heatmap des rendements mensuels
    fig4 = analyzer.plot_monthly_returns_heatmap()
    fig4.savefig("../../results/figures/performance_monthly_heatmap.png")
    
    # Tracer la matrice de corrélation
    fig5 = analyzer.plot_correlation_matrix()
    fig5.savefig("../../results/figures/performance_correlation_matrix.png")
    
    logger.info("Test de l'analyseur de performance terminé avec succès.")

if __name__ == "__main__":
    main()
