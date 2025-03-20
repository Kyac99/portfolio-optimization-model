#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module pour le traitement et la préparation des données financières.
Ce module fournit des fonctions pour nettoyer, transformer et préparer
les données brutes pour l'optimisation de portefeuille.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ensure_directory_exists(directory):
    """
    Crée le répertoire s'il n'existe pas déjà.
    
    Args:
        directory (str): Chemin du répertoire à créer
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Répertoire créé: {directory}")

def load_data(file_path):
    """
    Charge les données à partir d'un fichier pickle.
    
    Args:
        file_path (str): Chemin vers le fichier pickle
        
    Returns:
        pandas.DataFrame ou dict: Données chargées
    """
    try:
        logger.info(f"Chargement des données depuis {file_path}")
        return pd.read_pickle(file_path)
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {e}")
        return None

def clean_returns_data(returns_df, threshold=0.05):
    """
    Nettoie les données de rendements en supprimant les valeurs aberrantes
    et en gérant les valeurs manquantes.
    
    Args:
        returns_df (pandas.DataFrame): DataFrame des rendements
        threshold (float): Seuil pour identifier les valeurs aberrantes
        
    Returns:
        pandas.DataFrame: DataFrame des rendements nettoyé
    """
    logger.info("Nettoyage des données de rendements")
    
    # Copie des données pour éviter de modifier l'original
    clean_returns = returns_df.copy()
    
    # Gestion des valeurs manquantes
    missing_pct = clean_returns.isnull().mean() * 100
    logger.info(f"Pourcentage de valeurs manquantes par actif: \n{missing_pct}")
    
    # Suppression des actifs avec trop de valeurs manquantes (plus de 10%)
    assets_to_drop = missing_pct[missing_pct > 10].index.tolist()
    if assets_to_drop:
        logger.warning(f"Suppression des actifs avec > 10% de valeurs manquantes: {assets_to_drop}")
        clean_returns = clean_returns.drop(columns=assets_to_drop)
    
    # Remplissage des valeurs manquantes restantes avec la moyenne
    clean_returns = clean_returns.fillna(clean_returns.mean())
    
    # Détection et traitement des valeurs aberrantes
    z_scores = np.abs(stats.zscore(clean_returns))
    outliers = (z_scores > 3).any(axis=1)
    
    logger.info(f"Nombre de jours avec des valeurs aberrantes: {outliers.sum()}")
    
    # Option 1: Remplacement des valeurs aberrantes par la médiane
    for col in clean_returns.columns:
        col_z_scores = np.abs(stats.zscore(clean_returns[col]))
        outliers_idx = col_z_scores > 3
        if outliers_idx.any():
            clean_returns.loc[outliers_idx, col] = clean_returns[col].median()
    
    # Option 2 (alternative): Suppression des jours avec des valeurs aberrantes
    # clean_returns = clean_returns[~outliers]
    
    logger.info(f"Nettoyage terminé. Shape des données: {clean_returns.shape}")
    
    return clean_returns

def calculate_return_statistics(returns_df, risk_free_rate=None, periods_per_year=252):
    """
    Calcule les statistiques de rendement pour chaque actif.
    
    Args:
        returns_df (pandas.DataFrame): DataFrame des rendements
        risk_free_rate (pandas.Series, optional): Taux sans risque
        periods_per_year (int): Nombre de périodes par an (252 pour données quotidiennes)
        
    Returns:
        pandas.DataFrame: DataFrame des statistiques de rendement
    """
    logger.info("Calcul des statistiques de rendement")
    
    # Calculer le rendement moyen annualisé
    mean_returns = returns_df.mean() * periods_per_year
    
    # Calculer la volatilité annualisée
    volatility = returns_df.std() * np.sqrt(periods_per_year)
    
    # Calculer le ratio de Sharpe
    if risk_free_rate is not None:
        # Calculer le taux sans risque annualisé moyen
        mean_rf = risk_free_rate.mean() * periods_per_year
        sharpe_ratio = (mean_returns - mean_rf) / volatility
    else:
        sharpe_ratio = mean_returns / volatility
    
    # Calculer le skewness et kurtosis
    skewness = returns_df.skew()
    kurtosis = returns_df.kurtosis()
    
    # Calculer le maximum drawdown
    cum_returns = (1 + returns_df).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdown = (cum_returns / rolling_max - 1)
    max_drawdown = drawdown.min()
    
    # Créer un DataFrame avec les statistiques
    stats_df = pd.DataFrame({
        'Rendement annualisé': mean_returns,
        'Volatilité annualisée': volatility,
        'Ratio de Sharpe': sharpe_ratio,
        'Skewness': skewness,
        'Kurtosis': kurtosis,
        'Maximum Drawdown': max_drawdown
    })
    
    logger.info("Calcul des statistiques terminé")
    
    return stats_df

def calculate_correlation_matrix(returns_df):
    """
    Calcule la matrice de corrélation entre les actifs.
    
    Args:
        returns_df (pandas.DataFrame): DataFrame des rendements
        
    Returns:
        pandas.DataFrame: Matrice de corrélation
    """
    logger.info("Calcul de la matrice de corrélation")
    return returns_df.corr()

def calculate_covariance_matrix(returns_df, periods_per_year=252):
    """
    Calcule la matrice de covariance annualisée entre les actifs.
    
    Args:
        returns_df (pandas.DataFrame): DataFrame des rendements
        periods_per_year (int): Nombre de périodes par an (252 pour données quotidiennes)
        
    Returns:
        pandas.DataFrame: Matrice de covariance annualisée
    """
    logger.info("Calcul de la matrice de covariance annualisée")
    return returns_df.cov() * periods_per_year

def prepare_data_for_optimization(returns_df, risk_free_rate=None, save_path=None):
    """
    Prépare les données pour l'optimisation de portefeuille.
    
    Args:
        returns_df (pandas.DataFrame): DataFrame des rendements
        risk_free_rate (pandas.Series, optional): Taux sans risque
        save_path (str, optional): Chemin pour sauvegarder les données
        
    Returns:
        dict: Dictionnaire contenant les données préparées
    """
    logger.info("Préparation des données pour l'optimisation")
    
    # Nettoyer les données de rendements
    clean_returns = clean_returns_data(returns_df)
    
    # Calculer les statistiques de rendement
    return_stats = calculate_return_statistics(clean_returns, risk_free_rate)
    
    # Calculer la matrice de corrélation
    corr_matrix = calculate_correlation_matrix(clean_returns)
    
    # Calculer la matrice de covariance annualisée
    cov_matrix = calculate_covariance_matrix(clean_returns)
    
    # Préparer les données pour l'optimisation
    optimization_data = {
        'clean_returns': clean_returns,
        'return_stats': return_stats,
        'correlation_matrix': corr_matrix,
        'covariance_matrix': cov_matrix,
        'expected_returns': return_stats['Rendement annualisé'],
        'risk': return_stats['Volatilité annualisée']
    }
    
    # Sauvegarder les données si un chemin est fourni
    if save_path:
        ensure_directory_exists(os.path.dirname(save_path))
        for key, data in optimization_data.items():
            data.to_pickle(f"{save_path}_{key}.pkl")
        logger.info(f"Données préparées sauvegardées dans {save_path}")
    
    return optimization_data

def combine_stock_bond_data(stock_returns, bond_returns, save_path=None):
    """
    Combine les données de rendements d'actions et d'obligations.
    
    Args:
        stock_returns (pandas.DataFrame): DataFrame des rendements d'actions
        bond_returns (pandas.DataFrame): DataFrame des rendements d'obligations
        save_path (str, optional): Chemin pour sauvegarder les données
        
    Returns:
        pandas.DataFrame: DataFrame combiné des rendements
    """
    logger.info("Combinaison des données d'actions et d'obligations")
    
    # Vérifier que les index (dates) sont du même type
    stock_returns.index = pd.to_datetime(stock_returns.index)
    bond_returns.index = pd.to_datetime(bond_returns.index)
    
    # Aligner les données sur les mêmes dates
    combined_returns = pd.concat([stock_returns, bond_returns], axis=1)
    
    # Gérer les valeurs manquantes (le cas échéant)
    combined_returns = combined_returns.dropna()
    
    # Sauvegarder les données si un chemin est fourni
    if save_path:
        ensure_directory_exists(os.path.dirname(save_path))
        combined_returns.to_pickle(f"{save_path}_combined_returns.pkl")
        logger.info(f"Données combinées sauvegardées dans {save_path}")
    
    return combined_returns

def main():
    """
    Fonction principale pour exécuter le traitement et la préparation des données
    """
    # Exemple d'utilisation
    input_dir = "../../data/raw/"
    output_dir = "../../data/processed/"
    ensure_directory_exists(output_dir)
    
    # Charger les données brutes
    stock_returns = load_data(input_dir + "stocks_returns.pkl")
    bond_returns = load_data(input_dir + "bonds_returns.pkl")
    risk_free_rate = load_data(input_dir + "risk_free_rate.pkl")
    
    if stock_returns is None or bond_returns is None:
        logger.error("Impossible de charger les données. Arrêt du traitement.")
        return
    
    # Combiner les données d'actions et d'obligations
    combined_returns = combine_stock_bond_data(stock_returns, bond_returns, 
                                              save_path=output_dir + "combined")
    
    # Préparer les données pour l'optimisation
    optimization_data = prepare_data_for_optimization(combined_returns, risk_free_rate, 
                                                    save_path=output_dir + "optimization")
    
    logger.info("Traitement et préparation des données terminés avec succès.")

if __name__ == "__main__":
    main()
