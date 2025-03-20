#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module pour la collecte de données financières.
Ce module fournit des fonctions pour télécharger des données historiques
d'actions et d'obligations à partir de sources en ligne.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
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

def download_stock_data(tickers, start_date, end_date, save_path=None):
    """
    Télécharge les données historiques de cours pour une liste de tickers d'actions.
    
    Args:
        tickers (list): Liste des symboles d'actions à télécharger
        start_date (str): Date de début au format 'YYYY-MM-DD'
        end_date (str): Date de fin au format 'YYYY-MM-DD'
        save_path (str, optional): Chemin pour sauvegarder les données
        
    Returns:
        dict: Dictionnaire contenant les prix et les rendements
    """
    logger.info(f"Téléchargement des données pour {len(tickers)} actions de {start_date} à {end_date}")
    
    try:
        data = yf.download(tickers, start=start_date, end=end_date, 
                          group_by='ticker', auto_adjust=True)
        
        # Si un seul ticker est fourni, réorganiser les données
        if len(tickers) == 1:
            data.columns = pd.MultiIndex.from_product([tickers, data.columns])
            
        # Calculer les rendements journaliers
        returns_data = {}
        for ticker in tickers:
            try:
                # Sélectionner les prix de clôture et calculer les rendements
                returns_data[ticker] = data[ticker]['Close'].pct_change().dropna()
            except Exception as e:
                logger.warning(f"Erreur lors du calcul des rendements pour {ticker}: {e}")
        
        returns_df = pd.DataFrame(returns_data)
        
        # Sauvegarder les données si un chemin est fourni
        if save_path:
            ensure_directory_exists(os.path.dirname(save_path))
            data.to_pickle(save_path + "_prices.pkl")
            returns_df.to_pickle(save_path + "_returns.pkl")
            logger.info(f"Données sauvegardées dans {save_path}")
        
        return {
            'prices': data,
            'returns': returns_df
        }
    
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement des données: {e}")
        return None

def download_bond_data(tickers, start_date, end_date, save_path=None):
    """
    Télécharge les données historiques d'ETFs obligataires ou d'indices obligataires
    
    Args:
        tickers (list): Liste des symboles d'ETFs obligataires (e.g., 'AGG', 'BND', 'TLT')
        start_date (str): Date de début au format 'YYYY-MM-DD'
        end_date (str): Date de fin au format 'YYYY-MM-DD'
        save_path (str, optional): Chemin pour sauvegarder les données
        
    Returns:
        dict: Dictionnaire contenant les prix et les rendements
    """
    logger.info(f"Téléchargement des données obligataires pour {len(tickers)} ETFs de {start_date} à {end_date}")
    
    # Pour les obligations, nous utilisons des ETFs ou des indices comme proxy
    return download_stock_data(tickers, start_date, end_date, save_path)

def download_benchmark_data(benchmark_ticker, start_date, end_date, save_path=None):
    """
    Télécharge les données historiques pour un indice de référence (benchmark)
    
    Args:
        benchmark_ticker (str): Symbole de l'indice de référence (e.g., '^GSPC' pour S&P 500)
        start_date (str): Date de début au format 'YYYY-MM-DD'
        end_date (str): Date de fin au format 'YYYY-MM-DD'
        save_path (str, optional): Chemin pour sauvegarder les données
        
    Returns:
        pandas.DataFrame: Données historiques du benchmark
    """
    logger.info(f"Téléchargement des données pour l'indice {benchmark_ticker} de {start_date} à {end_date}")
    
    try:
        benchmark_data = yf.download(benchmark_ticker, start=start_date, end=end_date)
        benchmark_returns = benchmark_data['Close'].pct_change().dropna()
        
        # Sauvegarder les données si un chemin est fourni
        if save_path:
            ensure_directory_exists(os.path.dirname(save_path))
            benchmark_data.to_pickle(save_path + "_benchmark_prices.pkl")
            benchmark_returns.to_pickle(save_path + "_benchmark_returns.pkl")
            logger.info(f"Données du benchmark sauvegardées dans {save_path}")
        
        return {
            'prices': benchmark_data,
            'returns': benchmark_returns
        }
    
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement des données du benchmark: {e}")
        return None

def get_risk_free_rate(start_date, end_date, save_path=None):
    """
    Obtenir le taux sans risque (en utilisant les bons du Trésor américain à 3 mois comme proxy)
    
    Args:
        start_date (str): Date de début au format 'YYYY-MM-DD'
        end_date (str): Date de fin au format 'YYYY-MM-DD'
        save_path (str, optional): Chemin pour sauvegarder les données
        
    Returns:
        pandas.Series: Série temporelle du taux sans risque
    """
    logger.info(f"Récupération du taux sans risque de {start_date} à {end_date}")
    
    try:
        # Utiliser le rendement des bons du Trésor US à 3 mois comme taux sans risque
        treasury_ticker = "^IRX"  # Rendement des bons du Trésor à 3 mois
        rf_data = yf.download(treasury_ticker, start=start_date, end=end_date)
        
        # Convertir le rendement annuel en rendement quotidien
        rf_daily = rf_data['Close'] / 100 / 252  # 252 jours de trading par an
        
        # Sauvegarder les données si un chemin est fourni
        if save_path:
            ensure_directory_exists(os.path.dirname(save_path))
            rf_daily.to_pickle(save_path + "_risk_free_rate.pkl")
            logger.info(f"Données du taux sans risque sauvegardées dans {save_path}")
        
        return rf_daily
    
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du taux sans risque: {e}")
        return None

def main():
    """
    Fonction principale pour exécuter le téléchargement de données
    """
    # Exemple d'utilisation
    output_dir = "../../data/raw/"
    ensure_directory_exists(output_dir)
    
    # Paramètres de téléchargement
    start_date = "2015-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Liste des actions (exemple: grandes capitalisations US)
    stock_tickers = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 
        'BRK-B', 'JPM', 'JNJ', 'V', 'PG', 
        'UNH', 'HD', 'BAC', 'MA', 'XOM',
        'DIS', 'VZ', 'CSCO', 'PFE', 'KO'
    ]
    
    # Liste des ETFs obligataires
    bond_tickers = [
        'AGG',  # iShares Core U.S. Aggregate Bond ETF
        'BND',  # Vanguard Total Bond Market ETF
        'TLT',  # iShares 20+ Year Treasury Bond ETF
        'SHY',  # iShares 1-3 Year Treasury Bond ETF
        'LQD',  # iShares iBoxx $ Investment Grade Corporate Bond ETF
        'HYG',  # iShares iBoxx $ High Yield Corporate Bond ETF
    ]
    
    # Indice de référence
    benchmark_ticker = '^GSPC'  # S&P 500
    
    # Télécharger les données
    stock_data = download_stock_data(stock_tickers, start_date, end_date, 
                                    save_path=output_dir + "stocks")
    bond_data = download_bond_data(bond_tickers, start_date, end_date, 
                                  save_path=output_dir + "bonds")
    benchmark_data = download_benchmark_data(benchmark_ticker, start_date, end_date, 
                                           save_path=output_dir + "benchmark")
    risk_free_rate = get_risk_free_rate(start_date, end_date, 
                                      save_path=output_dir + "risk_free")
    
    logger.info("Téléchargement des données terminé avec succès.")

if __name__ == "__main__":
    main()
