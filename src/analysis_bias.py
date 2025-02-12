"""
Analysis of potential biases in film reviews on Fandango in 2015.
"""

from pathlib import Path

import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns


def load_data():
    """Load data"""
    fandango = pd.read_csv('data/fandango_scrape.csv')
    all_sites = pd.read_csv('data/all_sites_scores.csv')
    return fandango, all_sites

def preprocess(fandango, all_sites):
    # Extraction of year from film title
    fandango['YEAR'] = fandango['FILM'].str.extract(r'\((\d{4})\)')
    fandango['YEAR'] = pd.to_numeric(fandango['YEAR'])

    # Merge DataFrames : Keep only the films present in both
    df_merge = pd.merge(fandango, all_sites, on='FILM', how='inner')

    # Normalization of scores in all_sites (reset to 0-5 scale)
    df_merge['RottenTomatoes_norm'] = df_merge['RottenTomatoes'] / 20
    df_merge['RottenTomatoes_User_norm'] = df_merge['RottenTomatoes_User'] / 20
    df_merge['Metacritic_norm'] = df_merge['Metacritic'] / 20
    df_merge['Metacritic_User_norm'] = df_merge['Metacritic_User'] * 0.5
    df_merge['IMDB_norm'] = df_merge['IMDB'] * 0.5

    # Create a DataFrame with normalized scores only
    norm_scores = df_merge[['FILM',
                            'STARS', 'RATING',
                            'RottenTomatoes_norm', 'RottenTomatoes_User_norm',
                            'Metacritic_norm', 'Metacritic_User_norm',
                            'IMDB_norm']]


    return norm_scores


def plot_kde_distributions(norm_scores):
    """Creates a KDE graph comparing normalized score distributions across all platforms"""

    plt.figure(figsize=(12, 8))

    # Draw KDE curves
    sns.kdeplot(data=norm_scores, x='RottenTomatoes_norm', label='Rotten Tomatoes (Critiques)', fill=True, alpha=0.5)
    sns.kdeplot(data=norm_scores, x='RottenTomatoes_User_norm', label='Rotten Tomatoes (Utilisateurs)', fill=True,
                alpha=0.5)
    sns.kdeplot(data=norm_scores, x='Metacritic_norm', label='Metacritic (Critiques)', fill=True, alpha=0.5)
    sns.kdeplot(data=norm_scores, x='Metacritic_User_norm', label='Metacritic (Utilisateurs)', fill=True, alpha=0.5)
    sns.kdeplot(data=norm_scores, x='IMDB_norm', label='IMDb', fill=True, alpha=0.5)
    sns.kdeplot(data=norm_scores, x='STARS', label='Fandango Stars', fill=True, alpha=0.5)
    sns.kdeplot(data=norm_scores, x='RATING', label='Fandango Rating', fill=True, alpha=0.5)

    plt.xlim(0, 5)
    plt.xlabel("Normalized Score (0-5)")
    plt.ylabel("Density")
    plt.title("Comparison of normalized score distributions on different platforms")
    plt.legend()
    plt.tight_layout()

    # Create the 'visualizations' folder if it doesn't already exist
    Path('visualizations').mkdir(exist_ok=True)

    # Save graph
    plt.savefig(Path('visualizations') / 'kde_distributions.png')


def main():
    # Load data
    fandango, all_sites = load_data()

    # Data pre-processing and merging
    norm_scores = preprocess(fandango, all_sites)

    # Visualization
    plot_kde_distributions(norm_scores)



if __name__ == '__main__':
    main()