#!/usr/bin/env python3
import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt

def parse_arg():
    """
    Parse les arguments de la ligne de commande.

    Returns:
        Namespace: les arguments parsés (input_csv).
    """
    # argparse.ArgumentParser:
    # Je crée un nouvel analyseur d’arguments (ArgumentParser)
    parser = argparse.ArgumentParser(
        description="Générer un scatter plot des deux features les plus similaires."
    )
    parser.add_argument(
        "input_csv",
        help="Chemin vers le fichier CSV d'entrée (dataset_train.csv)."
    )
    # Dossier de sortie pour les images
    parser.add_argument(
        "--outdir", "-o",
        default="visual_pair",
        help="Dossier de sortie pour les PNG (défaut : 'visualization')."
    )
    return parser.parse_args()


def get_numeric_features(df):
    """
    Identifie et retourne les colonnes numériques correspondant aux matières.

    Args:
        df (pandas.DataFrame): Données complètes du fichier.

    Returns:
        list of str: Liste des noms de colonnes numériques, excluant 'Index'.
    """
    # Initialisation de la liste des colonnes numériques
    numeric_cols = []
    # Parcours de chaque colonne du DataFrame
    for col in df.columns:
        # df[col] renvoie une Series contenant toutes les valeurs de la colonne
        # df[col].dtype.kind renvoie un code à un caractère pour le type :
        #   'i' pour int, 'f' pour float, 'O' pour object (texte), 'M' pour datetime, etc.
        kind = df[col].dtype.kind
        # On sélectionne uniquement les colonnes dont le type est entier ou flottant
        if kind in ("i", "f"):
            numeric_cols.append(col)
    # Exclusion de la colonne 'Index' si elle apparaît dans les numériques
    if "Index" in numeric_cols:
        numeric_cols.remove("Index")
    return numeric_cols

def plot_pair_plot(df, features, outdir):
    """
    Trace et sauvegarde un pair plot (matrice d'histogrammes et scatter plots)
    pour toutes les features numériques, coloré par maison.

    Args:
        df (pandas.DataFrame): Dataset complet incluant 'Hogwarts House'.
        features (list of str): Liste des noms de colonnes numériques.
        outdir (str): Dossier de sortie pour le PNG.
    """
    n = len(features)
    # Préparer figure de taille adaptée
    fig, axes = plt.subplots(n, n, figsize=(n * 1.5, n * 1.5), squeeze=False)

    # Palette de couleurs par maison
    houses = df['Hogwarts House'].dropna().unique()
    cmap = plt.get_cmap('tab10', len(houses))
    color_dict = {house: cmap(i) for i, house in enumerate(houses)}

    for i, f1 in enumerate(features):
        for j, f2 in enumerate(features):
            ax = axes[i][j]
            if i == j:
                 # Histogramme univarié coloré par maison
                for house in houses:
                    vals = df.loc[df['Hogwarts House']==house, f1].dropna()
                    ax.hist(vals, bins=10, alpha=0.5, color=color_dict[house], label=house)
                # Histogramme univarié
                # vals = df[f1].dropna()
                # ax.hist(vals, bins=10, color='gray', alpha=0.7)
                ax.set_ylabel(f1, fontsize=6)
                ax.set_yticks([])
                ax.set_xticks([])
            else:
                # Scatter bivarié coloré par maison
                for house in houses:
                    mask = (
                        df[f1].notnull() & df[f2].notnull() &
                        (df['Hogwarts House'] == house)
                    )
                    ax.scatter(
                        df.loc[mask, f2],
                        df.loc[mask, f1],
                        s=5, alpha=0.6,
                        color=color_dict[house]
                    )
                ax.set_yticks([])
                ax.set_xticks([])
            # Only label outer axes
            if i == n - 1:
                ax.set_xlabel(f2, fontsize=6)
            if j == 0 and i != j:
                ax.set_ylabel(f1, fontsize=6)

    # Légende globale
    legend_ax = fig.add_subplot(111, frame_on=False)
    legend_ax.axis('off')
    for house in houses:
        legend_ax.scatter([], [], color=color_dict[house], label=house)
    legend_ax.legend(title='Maison', loc='upper center', ncol=len(houses), fontsize=6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    title = 'Pair Plot des matières'
    fig.suptitle(title, fontsize=10)

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, 'pair_plot.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f"→ pair_plot.png créé dans {outdir}/")


def main():
    
    args = parse_arg()
    # Met le tableau entier dans un pandas.dataframe
    df = pd.read_csv(args.input_csv)
    # Identifier les colonnes numériques (features)
    features = get_numeric_features(df)
    # Générer et sauvegarder le pair plot
    plot_pair_plot(df, features, args.outdir)
    
if __name__ == "__main__":
    main()