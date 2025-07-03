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
        default="visual_scatter",
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

def find_most_similar_features(df, features):
    """
    Parcourt toutes les paires de features, calcule le coefficient de corrélation
    de Pearson "from scratch", et renvoie la paire la plus corrélée.

    Args:
        df (pandas.DataFrame): DataFrame complet.
        features (list of str): Liste des noms de colonnes numériques.

    Returns:
        tuple: ((feature1, feature2), correlation_value)
    """
    # Étape 1 : initialiser la meilleure paire et le meilleur score
    best_pair = (None, None)
    best_score = -1.0
    # Étape 2 : parcourir toutes les paires de features sans redondance
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1 = features[i]
            f2 = features[j]
            # Étape 3 : filtrer et aligner les valeurs non-nulles pour f1 et f2
            # produit une série de booléens (True/False) indiquant pour
            # chaque élève si sa note en f1 et en f2 n’est pas manquante (NaN).
            mask = df[f1].notnull() & df[f2].notnull()
            # sélectionne, dans la colonne f1 et f2, seulement les
            # lignes où mask est True dans les deux matiere
            x_values = df.loc[mask, f1].tolist()
            y_values = df.loc[mask, f2].tolist()
            n = len(x_values)
            if n == 0:
                continue
            # Étape 4 : calculer le coefficient de corrélation de Pearson
            # Calcul des moyennes
            mean_x = sum(x_values) / n
            mean_y = sum(y_values) / n
            # Numérateur : somme des produits centrés (covariance non normalisée)
            covariance = 0.0
            # Pour chaque paire de valeurs, ajouter le produit des écarts
            for x, y in zip(x_values, y_values):
                dx = x - mean_x  # écart de x à sa moyenne
                dy = y - mean_y  # écart de y à sa moyenne
                covariance += dx * dy
            num = covariance
            # Dénominateurs : somme des carrés des écarts
            # calcule l'ecart avec la moyenne au carree de chaque matiere
            # ajoute tout ces carree pour avec den_x et den_y
            den_x = sum((x - mean_x) ** 2 for x in x_values)
            den_y = sum((y - mean_y) ** 2 for y in y_values)
            if den_x == 0 or den_y == 0:
                continue
            # Coefficient r = covariance / (écart-type x * écart-type y)
            r = num / ((den_x ** 0.5) * (den_y ** 0.5))
            # enregistre valeur absolue
            score = abs(r)
            # Étape 5 : mise à jour du meilleur score et de la meilleure paire
            if score > best_score:
                best_score = score
                best_pair = (f1, f2)

    return best_pair, best_score

def plot_scatter_for_best_pair(df, best_pair, outdir):
    """
    Trace et sauvegarde un scatter plot comparant les deux features les plus similaires,
    coloré par maison avec légende.

    Args:
        df (pandas.DataFrame): Dataset complet incluant 'Hogwarts House'.
        best_pair (tuple): Tuple des deux noms de colonnes (feature_x, feature_y).
        outdir (str): Dossier de sortie pour l'image.
    """
    # Préparer les deux features
    f1, f2 = best_pair
    # Récupérer les maisons et construire une palette de couleurs
    houses = df['Hogwarts House'].dropna().unique()
    cmap = plt.get_cmap('tab10', len(houses))
    color_dict = {house: cmap(i) for i, house in enumerate(houses)}

    # Filtrer lignes valides
    mask = df[f1].notnull() & df[f2].notnull() & df['Hogwarts House'].notnull()
    data = df.loc[mask, [f1, f2, 'Hogwarts House']]

    # Tracer
    plt.figure(figsize=(8, 6))
    for house in houses:
        subset = data[data['Hogwarts House'] == house]
        plt.scatter(
            subset[f1],
            subset[f2],
            alpha=0.7,
            label=house,
            color=color_dict[house]
        )
    plt.title("Scatter plot")
    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.legend(title="Maison")
    plt.grid(True)

    # Sauvegarde
    os.makedirs(outdir, exist_ok=True)
    fname = f"scatter_{f1.replace(' ', '_')}_{f2.replace(' ', '_')}.png"
    path = os.path.join(outdir, fname)
    plt.savefig(path)
    plt.close()
    print(f"→ Scatter plot créé : {path}")

def main():
    try:
        args = parse_arg()
        # Met le tableau entier dans un pandas.dataframe
        df = pd.read_csv(args.input_csv)
        # recupere toute les collone numeric
        numfeatures = get_numeric_features(df)
        # best_pair = deux noms de matières les plus similaires
        # best_score la valeur du coefficient qui montre leur degré de similarité.
        best_pair, best_score = find_most_similar_features(df, numfeatures)

        print(f"best_ pair = ", best_pair)
        print(f"best_score = ", best_score)
        # creer le fichier de sortie
        plot_scatter_for_best_pair(df, best_pair, args.outdir)

    except Exception as e:
        print(f"Une erreur est survenue : {e}")

if __name__ == "__main__":
    main()