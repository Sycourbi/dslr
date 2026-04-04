import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt


DEFAULT_FIG_WIDTH = 16.0
DEFAULT_FIG_HEIGHT = 9.0
DEFAULT_DPI = 120


def parse_args():
    """
    Parse les arguments de la ligne de commande.

    Returns:
        Namespace: les arguments parsés (input_csv).
    """
    # argparse.ArgumentParser:
    # Je crée un nouvel analyseur d’arguments (ArgumentParser)
    parser = argparse.ArgumentParser(
        description="Générer un histogramme pour la matière la plus homogène entre les maisons."
    )
    parser.add_argument(
        "input_csv",
        help="Chemin vers le fichier CSV d'entrée (dataset_train.csv)."
    )
    # Nombre de bins pour l'histogramme
    parser.add_argument(
        "--bins", "-b",
        type=int,
        default=50,
        help="Nombre de bins pour l'histogramme (défaut : 50)."
    )
    # Dossier de sortie pour les images
    parser.add_argument(
        "--outdir", "-o",
        default="visuals",
        help="Dossier de sortie pour les PNG (défaut : 'visuals')."
    )
    parser.add_argument(
        "--width",
        type=float,
        default=DEFAULT_FIG_WIDTH,
        help="Largeur de la figure en pouces (défaut : 16)."
    )
    parser.add_argument(
        "--height",
        type=float,
        default=DEFAULT_FIG_HEIGHT,
        help="Hauteur de la figure en pouces (défaut : 9)."
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="Résolution de sortie PNG (défaut : 120, soit 1920x1080 en 16x9)."
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

def find_most_homogeneous(df, features, houses):
    """
    Trouve la matière dont la distribution des notes est la plus homogène
    entre les différentes maisons.

    Args:
        df (pandas.DataFrame): Le dataset complet contenant 'Hogwarts House'.
        features (list of str): Liste des colonnes numériques (matières).
        houses (array-like): Liste des maisons.

    Returns:
        str: Nom de la matière la plus homogène (écart minimal entre les moyennes).
    """
    # Initialisation de la meilleure feature et du meilleur score
    best_feature = None
    best_score = float("inf")  # On commence avec une valeur très grande

    # Parcours de chaque matière (feature)
    for feat in features:
        means = []
        # Parcours de chaque maison
        for house in houses:
            # Filtrer les valeurs de la matière pour une maison donnée
            vals = df[df["Hogwarts House"] == house][feat].dropna()
            # On ne garde que les maisons ayant des valeurs
            if len(vals) > 0:
                # Calcul de la moyenne pour cette maison
                means.append(vals.mean())

        # On s'assure d'avoir au moins deux maisons pour comparer
        if len(means) > 1:
            # Calcul de l'écart entre la moyenne max et min
            score = max(means) - min(means)
            # Si cet écart est plus petit que le meilleur score actuel
            if score < best_score:
                # On met à jour le meilleur score et la meilleure feature
                best_score = score
                best_feature = feat

    # Retourne la matière la plus homogène
    return best_feature

def one_histogram(df, feature, houses, bins, outdir, width, height, dpi):
    """
    Trace un histogramme superposé pour une matière donnée et sauvegarde
    le résultat dans un seul PNG.

    Args:
        df (pandas.DataFrame): Le dataset complet contenant 'Hogwarts House'.
        feature (str): Nom de la colonne numérique à afficher.
        houses (array-like): Liste des quatre maisons.
        bins (int): Nombre de bins pour l'histogramme.
        outdir (str): Dossier où enregistrer le fichier unique.
    """

    # plt.subplots() crée une figure (fig) et un axe (ax)
    # - fig = le conteneur global de l'image
    # - ax = la zone où l'on dessine (histogramme ici)
    # figsize permet de définir la taille de l'image en pouces
    fig, ax = plt.subplots(figsize=(width, height))

    for house in houses:
        vals = df[df["Hogwarts House"] == house][feature].dropna()
        ax.hist(vals, bins=bins, alpha=0.5, label=house)
    
    # ax.set_title(feat)
    ax.set_xlabel(feature)
    ax.set_ylabel("Fréquence")
    ax.set_title(f"Histogramme de {feature} par maison")
    ax.legend(fontsize='small')

    # Ajuster la mise en page
    plt.tight_layout()
    # Enregistrer le fichier unique
    outfile = os.path.join(outdir, "histogram.png")
    fig.savefig(outfile, dpi=dpi)
    plt.close(fig)
    print(f"→ histogram.png créé dans {outdir}/")


def main():
    try:
        # Parser les arguments
        args = parse_args()
        # Charger le dataset
        df = pd.read_csv(args.input_csv)

        # Récupérer la liste des maisons
        houses = df["Hogwarts House"].dropna().unique()
        # Identifier les colonnes numériques (features)
        features = get_numeric_features(df)
        best_feature = find_most_homogeneous(df, features, houses)
        # Préparer le dossier de sortie 'visuals'
        os.makedirs(args.outdir, exist_ok=True)
        # Trace et sauvegarde un histogramme superposé pour la matière sélectionnée.
        one_histogram(
            df,
            best_feature,
            houses,
            args.bins,
            args.outdir,
            args.width,
            args.height,
            args.dpi,
        )
    
    except Exception as e:
        print(f"Une erreur est survenue : {e}")
    
if __name__ == "__main__":
    main()