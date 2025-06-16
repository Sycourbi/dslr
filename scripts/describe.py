#!/usr/bin/env python3
# argparse : pour parser les arguments de ligne de commande (chemin du fichier CSV)
import argparse
# pandas (pd) : pour charger le CSV et manipuler les données sous forme de DataFrame
import pandas as pd

import math

def parse_args():
    """
    Parse les arguments de la ligne de commande.

    Returns:
        Namespace: les arguments parsés (input_csv).
    """
    # argparse.ArgumentParser:
    # Je crée un nouvel analyseur d’arguments (ArgumentParser)
    parser = argparse.ArgumentParser(
        description="Afficher l'entête du tableau descriptif du dataset."
    )
    parser.add_argument(
        "input_csv",
        help="Chemin vers le fichier CSV d'entrée (dataset_train.csv)."
    )
    return parser.parse_args()


def load_data(csv_path):
    """
    Charge le dataset CSV dans un DataFrame pandas.
    DataFrame = tableau en 2D de la totaliter du fichier

    Args:
        csv_path (str): Chemin vers le fichier CSV. (dataset/dataset_train.csv)

    Returns:
        pandas.DataFrame: Le DataFrame contenant toutes les données.
    """
    return pd.read_csv(csv_path)


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


def compute_counts(df, features):
    """
    Calcule le nombre de valeurs non manquantes pour chaque feature.

    Args:
        df (pandas.DataFrame): Le DataFrame chargé.
        features (list of str): Liste des noms de colonnes à compter.

    Returns:
        list of int: Liste des counts pour chaque colonne.
    """
    counts = []
    for col in features:
        # Compter manuellement les valeurs non nulles
        count = sum(1 for x in df[col] if pd.notnull(x))
        counts.append(count)
    return counts

def compute_means(df, features):
    """
    Calcule la moyenne de chaque feature (sans utiliser df.mean()).

    Args:
        df (pandas.DataFrame): Le DataFrame chargé.
        features (list of str): Liste des noms de colonnes à traiter.

    Returns:
        list of float: Moyennes des valeurs non nulles pour chaque colonne.
    """
    means = []
    for col in features:
        # Récupérer les valeurs non nulles
        vals = [x for x in df[col] if pd.notnull(x)]
        # Somme et comptage
        total = sum(vals)
        count = len(vals)
        mean = total / count if count else 0.0
        means.append(mean)
    return means

def compute_stds(df, features):
    """
    Calcule l'écart-type (population) pour chaque feature

    Args:
        df (pandas.DataFrame): Le DataFrame contenant les données.
        features (list of str): Liste des noms de colonnes (features) à traiter.

    Returns:
        list of float: Écarts-types calculés pour chaque feature.
    """
    stds = []  # Liste qui contiendra l'écart-type de chaque colonne
    # Parcours de chaque feature (colonne) numérique
    for col in features:
        # Filtrer les valeurs non-nulles pour éviter les NaN
        vals = [x for x in df[col] if pd.notnull(x)]
        n = len(vals)  # Nombre de valeurs valides
        if n:
            # Calculer la moyenne
            mean = sum(vals) / n
            # Calculer la variance population : moyenne des carrés des écarts
            variance = sum((x - mean) ** 2 for x in vals) / n
            # L'écart-type est la racine carrée de la variance
            stds.append(math.sqrt(variance))
        else:
            # Si aucune valeur valide, on ajoute 0.0
            stds.append(0.0)
    return stds

def compute_mins(df, features):
    """
    Calcule la valeur minimale pour chaque feature sans utiliser df.min().

    Args:
        df (pandas.DataFrame): Le DataFrame chargé.
        features (list of str): Liste des noms de colonnes à traiter.

    Returns:
        list of float: Valeurs minimales non nulles pour chaque colonne.
    """
    mins = []  # Liste qui contiendra le minimum de chaque colonne
    for col in features:
        # Filtrer les valeurs non-nulles pour éviter les NaN
        vals = [x for x in df[col] if pd.notnull(x)]
        if vals:
            # Trouver la plus petite valeur manuellement
            min_val = vals[0]
            for x in vals[1:]:
                if x < min_val:
                    min_val = x
            mins.append(min_val)
        else:
            # Si aucune valeur valide, on ajoute 0.0
            mins.append(0.0)
    return mins


def compute_pourcent(df, features, pourcent):
    """
    Calcule manuellement le pourcentage donné pour chaque feature.

    Args:
        df (pandas.DataFrame): Le DataFrame chargé.
        features (list of str): Liste des noms de colonnes.
        pourcent (float): Le percentile à calculer (25, 50 ou 75).

    Returns:
        list of float: Valeurs des pourcentage pour chaque colonne.
    """
    
    results = []
    for col in features:
        # Filtrer et trier les valeurs non nulles
        vals = sorted(x for x in df[col] if pd.notnull(x))
        n = len(vals)  # nombre d'éléments
        if n:
            # Calcul de l'index exact dans la liste triée
            # idx = (pourcent/100) * (n - 1)
            idx = pourcent / 100 * (n - 1)
            # Si idx n’est pas entier (ex. 2.3), 
            # il se situe entre deux indices entiers :
            lo = int(math.floor(idx))
            hi = int(math.ceil(idx))
            if lo == hi:
                # Si idx est entier, on prend la valeur exacte
                results.append(vals[lo])
            else:
                # Partie fractionnaire pour interpolation
                frac = idx - lo
                # Interpolation linéaire entre les deux valeurs
                lower_val = vals[lo]
                upper_val = vals[hi]
                interp_val = lower_val + (upper_val - lower_val) * frac
                results.append(interp_val)
        else:
            # Si aucune valeur valide, renvoyer 0.0
            results.append(0.0)
    return results


def compute_maxs(df, features):
    """
    Calcule la valeur maximale pour chaque feature 

    Args:
        df (pandas.DataFrame): Le DataFrame contenant les données.
        features (list of str): Liste des colonnes à traiter.

    Returns:
        list of float: Valeurs maximales non-nulles pour chaque colonne.
    """
    maxs = []  # Liste à remplir avec le maximum de chaque feature
    # Parcours de chaque colonne spécifiée
    for col in features:
        # Filtrer les valeurs non-nulles pour éviter les NaN
        vals = [x for x in df[col] if pd.notnull(x)]
        if vals:
            # Initialiser max_val à la première valeur
            max_val = vals[0]
            # Comparer chaque valeur suivante pour trouver le maximum
            for x in vals[1:]:
                if x > max_val:
                    max_val = x
            maxs.append(max_val)
        else:
            # Si aucune valeur n'est valide, on ajoute 0.0 par défaut
            maxs.append(0.0)
    return maxs

def abbreviate(features, max_len=10):
    """
    Tronque chaque nom de matiere à max_len caractères,
    en ajoutant '…' si on coupe.

    Returns:
        list de str: Liste des matiere abreger
    """
    abbr = []
    for f in features:
        if len(f) > max_len:
            abbr.append(f[: max_len - 1] + "…")
        else:
            abbr.append(f)
    return abbr

def creat_dataframe(features, counts, means, stds, mins, p25, p50, p75, maxs):
    """
    Assemble les statistiques dans un DataFrame : statistiques en lignes,
    matières en colonnes.

    Returns:
        pandas.DataFrame
    """
    stats_dict = {
        "Count": counts,
        "Mean":  means,
        "Std":   stds,
        "Min":   mins,
        "25%":   p25,
        "50%":   p50,
        "75%":   p75,
        "Max":   maxs,
    }
    #(.T) échange lignes et colonnes pour obtenir lignes=statistiques et colonnes=features
    df_stats = pd.DataFrame(stats_dict, index=features).T
    return df_stats

def main():
    try:
        # 1. Parser les arguments
        args = parse_args()

        # 2. Met tout les donnee du fichier dans DataFrame
        df = load_data(args.input_csv)

        # 3. Identifier les colonnes numériques (matières)
        features = get_numeric_features(df)

        # 4. Calculer Count, Mean et Std
        counts = compute_counts(df, features)
        means  = compute_means(df, features)
        stds   = compute_stds(df, features)
        min    = compute_mins(df, features)
        p25    = compute_pourcent(df, features, 25)
        p50    = compute_pourcent(df, features, 50)
        p75    = compute_pourcent(df, features, 75)
        max    = compute_maxs(df, features)

        # 5. Abrege les nom des matieres
        features = abbreviate(features, max_len=12)
        # . Construire et afficher le tableau descriptif
        df_stats = creat_dataframe(features, counts, means, stds, min, p25, p50, p75, max)
        print(df_stats.to_string(float_format="%.6f"))

    except Exception as e:
        # Un simple message d'erreur pour indiquer le problème
        print(f"Une erreur est survenue : {e}")

if __name__ == "__main__":
    main()