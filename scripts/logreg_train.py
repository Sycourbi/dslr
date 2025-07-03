import argparse
import os
import json

import pandas as pd
import numpy as np


def parse_args():
    """
    Parse les arguments de la ligne de commande pour logreg_train.py.

    Returns:
        Namespace: contient les attributs suivants
          - input_csv (str)      : chemin vers dataset_train.csv
          - alpha (float)        : taux d'apprentissage (learning rate)
          - iterations (int)     : nombre d'itérations de gradient descent
          - out (str)            : chemin du fichier de sortie pour les poids
    """
    parser = argparse.ArgumentParser(
        description="Entraîner un classifieur logistisque one-vs-all et sauvegarder les poids."
    )
    parser.add_argument(
        "input_csv",
        help="Chemin vers le fichier d'entraînement (dataset_train.csv)."
    )
    parser.add_argument(
        "--alpha", "-a",
        type=float,
        default=0.01,
        help="Taux d'apprentissage pour la descente de gradient (défaut : 0.01)."
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=1000,
        help="Nombre d'itérations de gradient descent (défaut : 1000)."
    )
    parser.add_argument(
        "--out", "-o",
        default="weights.json",
        help="Fichier de sortie pour les poids entraînés (défaut : 'weights.json')."
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

def load_and_prepare_data(path):
    """
    Lit le CSV d'entraînement et prépare X, y pour l'entraînement.

    Étapes :
    1. Lecture du fichier CSV via pandas.
    2. Suppression des lignes comportant des NaN dans les colonnes de notes ou dans 'Hogwarts House'.
    3. Extraction de X : les 13 colonnes numériques correspondant aux matières.
    4. Extraction de y : encodage de 'Hogwarts House' en entiers 0–3.
    5. Retourne :
         - X (pandas.DataFrame) : les features numériques,
         - y (list[int])        : la liste des labels encodés,
         - house_map (dict)     : mapping maison → label numérique,
         - inv_house_map (dict) : mapping label numérique → maison (pour la prédiction).
    """
    df = pd.read_csv(path)

    # On enlève tout élève ne disposant pas d'une valeur pour la maison
    # ou pour au moins une matière (pour garantir X et y alignés).
    df = df.dropna(subset=['Hogwarts House'] + get_numeric_features(df))

    # get_numeric_features(df) renvoie la liste des matières (colonnes de notes).
    # reset_index(drop=True) réindexe les élèves de 0 à m-1 après le dropna.
    features = get_numeric_features(df)
    X = df[features].reset_index(drop=True)

    # 4. Encoder y (cible) en entiers
    # 'houses' contient la liste ordonnée des noms de chaque maison.
    # 'house_map' associe chaque nom de maison à un entier unique (0,1,2,3).
    # 'inv_house_map' permet d'inverser ce mapping pour la prédiction.
    # 'y' devient une liste d'entiers correspondant à chaque élève.
    houses = df['Hogwarts House'].unique()
    house_map = {house: idx for idx, house in enumerate(houses)}
    inv_house_map = {idx: house for house, idx in house_map.items()}
    y = df['Hogwarts House'].map(house_map).tolist()

    return X, y, house_map, inv_house_map

def normalize_features(X_df):
    """
    Centre et réduit les features "from scratch".

    Returns:
        X_norm (ndarray): matrice normalisée.
        mu (ndarray): moyennes de chaque colonne.
        sigma (ndarray): écart-types de chaque colonne.
    """
    # Convertir le DataFrame en numpy array de floats
    X = X_df.values.astype(float)
    # Calculer la moyenne de chaque colonne (feature)
    mu = X.mean(axis=0)
    # Calculer l'écart-type (population) de chaque colonne
    sigma = X.std(axis=0, ddof=0)
    # Normaliser chaque valeur: (x_ij - mu_j) / sigma_j
    X_norm = (X - mu) / sigma
    # Retourner la matrice normalisée et les paramètres de normalisation sous forme de listes
    return X_norm, mu.tolist(), sigma.tolist()


def sigmoid(z):
    """Fonction sigmoïde."""
    return 1.0 / (1.0 + np.exp(-z))


def train_one_vs_all(X, y, alpha, num_iters):
    """
    Entraîne un classifieur logistique one-vs-all.

    Args:
        X (ndarray): matrice normalisée avec biais ajouté.
        y (ndarray): labels 0..K-1.
        alpha (float): learning rate.
        num_iters (int): nombre d'itérations.

    Returns:
        thetas (ndarray): poids de dimension (K, n+1).
    """
    # Récupérer dimensions de X
    # m = nombre d'exemples d'entraînement, n = nombre de paramètres (features + biais)
    m, n = X.shape

    # Identifier les classes uniques dans y (p. ex. 4 maisons)
    classes = np.unique(y)
    # K = nombre total de classes
    K = len(classes)

    # Initialiser la matrice des poids à zeros (une ligne par classe)
    thetas = np.zeros((K, n))

    # Parcourir chaque classe pour entraîner un classifieur binaire
    for k in classes:
        # Initialiser les poids theta pour la classe k
        theta = np.zeros(n)
        # Binariser y : yk[i] = 1 si l'exemple i appartient à la classe k, sinon 0
        yk = (y == k).astype(float)

        # Descente de gradient pour cette classe
        for _ in range(num_iters):
            # Calculer la prédiction sigmoïde pour tous les m exemples : h = g(X · theta)
            h = sigmoid(X.dot(theta))
            # Calculer le gradient du cost w.r.t. theta : (1/m) * X^T · (h - yk)
            grad = (1/m) * X.T.dot(h - yk)
            # Mettre à jour les paramètres : theta ← theta - alpha * gradient
            theta -= alpha * grad

        # Stocker la solution finale pour la classe k dans la matrice thetas
        thetas[int(k), :] = theta

    # Retourner la matrice des poids formée de K vecteurs theta
    return thetas


def main():
    try:
        # Parser les arguments
        args = parse_args()
         # Chargement et préparation
        X_df, y, house_map, inv_house_map = load_and_prepare_data(args.input_csv)
        # Normalisation
        X_norm, mu, sigma = normalize_features(X_df)
        # Ajouter la colonne biais
        m = X_norm.shape[0]
        X_bias = np.hstack([np.ones((m, 1)), X_norm])
        # Entraînement
        thetas = train_one_vs_all(X_bias, y, args.alpha, args.iterations)
        # Sauvegarde
        output = {
            'thetas': thetas.tolist(),
            'mu': mu,
            'sigma': sigma,
            'house_map': house_map,
            'inv_house_map': inv_house_map
        }
        with open(args.out, 'w') as f:
            json.dump(output, f)
        print(f"→ Poids et paramètres enregistrés dans {args.out}")

    except Exception as e:
        print(f"Une erreur est survenue : {e}")
    
if __name__ == "__main__":
    main()