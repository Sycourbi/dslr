# Pour exposer un contrat CLI stable et standard.
import argparse

# Pour persister les parametres appris dans un format inter-outils.
import json

# Pour vectoriser les calculs de regression logistique.
import numpy as np

# Pour charger et manipuler les donnees tabulaires du sujet DSLR.
import pandas as pd

# Pour accepter l'execution depuis le dossier scripts directement.
try:
    # Pour charger le logger d'analyse en mode script local.
    from analysis_log_train import AnalysisLogger
# Pour rester compatible si l'import local n'est pas resolu.
except ImportError:
    # Pour charger le meme logger via le chemin package scripts.
    from scripts.analysis_log_train import AnalysisLogger


# Pour centraliser la definition des options CLI de logreg_train.
def parse_command_line_arguments():
    """
    Definit le contrat CLI de l'entrainement logistique one-vs-all.

    Returns:
        argparse.Namespace: arguments valides pour piloter le train.
    """
    # Pour regrouper la validation des arguments dans argparse.
    argument_parser = argparse.ArgumentParser(
        description=(
            "Entrainer un classifieur logistique one-vs-all et "
            "sauvegarder ses parametres."
        )
    )
    # Pour imposer le dataset d'entrainement comme entree obligatoire.
    argument_parser.add_argument(
        "input_csv_path",
        help="Chemin vers le fichier d'entrainement (dataset_train.csv).",
    )
    # Pour permettre d'ajuster le pas de descente sans modifier le code.
    argument_parser.add_argument(
        "--alpha",
        "-a",
        dest="learning_rate",
        type=float,
        default=0.01,
        help=(
            "Taux d'apprentissage pour la descente de gradient "
            "(defaut : 0.01)."
        ),
    )
    # Pour controler le nombre d'iterations de gradient descent en CLI.
    argument_parser.add_argument(
        "--iterations",
        "-n",
        dest="iteration_count",
        type=int,
        default=1000,
        help="Nombre d'iterations de gradient descent (defaut : 1000).",
    )
    # Pour garder le nom du fichier de sortie configurable.
    argument_parser.add_argument(
        "--out",
        "-o",
        dest="output_parameter_path",
        default="weights.json",
        help=(
            "Fichier de sortie des parametres appris "
            "(defaut : 'weights.json')."
        ),
    )
    # Pour activer les logs d'analyse sans polluer la sortie standard.
    argument_parser.add_argument(
        "--analysis-log",
        dest="enable_analysis_log",
        action="store_true",
        help="Active les logs detailles pour analyser l'entrainement.",
    )
    # Pour retourner des arguments deja verifies par argparse.
    return argument_parser.parse_args()


# Pour extraire un schema de features numeriques coherent avec DSLR.
def get_discipline_names(dataset):
    """
    Extrait les noms des disciplines numeriques utiles au modele.

    Args:
        dataset (pandas.DataFrame): dataset brut charge depuis le CSV.

    Returns:
        list[str]: disciplines numeriques, sans la colonne technique Index.
    """
    # Pour construire l'ordre des features retenues pour train et predict.
    discipline_names = []
    # Pour inspecter toutes les colonnes candidates presentes dans le CSV.
    for discipline_name in dataset.columns:
        # Pour recuperer un code de type compact et fiable de pandas.
        discipline_kind = dataset[discipline_name].dtype.kind
        # Pour ne garder que les colonnes entieres ou flottantes.
        if discipline_kind in ("i", "f"):
            # Pour conserver cette colonne dans le schema des features.
            discipline_names.append(discipline_name)
    # Pour exclure l'identifiant technique qui n'est pas une discipline.
    if "Index" in discipline_names:
        # Pour eviter qu'un identifiant biaise l'apprentissage du modele.
        discipline_names.remove("Index")
    # Pour fournir un ordre de colonnes stable au reste du pipeline.
    return discipline_names


# Pour charger le train CSV et aligner les structures attendues par le modele.
def load_and_prepare_dataset(input_csv_path):
    """
    Charge le jeu d'entrainement et construit X, y et mappings maison.

    Args:
        input_csv_path (str): chemin du CSV d'entrainement.

    Returns:
        tuple: X disciplines, y codes maison, mappings maison et features.
    """
    # Pour materialiser les donnees sources dans un DataFrame unique.
    raw_students_dataset = pd.read_csv(input_csv_path)
    # Pour fixer l'ordre des features numeriques retenues par le modele.
    discipline_names = get_discipline_names(raw_students_dataset)
    # Pour eviter les NaN dans X ou y pendant la descente de gradient.
    students_with_complete_disciplines_scores = raw_students_dataset.dropna(
        subset=["Hogwarts House"] + discipline_names
    )
    # Pour extraire uniquement les disciplines dans l'ordre fige ci-dessus.
    students_disciplines_scores = students_with_complete_disciplines_scores[
        discipline_names
    ].reset_index(drop=True)
    # Pour imposer un ordre canonique des classes sur tout le projet.
    house_names = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    # Pour mapper les noms texte vers des codes numeriques stables.
    house_code_by_name = {
        house_name: code for code, house_name in enumerate(house_names)
    }
    # Pour conserver la traduction inverse au moment d'exporter.
    house_name_by_code = {
        code: house_name for house_name, code in house_code_by_name.items()
    }
    # Pour convertir la cible texte en vecteur d'entiers exploitable.
    assigned_house_codes_for_students = (
        students_with_complete_disciplines_scores["Hogwarts House"]
        .map(house_code_by_name)
        .to_numpy(dtype=int)
    )
    # Pour renvoyer toutes les structures necessaires a l'entrainement.
    return (
        students_disciplines_scores,
        assigned_house_codes_for_students,
        house_code_by_name,
        house_name_by_code,
        discipline_names,
    )


# Pour normaliser les disciplines selon la convention statistique du train.
def standardize_disciplines_scores(students_disciplines_scores):
    """
    Standardise les notes discipline par discipline avec moyenne et ecart-type.

    Args:
        students_disciplines_scores (pandas.DataFrame): features brutes.

    Returns:
        tuple: matrice standardisee, moyennes et ecarts-types de reference.
    """
    # Pour convertir explicitement en float et fiabiliser les operations numpy.
    disciplines_scores_for_students = students_disciplines_scores.to_numpy(
        dtype=float
    )
    # Pour memoriser la moyenne par discipline necessaire au predict.
    average_scores_by_disciplines = disciplines_scores_for_students.mean(axis=0)
    # Pour memoriser l'echelle par discipline necessaire au predict.
    standard_deviations_by_disciplines = disciplines_scores_for_students.std(
        axis=0, ddof=0
    )
    # Pour projeter toutes les disciplines sur une echelle comparable.
    standardized_disciplines_scores = (
        disciplines_scores_for_students - average_scores_by_disciplines
    ) / standard_deviations_by_disciplines
    # Pour fournir les stats de normalisation au fichier de poids final.
    return (
        standardized_disciplines_scores,
        average_scores_by_disciplines.tolist(),
        standard_deviations_by_disciplines.tolist(),
    )


# Pour transformer un score lineaire en probabilite exploitable.
def compute_sigmoid(linear_score_array):
    """
    Applique la sigmoide element par element.

    Args:
        linear_score_array (numpy.ndarray): logits a convertir.

    Returns:
        numpy.ndarray: probabilites dans l'intervalle [0, 1].
    """
    # Pour convertir les logits en probabilites dans le schema logistique.
    return 1.0 / (1.0 + np.exp(-linear_score_array))


# Pour regrouper les logs lies au contexte d'une maison one-vs-all.
def log_house_training_context(
    analysis_logger, current_house_code, are_students_assigned_to_current_house
):
    """
    Emet les logs d'entree de boucle maison.

    Args:
        analysis_logger (AnalysisLogger): logger d'analyse conditionnel.
        current_house_code (int): code de la maison courante.
        are_students_assigned_to_current_house (numpy.ndarray): cible binaire.
    """
    analysis_logger.log_house_header(current_house_code)
    analysis_logger.log_students_assigned_to_current_house(
        are_students_assigned_to_current_house
    )


# Pour regrouper les logs d'une iteration avant la mise a jour des poids.
def log_house_iteration_before_weight_update(
    analysis_logger,
    iteration_index,
    students_disciplines_scores_with_bias,
    current_house_weights,
    predicted_probability_of_current_house,
    are_students_assigned_to_current_house,
    prediction_error_by_students,
    bias_and_standardized_disciplines_scores_error_sum,
    students_count,
    current_house_weight_gradient,
    learning_rate,
):
    """
    Emet les logs detailles d'une iteration avant update.

    Args:
        analysis_logger (AnalysisLogger): logger d'analyse conditionnel.
        iteration_index (int): index d'iteration de GD.
        students_disciplines_scores_with_bias (numpy.ndarray): X avec biais.
        current_house_weights (numpy.ndarray): poids actuels.
        predicted_probability_of_current_house (numpy.ndarray):
            proba pre-update.
        are_students_assigned_to_current_house (numpy.ndarray): cible binaire.
        prediction_error_by_students (numpy.ndarray): erreur proba-cible.
        bias_and_standardized_disciplines_scores_error_sum (numpy.ndarray):
            somme des erreurs ponderees.
        students_count (int): nombre d'eleves dans le batch.
        current_house_weight_gradient (numpy.ndarray): gradient moyen.
        learning_rate (float): pas de descente de gradient.
    """
    analysis_logger.log_iteration_header(iteration_index)
    analysis_logger.log_predicted_probability(
        students_disciplines_scores_with_bias,
        current_house_weights,
        predicted_probability_of_current_house,
    )
    analysis_logger.log_prediction_error(
        predicted_probability_of_current_house,
        are_students_assigned_to_current_house,
        prediction_error_by_students,
    )
    analysis_logger.log_bias_and_standardized_disciplines_scores_error_sum(
        students_disciplines_scores_with_bias,
        prediction_error_by_students,
        bias_and_standardized_disciplines_scores_error_sum,
    )
    analysis_logger.log_current_house_weight_gradient(
        students_count,
        bias_and_standardized_disciplines_scores_error_sum,
        current_house_weight_gradient,
    )
    analysis_logger.log_current_house_weights_before_update(
        current_house_weights, learning_rate, current_house_weight_gradient
    )


# Pour isoler le log post-update des poids d'iteration.
def log_house_iteration_after_weight_update(
    analysis_logger, current_house_weights
):
    """
    Emet le log des poids apres la mise a jour d'iteration.

    Args:
        analysis_logger (AnalysisLogger): logger d'analyse conditionnel.
        current_house_weights (numpy.ndarray): poids apres update.
    """
    analysis_logger.log_current_house_weights_after_update(
        current_house_weights
    )


# Pour isoler le log recapitulatif des poids apres une maison.
def log_house_weights_summary(analysis_logger, house_disciplines_weights):
    """
    Emet le log de synthese des poids apres la boucle maison.

    Args:
        analysis_logger (AnalysisLogger): logger d'analyse conditionnel.
        house_disciplines_weights (numpy.ndarray): matrice complete des poids.
    """
    analysis_logger.log_house_disciplines_weights(house_disciplines_weights)


# Pour entrainer un modele one-vs-all maison par maison.
def fit_one_vs_rest_house_classifier(
    students_disciplines_scores_with_bias,
    assigned_house_codes_for_students,
    learning_rate,
    iteration_count,
    analysis_logger,
):
    """
    Entraine les poids one-vs-all via descente de gradient batch.

    Args:
        students_disciplines_scores_with_bias (numpy.ndarray): X avec biais.
        assigned_house_codes_for_students (numpy.ndarray): y code par eleve.
        learning_rate (float): pas de mise a jour des poids.
        iteration_count (int): nombre d'iterations de gradient descent.
        analysis_logger (AnalysisLogger): logger verbeux conditionnel.

    Returns:
        numpy.ndarray: poids de chaque maison, biais inclus en colonne 0.
    """
    # Pour recuperer la taille du batch et la dimension des poids par classe.
    (
        students_count,
        disciplines_plus_bias_count,
    ) = students_disciplines_scores_with_bias.shape
    # Pour identifier les classes effectivement presentes dans le dataset.
    unique_house_codes = np.unique(assigned_house_codes_for_students)
    # Pour allouer une ligne de poids par maison cible.
    house_count = len(unique_house_codes)
    # Pour initialiser tous les poids a zero avant apprentissage.
    house_disciplines_weights = np.zeros(
        (house_count, disciplines_plus_bias_count)
    )
    # Pour entrainer une regression binaire independente par maison.
    for current_house_code in unique_house_codes:
        # Pour repartir de zero pour le classifieur binaire courant.
        current_house_weights = np.zeros(disciplines_plus_bias_count)
        # Pour construire la cible binaire de la maison courante.
        are_students_assigned_to_current_house = (
            assigned_house_codes_for_students == current_house_code
        ).astype(float)
        # Pour exposer le contexte de la maison courante en mode analyse.
        log_house_training_context(
            analysis_logger,
            current_house_code,
            are_students_assigned_to_current_house,
        )
        # Pour appliquer les mises a jour de gradient un nombre fixe de fois.
        for iteration_index in range(iteration_count):
            # Pour produire les probabilites de la maison courante.
            predicted_probability_of_current_house = compute_sigmoid(
                students_disciplines_scores_with_bias.dot(current_house_weights)
            )
            # Pour mesurer l'ecart signe entre prediction et cible binaire.
            prediction_error_by_students = (
                predicted_probability_of_current_house
                - are_students_assigned_to_current_house
            )
            # Pour agreger l'erreur sur chaque coefficient via X^T . erreur.
            bias_and_standardized_disciplines_scores_error_sum = (
                students_disciplines_scores_with_bias.T.dot(
                    prediction_error_by_students
                )
            )
            # Pour moyenner le gradient et garder un pas stable.
            current_house_weight_gradient = (
                1 / students_count
            ) * bias_and_standardized_disciplines_scores_error_sum
            # Pour centraliser les logs detailles de l'iteration pre-update.
            log_house_iteration_before_weight_update(
                analysis_logger,
                iteration_index,
                students_disciplines_scores_with_bias,
                current_house_weights,
                predicted_probability_of_current_house,
                are_students_assigned_to_current_house,
                prediction_error_by_students,
                bias_and_standardized_disciplines_scores_error_sum,
                students_count,
                current_house_weight_gradient,
                learning_rate,
            )
            # Pour appliquer la descente de gradient sur les poids courants.
            current_house_weights -= (
                learning_rate * current_house_weight_gradient
            )
            # Pour afficher les poids apres update via un helper dedie.
            log_house_iteration_after_weight_update(
                analysis_logger, current_house_weights
            )
        # Pour stocker les poids finaux de la maison a son index canonique.
        house_disciplines_weights[
            int(current_house_code), :
        ] = current_house_weights
        # Pour afficher l'etat global des poids via helper dedie.
        log_house_weights_summary(analysis_logger, house_disciplines_weights)
    # Pour renvoyer la matrice de poids complete au flux principal.
    return house_disciplines_weights


# Pour orchestrer train, serialisation JSON et gestion d'erreurs CLI.
def main():
    """
    Execute l'entrainement et sauvegarde les parametres appris.

    Effets de bord:
        Ecrit un JSON de poids, de stats et de mapping de classes.
    """
    # Pour centraliser tous les echecs sous un message CLI explicite.
    try:
        # Pour recuperer toutes les options d'execution validees.
        cli_arguments = parse_command_line_arguments()
        # Pour activer ou non les traces detaillees selon le flag CLI.
        analysis_logger = AnalysisLogger(cli_arguments.enable_analysis_log)
        # Pour charger X/y train et mappings maison dans un bloc coherent.
        (
            students_disciplines_scores,
            assigned_house_codes_for_students,
            house_code_by_name,
            house_name_by_code,
            discipline_names,
        ) = load_and_prepare_dataset(cli_arguments.input_csv_path)
        # Pour normaliser X et conserver les stats utiles au predict.
        (
            standardized_disciplines_scores,
            average_scores_by_disciplines,
            standard_deviations_by_disciplines,
        ) = standardize_disciplines_scores(students_disciplines_scores)
        # Pour dimensionner la colonne de biais de facon deterministe.
        students_count = standardized_disciplines_scores.shape[0]
        # Pour reproduire la convention train/predict avec biais en colonne 0.
        students_disciplines_scores_with_bias = np.hstack(
            [np.ones((students_count, 1)), standardized_disciplines_scores]
        )
        # Pour afficher l'etat initial des donnees si le mode analyse est actif.
        analysis_logger.log_initial_scores(
            students_disciplines_scores,
            average_scores_by_disciplines,
            standard_deviations_by_disciplines,
            standardized_disciplines_scores,
            students_count,
            students_disciplines_scores_with_bias,
        )
        # Pour lancer l'apprentissage one-vs-all sur les donnees preparees.
        house_disciplines_weights = fit_one_vs_rest_house_classifier(
            students_disciplines_scores_with_bias,
            assigned_house_codes_for_students,
            cli_arguments.learning_rate,
            cli_arguments.iteration_count,
            analysis_logger,
        )
        # Pour figer toutes les infos necessaires au predict dans un bundle.
        trained_parameter_bundle = {
            "thetas": house_disciplines_weights.tolist(),
            "mu": average_scores_by_disciplines,
            "sigma": standard_deviations_by_disciplines,
            "features": discipline_names,
            "house_map": house_code_by_name,
            "inv_house_map": house_name_by_code,
        }
        # Pour ouvrir le fichier de sortie cible en mode ecriture texte.
        with open(cli_arguments.output_parameter_path, "w") as output_file:
            # Pour serialiser le bundle sans perdre la structure attendue.
            json.dump(trained_parameter_bundle, output_file)
        # Pour signaler explicitement la fin du train et le chemin de sortie.
        print(
            "→ Poids et parametres enregistres dans "
            f"{cli_arguments.output_parameter_path}"
        )
    # Pour eviter un crash brut et garder un message de diagnostic stable.
    except Exception as exception:
        # Pour remonter la cause immediate dans la sortie standard CLI.
        print(f"Une erreur est survenue : {exception}")


# Pour eviter l'execution automatique lors d'un import en module.
if __name__ == "__main__":
    # Pour lancer le pipeline complet uniquement en execution directe.
    main()
