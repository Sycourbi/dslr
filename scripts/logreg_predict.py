# Pour exposer un contrat CLI stable sans dependances externes.
import argparse

# Pour relire exactement le format des parametres serialises au train.
import json

# Pour conserver les calculs vectorises deja utilises au train.
import numpy as np

# Pour garder une lecture et une ecriture CSV robustes et explicites.
import pandas as pd

try:
    from analysis_log_predict import AnalysisPredictLogger
except ImportError:
    from scripts.analysis_log_predict import AnalysisPredictLogger


# Ce point d'entree isole le contrat des arguments de prediction.
def parse_command_line_arguments():
    """
    Definit le contrat CLI de prediction.

    Cette interface impose explicitement le couple
    dataset + parametres appris pour eviter les executions ambigues
    (ex: mauvais fichier de poids applique au mauvais dataset).

    Returns:
        Namespace: contient dataset_csv_path,
        trained_parameter_json_file_path, output_csv_path
        et enable_analysis_log.
    """
    # Pour centraliser toutes les erreurs de saisie dans argparse.
    argument_parser = argparse.ArgumentParser(
        description=(
            "Predit la maison de chaque eleve a partir d'un "
            "dataset et d'un fichier de parametres"
        )
    )
    # Pour rendre obligatoire le fichier des observations a predire.
    argument_parser.add_argument(
        "dataset_csv_path",
        help="Chemin vers le fichier dataset_test.csv (sans colonne House)",
    )
    # Pour verrouiller le fichier de poids associe a l'entrainement.
    argument_parser.add_argument(
        "trained_parameter_json_file_path",
        help=(
            "Fichier JSON contenant les parametres appris "
            "(creer par logreg_train.py)"
        ),
    )
    # Pour laisser le chemin de sortie configurable sans casser le defaut.
    argument_parser.add_argument(
        "--out",
        "-o",
        dest="output_csv_path",
        default="houses.csv",
        help="Fichier de sortie contenant les predictions",
    )
    # Pour activer les logs d'analyse sans polluer l'execution standard.
    argument_parser.add_argument(
        "--analysis-log",
        dest="enable_analysis_log",
        action="store_true",
        help="Active les logs detailles d'analyse de la prediction",
    )
    # Pour fournir au reste du flux des arguments deja valides.
    return argument_parser.parse_args()


# Cette fonction rehydrate le bundle de parametres appris.
def load_house_classifier_parameters(trained_parameter_json_file_path):
    """
    Recharge les parametres produits par l'etape d'entrainement.

    Cette lecture preserve le contrat train/predict:
    meme ordre de variables, meme normalisation et meme mapping de classes.

    Args:
        trained_parameter_json_file_path (str): chemin vers le fichier JSON

    Returns:
        house_discipline_weights_with_bias (numpy.ndarray):
            poids appris (disciplines+bias)
        average_discipline_scores (list[float]):
            moyenne discipline par discipline
        discipline_standard_deviations (list[float]):
            ecart-type discipline par discipline
        house_name_by_code (dict[int, str]): nom des differentes maisons
        discipline_names (list[str]): noms des matieres
    """
    # Pour lire la source officielle des parametres sauvegardes au train.
    with open(trained_parameter_json_file_path, "r") as parameter_file:
        # Pour reconstruire le dictionnaire serialise sans reinterpretation.
        trained_parameter_bundle = json.load(parameter_file)

    # Pour retrouver les poids dans un format numerique exploitable par numpy.
    house_discipline_weights_with_bias = np.array(
        trained_parameter_bundle["thetas"]
    )
    # Pour reappliquer la meme moyenne que pendant l'entrainement.
    average_discipline_scores = trained_parameter_bundle["mu"]
    # Pour reappliquer la meme echelle que pendant l'entrainement.
    discipline_standard_deviations = trained_parameter_bundle["sigma"]
    # Pour aligner les cles sur les codes entiers sortis par argmax.
    house_name_by_code = {
        int(house_code_text): house_name
        for house_code_text, house_name in trained_parameter_bundle[
            "inv_house_map"
        ].items()
    }
    # Pour conserver l'ordre de variables attendu par les coefficients appris.
    discipline_names = trained_parameter_bundle["features"]

    # Pour renvoyer un paquet de donnees directement consommable par main.
    return (
        house_discipline_weights_with_bias,
        average_discipline_scores,
        discipline_standard_deviations,
        house_name_by_code,
        discipline_names,
    )


# Cette fonction derive les colonnes de features a partir d'un DataFrame.
def get_discipline_names(dataset):
    """
    Extrait des colonnes numeriques utilisables comme variables de modele.

    Cette fonction preserve une convention de schema:
    les colonnes techniques ou cibles ne doivent jamais devenir des features.

    Args:
        dataset (pandas.DataFrame): donnees completes du fichier.

    Returns:
        list[str]:
            noms de disciplines, excluant 'Index' et 'Hogwarts House'.
    """
    # Pour construire une liste stable de colonnes compatibles avec le modele.
    discipline_names = []
    # Pour inspecter toutes les colonnes candidates du dataset courant.
    for discipline_name in dataset.columns:
        # Pour identifier rapidement les types numeriques pandas via kind.
        discipline_kind = dataset[discipline_name].dtype.kind
        # Pour garder uniquement les colonnes entieres ou flottantes.
        if discipline_kind in ("i", "f"):
            # Pour ajouter cette colonne a la liste des features candidates.
            discipline_names.append(discipline_name)

    # Pour eviter qu'un identifiant technique entre dans le modele.
    if "Index" in discipline_names:
        # C'est ici qu'il faut agir si la colonne d'identifiant change de nom.
        discipline_names.remove("Index")
    # Pour eviter d'utiliser la cible comme variable d'entree.
    if "Hogwarts House" in discipline_names:
        # C'est ici qu'il faut agir si la colonne cible change de nom.
        discipline_names.remove("Hogwarts House")

    # Pour retourner une liste de features exploitable par le predictif.
    return discipline_names


# Cette fonction charge le CSV de test et valide son schema minimal.
def load_observations(dataset_csv_path, discipline_names):
    """
    Charge le dataset de prediction et verrouille son schema minimal.

    Args:
        dataset_csv_path (str): chemin du fichier CSV
        discipline_names (list[str]): disciplines attendues depuis weights.json

    Returns:
        index_list_of_students (list[int]): liste des index du dataset
        students_discipline_scores (pandas.DataFrame):
            disciplines dans le bon ordre
    """
    # Pour materialiser les donnees sources dans un DataFrame unique.
    raw_students_dataset = pd.read_csv(dataset_csv_path)

    # Pour garantir la presence de l'identifiant exporte en sortie.
    if "Index" not in raw_students_dataset.columns:
        # Pour signaler explicitement quel contrat de schema est viole.
        raise ValueError(
            "La colonne 'Index' est manquante dans le fichier CSV."
        )

    # Pour lister les features attendues mais absentes du CSV fourni.
    missing_discipline_names = [
        discipline_name
        for discipline_name in discipline_names
        if discipline_name not in raw_students_dataset.columns
    ]
    # Pour interrompre le flux avant un calcul incoherent de scores.
    if missing_discipline_names:
        # Pour fournir un diagnostic actionnable au mainteneur.
        raise ValueError(
            "Disciplines manquantes dans le fichier CSV: "
            f"{missing_discipline_names}"
        )

    # Pour conserver l'ordre original des eleves dans le fichier exporte.
    index_list_of_students = raw_students_dataset["Index"].tolist()
    # Pour figer l'ordre des colonnes selon celui appris au train.
    students_discipline_scores = raw_students_dataset[discipline_names].copy()

    # Pour fournir un couple identifiant + matrice de features
    # prete a normaliser.
    return index_list_of_students, students_discipline_scores


# Cette fonction applique la normalisation du train sur les donnees de test.
def standardize_discipline_scores(
    students_discipline_scores,
    average_discipline_scores,
    discipline_standard_deviations,
):
    """
    Applique la normalisation du train sur les observations a predire.

    Precondition:
        les statistiques proviennent du meme modele que les poids charges.
    Postcondition:
        le vecteur de chaque eleve est dans le meme espace numerique
        que celui vu pendant l'entrainement.

    Args:
        students_discipline_scores (pandas.DataFrame): notes brutes du dataset
        average_discipline_scores (list[float]):
            moyenne des matieres calculee au train
        discipline_standard_deviations (list[float]):
            ecart-type calcule au train

    Returns:
        numpy.ndarray: tableau normalise
    """
    # Pour garantir un type float coherent dans toutes les operations suivantes.
    average_discipline_scores = np.array(average_discipline_scores, dtype=float)
    # Pour garantir un type float coherent dans toutes les operations suivantes.
    discipline_standard_deviations = np.array(
        discipline_standard_deviations, dtype=float
    )

    # Pour eviter une division par zero sur une matiere sans variance.
    discipline_standard_deviations_without_zero = np.where(
        discipline_standard_deviations == 0, 1.0, discipline_standard_deviations
    )

    # Pour pouvoir imputer et convertir sans muter l'objet d'entree.
    students_discipline_scores_with_missing_scores = (
        students_discipline_scores.astype(float).copy()
    )

    # Pour appliquer l'imputation colonne par colonne en gardant le meme ordre.
    for discipline_index, discipline_name in enumerate(
        students_discipline_scores_with_missing_scores.columns
    ):
        # Pour reutiliser la reference du train et eviter une fuite test.
        students_discipline_scores_with_missing_scores[
            discipline_name
        ] = students_discipline_scores_with_missing_scores[
            discipline_name
        ].fillna(
            average_discipline_scores[discipline_index]
        )

    # Pour passer a une representation matricielle efficace pour numpy.
    discipline_scores_by_student = (
        students_discipline_scores_with_missing_scores.to_numpy(dtype=float)
    )

    # Pour placer chaque feature sur la meme echelle que le modele appris.
    standardized_students_discipline_scores = (
        discipline_scores_by_student - average_discipline_scores
    ) / discipline_standard_deviations_without_zero
    # Pour renvoyer la matrice standardisee attendue par la prediction.
    return standardized_students_discipline_scores


# Pour isoler le log des logits avant protection numerique.
def log_prediction_house_scores_before_clip(
    analysis_predict_logger,
    students_discipline_scores_with_bias,
    house_discipline_weights_with_bias,
    house_scores_before_value_limit_for_all_students,
):
    """
    Emet le log des scores bruts du modele.

    Args:
        analysis_predict_logger (AnalysisPredictLogger): logger conditionnel.
        students_discipline_scores_with_bias (numpy.ndarray): X avec biais.
        house_discipline_weights_with_bias (numpy.ndarray): poids appris.
        house_scores_before_value_limit_for_all_students (numpy.ndarray):
            logits bruts.
    """
    (
        analysis_predict_logger
        .log_house_scores_before_value_limit_for_all_students(
        students_discipline_scores_with_bias,
        house_discipline_weights_with_bias,
        house_scores_before_value_limit_for_all_students,
    )
    )


# Pour isoler le log des logits apres clipping numerique.
def log_prediction_house_scores_after_clip(
    analysis_predict_logger,
    house_scores_before_value_limit_for_all_students,
    house_scores_after_value_limit_for_all_students,
):
    """
    Emet le log des scores apres clipping.

    Args:
        analysis_predict_logger (AnalysisPredictLogger): logger conditionnel.
        house_scores_before_value_limit_for_all_students (numpy.ndarray):
            logits avant clipping.
        house_scores_after_value_limit_for_all_students (numpy.ndarray):
            logits apres clipping.
    """
    analysis_predict_logger.log_house_scores_after_value_limit_for_all_students(
        house_scores_before_value_limit_for_all_students,
        house_scores_after_value_limit_for_all_students,
    )


# Pour isoler le log des probabilites one-vs-rest.
def log_prediction_house_probabilities(
    analysis_predict_logger,
    house_scores_after_value_limit_for_all_students,
    house_probability_scores_for_all_students,
):
    """
    Emet le log des probabilites predites.

    Args:
        analysis_predict_logger (AnalysisPredictLogger): logger conditionnel.
        house_scores_after_value_limit_for_all_students (numpy.ndarray):
            logits clip.
        house_probability_scores_for_all_students (numpy.ndarray):
            probabilites finales.
    """
    analysis_predict_logger.log_house_probability_scores_for_all_students(
        house_scores_after_value_limit_for_all_students,
        house_probability_scores_for_all_students,
    )


# Pour isoler le log des codes de classes predits.
def log_prediction_house_codes(
    analysis_predict_logger,
    house_probability_scores_for_all_students,
    predicted_house_codes_for_all_students,
):
    """
    Emet le log des classes predites en codes entiers.

    Args:
        analysis_predict_logger (AnalysisPredictLogger): logger conditionnel.
        house_probability_scores_for_all_students (numpy.ndarray):
            probabilites par classe.
        predicted_house_codes_for_all_students (numpy.ndarray):
            code argmax par eleve.
    """
    analysis_predict_logger.log_predicted_house_codes_for_all_students(
        house_probability_scores_for_all_students,
        predicted_house_codes_for_all_students,
    )


# Pour isoler le log des labels finaux apres mapping code->nom.
def log_prediction_house_names(
    analysis_predict_logger,
    house_name_by_code,
    predicted_house_codes_for_all_students,
    predicted_house_names_for_all_students,
):
    """
    Emet le log des labels de maisons finaux.

    Args:
        analysis_predict_logger (AnalysisPredictLogger): logger conditionnel.
        house_name_by_code (dict[int, str]): table de mapping code->nom.
        predicted_house_codes_for_all_students (numpy.ndarray): codes predits.
        predicted_house_names_for_all_students (list[str]): noms predits.
    """
    analysis_predict_logger.log_predicted_house_names_for_all_students(
        house_name_by_code,
        predicted_house_codes_for_all_students,
        predicted_house_names_for_all_students,
    )


# Cette fonction applique le modele one-vs-rest a tous les eleves.
def predict_house_names(
    students_discipline_scores_with_bias,
    house_discipline_weights_with_bias,
    house_name_by_code,
    analysis_predict_logger,
):
    """
    Evalue le modele one-vs-rest et retourne la maison la plus probable.

    Args:
        students_discipline_scores_with_bias (numpy.ndarray):
            notes normalisees + bias
        house_discipline_weights_with_bias (numpy.ndarray):
            poids appris (disciplines+bias)
        house_name_by_code (dict[int, str]): nom des differentes maisons

    Returns:
        list[str]: maison predite pour chaque eleve
    """
    # Pour obtenir les logits de toutes les maisons pour chaque eleve.
    house_scores_before_value_limit_for_all_students = (
        students_discipline_scores_with_bias.dot(
            house_discipline_weights_with_bias.T
        )
    )
    log_prediction_house_scores_before_clip(
        analysis_predict_logger,
        students_discipline_scores_with_bias,
        house_discipline_weights_with_bias,
        house_scores_before_value_limit_for_all_students,
    )

    # Pour proteger np.exp des overflows sur des valeurs extremes.
    house_scores_after_value_limit_for_all_students = np.clip(
        house_scores_before_value_limit_for_all_students, -500, 500
    )
    log_prediction_house_scores_after_clip(
        analysis_predict_logger,
        house_scores_before_value_limit_for_all_students,
        house_scores_after_value_limit_for_all_students,
    )

    # Pour convertir les logits en scores comparables dans [0, 1].
    house_probability_scores_for_all_students = 1 / (
        1 + np.exp(-house_scores_after_value_limit_for_all_students)
    )
    log_prediction_house_probabilities(
        analysis_predict_logger,
        house_scores_after_value_limit_for_all_students,
        house_probability_scores_for_all_students,
    )
    # Pour forcer une classe unique par eleve dans le schema one-vs-rest.
    predicted_house_codes_for_all_students = np.argmax(
        house_probability_scores_for_all_students, axis=1
    )
    log_prediction_house_codes(
        analysis_predict_logger,
        house_probability_scores_for_all_students,
        predicted_house_codes_for_all_students,
    )

    # Pour traduire chaque code numerique en nom de maison exportable.
    predicted_house_names_for_all_students = [
        house_name_by_code[int(house_code)]
        for house_code in predicted_house_codes_for_all_students
    ]
    log_prediction_house_names(
        analysis_predict_logger,
        house_name_by_code,
        predicted_house_codes_for_all_students,
        predicted_house_names_for_all_students,
    )

    # Pour renvoyer une sequence de labels prete a etre exportee.
    return predicted_house_names_for_all_students


# Ce point d'entree orchestre tout le flux predictif de bout en bout.
def main():
    """
    Orchestre la prediction puis persiste le resultat au format CSV attendu.

    Effet de bord notable:
        ecrit un fichier contenant exactement deux colonnes:
        Index et Hogwarts House.
    """
    # Pour centraliser la gestion d'erreur et garder un message unique.
    try:
        # Pour recuperer les chemins d'entree et de sortie valides.
        cli_arguments = parse_command_line_arguments()
        analysis_predict_logger = AnalysisPredictLogger(
            cli_arguments.enable_analysis_log
        )
        # Pour recharger poids, normalisation, mapping et schema de features.
        (
            house_discipline_weights_with_bias,
            average_discipline_scores,
            discipline_standard_deviations,
            house_name_by_code,
            discipline_names,
        ) = load_house_classifier_parameters(
            cli_arguments.trained_parameter_json_file_path
        )

        # Pour charger le dataset de test avec le schema attendu par le modele.
        index_list_of_students, students_discipline_scores = load_observations(
            cli_arguments.dataset_csv_path, discipline_names
        )
        analysis_predict_logger.log_students_discipline_scores(
            students_discipline_scores
        )
        # Pour aligner les observations de test sur l'espace du train.
        standardized_students_discipline_scores = standardize_discipline_scores(
            students_discipline_scores,
            average_discipline_scores,
            discipline_standard_deviations,
        )
        analysis_predict_logger.log_standardized_students_discipline_scores(
            students_discipline_scores,
            average_discipline_scores,
            discipline_standard_deviations,
            standardized_students_discipline_scores,
        )
        # Pour dimensionner correctement la colonne de biais ajoutee ensuite.
        student_count = standardized_students_discipline_scores.shape[0]
        # Pour reproduire la convention train avec biais en colonne 0.
        students_discipline_scores_with_bias = np.hstack(
            [
                np.ones((student_count, 1)),
                standardized_students_discipline_scores,
            ]
        )
        analysis_predict_logger.log_students_discipline_scores_with_bias(
            student_count,
            standardized_students_discipline_scores,
            students_discipline_scores_with_bias,
        )

        # Pour produire les labels finaux a partir des donnees preparees.
        predicted_house_names_for_all_students = predict_house_names(
            students_discipline_scores_with_bias,
            house_discipline_weights_with_bias,
            house_name_by_code,
            analysis_predict_logger,
        )

        # Pour construire le format exact attendu par les consignes du projet.
        prediction_output = pd.DataFrame(
            {
                "Index": index_list_of_students,
                "Hogwarts House": predicted_house_names_for_all_students,
            }
        )
        # Pour persister un CSV directement exploitable par l'evaluation.
        prediction_output.to_csv(cli_arguments.output_csv_path, index=False)
        # Pour donner un signal clair de succes et de chemin de sortie.
        print(
            "→ Fichier de prediction enregistre dans "
            f"{cli_arguments.output_csv_path}"
        )

    # Pour capter tout echec et conserver une sortie CLI explicite.
    except Exception as exception:
        # Pour remonter la cause immediate sans stack trace verbeuse.
        print(f"Une erreur est survenue : {exception}")


# Pour garantir l'execution uniquement en mode script principal.
if __name__ == "__main__":
    # Pour declencher le flux complet lorsque le fichier est lance.
    main()
