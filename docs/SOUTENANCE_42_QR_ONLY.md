# Soutenance DSLR - Questions / Reponses

## Questions communes

## Question 1
Comment fonctionne la regression logistique dans ce projet ?

### Reponse
Le code calcule d'abord un score lineaire a partir des notes normalisees et des
poids. Ce score passe dans une sigmoid pour produire une probabilite entre 0
et 1. Ensuite l'entrainement compare cette probabilite a une cible binaire,
calcule un gradient et met a jour les poids. En prediction, on refait le meme
score avec les poids appris puis on choisit la maison la plus probable.

## Question 2
En quoi la regression logistique se compare-t-elle a la regression lineaire ?

### Reponse
Les deux commencent par une combinaison lineaire des features et des poids.
La difference, ici, est que la regression logistique transforme ce score avec
une sigmoid pour obtenir une probabilite exploitable en classification. Une
regression lineaire renverrait une valeur continue non bornee. Donc la base
algebrique est proche, mais l'objectif final n'est pas le meme.

## Question 3
Pourquoi normaliser les donnees dans ce projet ?

### Reponse
La normalisation remet toutes les matieres sur une echelle comparable avant le
calcul du gradient. Sans elle, une colonne numeriquement plus grande peut
prendre trop de poids dans l'apprentissage. Ici, elle sert aussi a figer un
repere commun entre `train` et `predict`, parce que le JSON sauvegarde `mu`
et `sigma` et oblige `predict` a reutiliser exactement la meme normalisation.

## Question 4
Qu'est-ce que la methode un-contre-tous dans ton code ?

### Reponse
Le code entraine un classifieur binaire par maison. Pour une maison donnee, la
cible vaut 1 si l'eleve est dans cette maison et 0 sinon. A la fin, on a donc
une ligne de poids par maison. En prediction, on calcule un score pour toutes
les maisons et on prend celle dont la probabilite est la plus haute.

## Question 5
Si je te demande une preuve concrete que `train` et `predict` sont coherents entre eux, tu montres quoi ?

### Reponse
Je montre que `train` ecrit `thetas`, `mu`, `sigma`, `features` et les
mappings de maisons dans le JSON, puis que `predict` relit exactement ces
champs. C'est ca qui prouve le contrat commun entre les deux scripts. Les
poids seuls ne suffisent pas : il faut aussi l'ordre des colonnes et la
normalisation du train.

## Question 6
Le sujet parle d'un minimum a 98% avec `evaluate.py`. Tu peux le prouver ici ?

### Reponse
Non, pas integralement ici. Le PDF mentionne `evaluate.py` et
`dataset_truth.csv`, mais ces fichiers ne sont pas presents dans ce depot. Je
peux demontrer que `train` cree bien un JSON et que `predict` cree bien un CSV
`houses.csv`, mais je ne peux pas prouver localement le score de 98% sans les
artefacts d'evaluation.

## Question 7
Quel est le flux global de `logreg_train.py` ?

### Reponse
Il parse les arguments, charge le CSV, identifie les features numeriques,
supprime les lignes incompletes, encode les maisons en entiers, normalise les
notes, ajoute une colonne de biais, entraine un modele one-vs-all par descente
de gradient batch, puis ecrit un JSON avec les poids et les stats de
normalisation.

## Question 8
Quel est le flux global de `logreg_predict.py` ?

### Reponse
Il parse les arguments, recharge le JSON produit par `train`, verifie le schema
minimal du CSV de test, remet les colonnes dans l'ordre appris, normalise avec
`mu` et `sigma` du train, ajoute la colonne de biais, calcule les probabilites
par maison, prend l'argmax, puis ecrit un fichier `houses.csv` avec `Index` et
`Hogwarts House`.

## Questions sur `logreg_train.py`

## Question 9
Pourquoi `pd.read_csv(...)` est appele au debut de `load_and_prepare_dataset` ?

### Reponse
Parce que tout le reste depend du DataFrame charge. Tant que le CSV n'est pas
materialise, on ne peut ni detecter les colonnes, ni filtrer les lignes, ni
encoder la cible. C'est le point d'entree reel des donnees, pas juste un
appel utilitaire.

## Question 10
Que fait exactement `get_discipline_names(dataset)` ?

### Reponse
Cette fonction parcourt les colonnes du DataFrame, garde celles dont le type
est numerique, puis retire `Index`. Elle fixe donc le schema reel des
features du modele. En pratique, elle decide quelles colonnes seront
apprises, sauvegardees dans le JSON, puis exigees plus tard par `predict`.

## Question 11
Pourquoi retirer `Index` dans `get_discipline_names` ?

### Reponse
Parce que `Index` est un identifiant technique, pas une note. Si je le garde,
le modele peut apprendre un faux signal base sur le numero de ligne et pas sur
les disciplines. Le retirer preserve le sens metier des features.

## Question 12
Pourquoi faire `dropna(subset=["Hogwarts House"] + discipline_names)` dans le train ?

### Reponse
Le train a besoin d'une cible et d'un vecteur de features complet pour chaque
ligne. Si une note ou la maison manque, le gradient ne correspond plus a un
exemple propre. Ici, le choix est de supprimer ces lignes pour garder un batch
coherent, plutot que d'imputer pendant l'apprentissage.

## Question 13
Pourquoi `reset_index(drop=True)` juste apres la selection des colonnes ?

### Reponse
`dropna` conserve les index d'origine, donc on peut se retrouver avec des
trous dans l'index pandas. `reset_index(drop=True)` recompacte le DataFrame et
supprime l'ancien index au lieu de le garder comme colonne parasite. Ca evite
un decalage implicite entre lignes pandas et vecteurs numpy ensuite.

## Question 14
Pourquoi imposer `house_names = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]` au lieu de deduire l'ordre automatiquement ?

### Reponse
Parce que le code veut un ordre canonique stable des classes sur tout le
projet. Si je laissais l'ordre dependre du dataset, les codes de maisons
pourraient varier d'un entrainement a l'autre. Ici, le mapping est fige, donc
le sens des lignes de `thetas` reste stable.

## Question 15
Pourquoi faire `.map(house_code_by_name).to_numpy(dtype=int)` sur la colonne `Hogwarts House` ?

### Reponse
`.map(...)` transforme les noms de maisons en codes entiers selon le mapping
canonique. `.to_numpy(dtype=int)` sort ensuite du monde pandas pour produire un
vecteur numerique exploitable par numpy. Le resultat, c'est une cible `y`
compacte, stable et directement utilisable dans le one-vs-all.

## Question 16
Pourquoi `to_numpy(dtype=float)` dans `standardize_disciplines_scores` ?

### Reponse
Parce que la suite du calcul est entierement vectorisee avec numpy. Le passage
explicite en float verrouille un type numerique coherent pour `mean`, `std`,
la normalisation, la sigmoid et le gradient. Ca evite des conversions implicites
ou des surprises de type au milieu du pipeline.

## Question 17
Explique exactement `mean(axis=0)` et `std(axis=0, ddof=0)`.

### Reponse
`axis=0` veut dire qu'on calcule une statistique par colonne, donc par
matiere, et pas par eleve. C'est bien ce qu'il faut pour normaliser les
features. `ddof=0` prend l'ecart-type calcule directement sur les donnees du
train, sans correction d'echantillon. Donc `mu` et `sigma` ont une valeur par
feature, dans le meme ordre que les colonnes.

## Question 18
Explique la formule de normalisation `(X - mu) / sigma` dans ce code.

### Reponse
`X` est la matrice des notes, `mu` la moyenne par matiere et `sigma` l'ecart-
type par matiere. On recentre d'abord chaque colonne avec `X - mu`, puis on la
remet sur une echelle comparable avec la division par `sigma`. En shape,
`X` vaut `n_eleves x n_features`, alors que `mu` et `sigma` valent
`n_features`, donc numpy diffuse ces vecteurs sur toutes les lignes.

## Question 19
Que se passe-t-il si une colonne a un ecart-type nul dans `train` ?

### Reponse
Ici, le train ne protege pas ce cas. Si une discipline est strictement
constante, la division par zero peut produire des `inf` ou des `nan` dans la
normalisation. Le predict a une protection sur `sigma == 0`, mais le train ne
l'a pas. Donc c'est une vraie limite actuelle du script.

## Question 20
Pourquoi `compute_sigmoid(z)` retourne `1 / (1 + exp(-z))` ?

### Reponse
Parce que cette formule transforme un score lineaire non borne en probabilite
entre 0 et 1. Dans ce code, `z` est le logit calcule a partir des notes
normalisees et des poids. La sigmoid sert donc de passerelle entre l'algebre
lineaire du modele et une sortie interpretable en classification.

## Question 21
Pourquoi `np.unique(assigned_house_codes_for_students)` dans le train ?

### Reponse
Parce que le code doit savoir pour quelles classes il va entrainer un
classifieur binaire. `np.unique(...)` liste les codes effectivement presents
parmi les eleves du dataset. Ensuite la boucle one-vs-all se fait sur ces
classes presentes.

## Question 22
Quelle est la limite de cette boucle sur `unique_house_codes` ?

### Reponse
La matrice des poids est allouee avec `len(unique_house_codes)` lignes, puis le
code ecrit a l'index numerique `current_house_code`. Donc si une maison manque
au train et que les codes presents ne sont pas compacts depuis 0, l'ecriture
peut sortir de la matrice. Le code suppose donc implicitement que toutes les
maisons existent dans le dataset d'entrainement.

## Question 23
Pourquoi initialiser les poids avec `np.zeros(...)` ?

### Reponse
Ici, le choix est la simplicite et le determinisme. On part d'un point neutre,
facile a auditer, puis les gradients differencient les classifieurs maison par
maison. Ce n'est pas une initialisation sophistiquee, mais pour cette
implementation logistique one-vs-all elle est suffisante et lisible.

## Question 24
Quelle est la shape exacte de `students_disciplines_scores_with_bias` juste avant l'entrainement ?

### Reponse
La matrice standardisee a la shape `n_eleves x n_features`. Le code lui ajoute
une colonne de `1`, donc `students_disciplines_scores_with_bias` a la shape
`n_eleves x (n_features + 1)`. La premiere colonne est le biais, les autres
sont les disciplines normalisees dans l'ordre sauvegarde dans `features`.

## Question 25
Pourquoi ajouter une colonne de `1` avec `np.hstack(...)` ?

### Reponse
Cette colonne represente le biais, donc l'intercept du modele. Sans elle, le
modele ne pourrait apprendre qu'un poids par discipline, mais pas un decalage
global independant des notes. `np.hstack(...)` construit donc une matrice ou le
biais est traite comme un coefficient appris exactement comme les autres.

## Question 26
Explique la formule `students_disciplines_scores_with_bias.dot(current_house_weights)`.

### Reponse
`X` a la shape `n_eleves x (n_features + 1)` et `w` a la shape
`(n_features + 1)`. Le produit est donc valide et retourne un vecteur de
shape `n_eleves`, un score par eleve pour la maison courante. Ce n'est pas une
multiplication element par element : c'est la combinaison lineaire complete des
features et du biais.

## Question 27
Explique `prediction_error_by_students = prediction - cible`.

### Reponse
La prediction est une probabilite par eleve, et la cible binaire vaut 0 ou 1
pour cette maison. La difference garde a la fois le signe et l'amplitude de
l'erreur. Si la proba est trop haute pour un negatif, l'erreur est positive ;
si elle est trop basse pour un positif, l'erreur est negative. C'est ce
vecteur d'erreur qui alimente ensuite le gradient.

## Question 28
Pourquoi `students_disciplines_scores_with_bias.T.dot(prediction_error_by_students)` ?

### Reponse
Avant transposee, `X` a la shape `n_eleves x (n_features + 1)`. Apres
transposee, `X.T` vaut `(n_features + 1) x n_eleves`, ce qui le rend
compatible avec le vecteur d'erreur de shape `n_eleves`. Le resultat a donc la
shape `(n_features + 1)` et donne une somme d'erreurs ponderees par
coefficient. Si j'enleve `.T`, le produit n'a plus le bon sens algebrique.

## Question 29
Pourquoi diviser par `students_count` dans le gradient ?

### Reponse
Parce qu'ici on calcule un gradient moyen sur tout le batch, pas une somme
brute. Diviser par le nombre d'eleves rend l'echelle du gradient moins
dependante de la taille du dataset. Ca stabilise aussi le sens du learning
rate, parce que le pas ne grossit pas juste parce qu'on a plus de lignes.

## Question 30
Explique la formule d'update `current_house_weights -= learning_rate * gradient`.

### Reponse
Le gradient indique la direction dans laquelle la fonction augmente. Comme on
veut reduire l'erreur, on se deplace dans la direction opposee, d'ou le signe
`-=`. `learning_rate` regle seulement la taille du pas. Si on inverse le signe,
on monte au lieu de descendre.

## Question 31
Pourquoi le code stocke `thetas`, `mu`, `sigma`, `features`, `house_map` et `inv_house_map` dans un JSON ?

### Reponse
Parce que `predict` doit reconstruire exactement le meme contrat que le train.
`thetas` donnent les coefficients, `mu` et `sigma` la normalisation, `features`
l'ordre des colonnes, et les mappings garantissent le lien entre codes et noms
de maisons. Sans cet ensemble, la prediction ne serait plus reproductible.

## Question 32
Si je retire toute la partie `analysis_logger`, qu'est-ce que je perds ?

### Reponse
Je ne change pas la logique numerique du modele, mais je perds la possibilite
de demonstrer l'intermediaire pendant la soutenance. Ces appels sont la pour
rendre visibles les scores, erreurs, gradients et poids a chaque etape. Donc
ce n'est pas vital pour l'algorithme, mais c'est utile pour la preuve et le
debogage.

## Question 33
Pourquoi `main()` attrape `Exception` puis fait juste `print(...)` ?

### Reponse
Le but est d'eviter un crash brut et de garder un message stable cote CLI.
Mais la contrepartie, c'est que le script ne renvoie pas explicitement un code
shell non nul. Donc le message d'erreur est lisible, mais la signalisation
systeme d'echec reste faible.

## Questions sur `logreg_predict.py`

## Question 34
Pourquoi `json.load(...)` puis `np.array(...)` sur `thetas` ?

### Reponse
Le JSON redonne des listes Python, mais les calculs du predict sont vectorises
avec numpy. Il faut donc retransformer `thetas` en matrice numpy avant les
produits matriciels. Sinon, on ne retrouve pas le contrat de forme attendu
par le calcul des scores.

## Question 35
Pourquoi convertir les cles de `inv_house_map` avec `int(house_code_text)` ?

### Reponse
Parce qu'en JSON les cles de dictionnaire reviennent en texte. Or `argmax`
retourne des codes numeriques, pas des chaines. Cette comprehension reconstruit
donc un mapping `int -> str` coherent avec ce que produit le modele au moment
de la prediction.

## Question 36
Que verifie exactement `load_observations()` ?

### Reponse
La fonction charge le CSV de test, verifie d'abord la presence de `Index`,
puis controle que toutes les disciplines attendues par le JSON sont presentes.
Ensuite elle extrait la liste des index et les colonnes de notes dans l'ordre
appris au train. Donc elle verrouille le schema minimal avant tout calcul.

## Question 37
Pourquoi `raw_students_dataset[discipline_names].copy()` ?

### Reponse
La selection par `discipline_names` force l'ordre exact appris au train. Et
`.copy()` evite de modifier indirectement le DataFrame source plus tard, par
exemple pendant l'imputation. Ici, on preserve a la fois l'alignement des
colonnes et l'absence d'effet de bord sur l'objet d'origine.

## Question 38
Pourquoi `predict` relit `mu`, `sigma` et `features` au lieu de les recalculer sur le test ?

### Reponse
Parce que les poids appris n'ont de sens que dans l'espace numerique du train.
Si je renormalise sur le test, je change le repere dans lequel les coefficients
ont ete appris. `predict` doit donc reutiliser exactement les stats et l'ordre
de colonnes du train, pas en inventer de nouveaux.

## Question 39
Pourquoi `np.where(discipline_standard_deviations == 0, 1.0, discipline_standard_deviations)` ?

### Reponse
Parce qu'un `sigma` nul ferait une division par zero au moment de normaliser.
Le code remplace uniquement ces cas-la par `1.0` pour garder la prediction
possible. Une telle colonne ne sera pas vraiment informative, mais au moins le
script ne casse pas numeriquement.

## Question 40
Pourquoi faire `astype(float).copy()` avant les `fillna(...)` ?

### Reponse
`astype(float)` garantit un type numerique coherent pour toute la suite des
calculs. `.copy()` isole ensuite une copie de travail pour ne pas muter le
DataFrame extrait auparavant. L'idee, c'est de preparer une structure stable
avant l'imputation puis la conversion finale en numpy.

## Question 41
Pourquoi faire les `fillna(...)` dans une boucle par colonne ?

### Reponse
Parce que chaque discipline doit etre imputee avec sa propre moyenne du train,
pas avec une valeur globale. La boucle suit exactement l'ordre des colonnes et
associe chaque `discipline_name` a son `mu` via `discipline_index`. Donc
l'imputation reste alignee sur le schema appris.

## Question 42
Pourquoi `predict` impute les valeurs manquantes au lieu de faire `dropna(...)` comme `train` ?

### Reponse
Parce qu'en prediction il faut rendre une reponse pour chaque `Index` fourni
dans le CSV test. Si je supprime des lignes, je casse l'alignement entre le
fichier d'entree et le fichier de sortie. Le choix ici est donc : suppression
au train pour apprendre proprement, imputation au predict pour conserver tous
les eleves.

## Question 43
Explique la formule de normalisation dans `predict` et ses shapes exactes.

### Reponse
La matrice des notes du test a la shape `n_eleves x n_features`. `mu` et
`sigma_safe` ont chacun la shape `n_features`. La formule `(X_test - mu) /
sigma_safe` utilise le broadcasting de numpy pour appliquer la meme statistique
a chaque ligne. Le resultat garde la shape `n_eleves x n_features`, donc il
reste compatible avec l'ajout du biais juste apres.

## Question 44
Quelle est la shape de `students_discipline_scores_with_bias` dans `predict` ?

### Reponse
Apres normalisation, on a `n_eleves x n_features`. Le code ajoute ensuite une
colonne de `1`, donc on obtient `n_eleves x (n_features + 1)`. Cette shape
doit correspondre exactement a celle attendue par chaque ligne de `thetas`,
sinon le produit matriciel suivant devient invalide.

## Question 45
Explique le produit matriciel `students_discipline_scores_with_bias.dot(house_discipline_weights_with_bias.T)`.

### Reponse
`X_test_bias` a la shape `n_eleves x (n_features + 1)`. `thetas` a la shape
`n_maisons x (n_features + 1)`, donc `thetas.T` vaut
`(n_features + 1) x n_maisons`. Le produit est donc valide et retourne une
matrice `n_eleves x n_maisons`, c'est-a-dire un score par maison pour chaque
eleve. Si j'oublie la transposee, les dimensions ne correspondent plus.

## Question 46
Pourquoi `np.clip(..., -500, 500)` avant la sigmoid ?

### Reponse
Parce que `exp` peut deborder si les logits sont trop grands en valeur
absolue. `np.clip(...)` borne donc les scores avant d'appeler la sigmoid et
protege la stabilite numerique du calcul. Ce n'est pas un detail cosmetique :
ca evite un overflow juste avant la transformation probabiliste.

## Question 47
Explique la formule `1 / (1 + np.exp(-scores))` dans `predict`.

### Reponse
Cette formule prend la matrice de logits `n_eleves x n_maisons` et la
transforme en matrice de probabilites de meme shape. Chaque case represente la
proba qu'un eleve appartienne a une maison dans le schema one-vs-all. Le but
est de comparer les maisons sur une echelle commune entre 0 et 1.

## Question 48
Pourquoi `np.argmax(house_probability_scores_for_all_students, axis=1)` ?

### Reponse
Parce qu'une ligne represente un eleve et les colonnes representent les
maisons. `axis=1` veut donc dire : prends, pour chaque eleve, l'indice de la
maison au score maximal. Si je mettais `axis=0`, je chercherais la meilleure
ligne par colonne, ce qui n'a pas de sens pour produire une prediction par
eleve.

## Question 49
Pourquoi la comprehension de liste finale sur `house_name_by_code[int(house_code)]` ?

### Reponse
Parce que `argmax` donne des codes entiers, alors que le CSV attendu doit
contenir des noms de maisons. La comprehension traduit donc chaque code en
label exportable. Elle finalise le passage du resultat numerique du modele au
format metier attendu par le sujet.

## Question 50
Pourquoi `prediction_output.to_csv(..., index=False)` ?

### Reponse
Parce que le fichier attendu doit contenir exactement les colonnes `Index` et
`Hogwarts House`, sans index pandas ajoute automatiquement. `index=False`
empeche justement pandas d'ecrire une colonne technique supplementaire. Sans
ca, le CSV final ne respecterait plus strictement le format voulu.

## Question 51
Qu'est-ce qui prouve ici que `logreg_train.py` cree bien un fichier de poids ?

### Reponse
Le `main()` construit un bundle JSON explicite puis l'ecrit avec `json.dump`
dans le chemin de sortie. Le script affiche ensuite un message de succes avec
ce chemin. Donc la preuve observable attendue en soutenance, c'est
l'execution du script suivie de l'existence et de l'ouverture de ce JSON.

## Question 52
Qu'est-ce qui prouve ici que `logreg_predict.py` cree bien `houses.csv` ?

### Reponse
Le `main()` construit un DataFrame avec exactement `Index` et
`Hogwarts House`, puis l'ecrit avec `to_csv`. Le script affiche ensuite le
chemin de sortie. Donc la preuve observable, c'est l'execution suivie de
l'ouverture du CSV et de la verification de ses deux colonnes.

## Question 53
Si l'evaluateur te dit: "Tu recites la formule, mais ta ligne fait quoi exactement ?", tu reponds quoi ?

### Reponse
Je reviens a la ligne precise et je la traduis en trois niveaux: sens metier,
sens mathematique, puis shape. Par exemple pour `X.T.dot(error)`, je dis que
metierement on agrege l'erreur de tous les eleves par coefficient, que
mathematiquement c'est un gradient, et que dimensionnellement on passe de
`(n_features+1) x n_eleves` multiplie par `n_eleves` a un resultat de
`n_features+1`, donc exactement la taille du vecteur de poids.

## Question 54
Si l'evaluateur te demande: "Ou est la preuve ?", tu montres quoi en premier ?

### Reponse
Je montre d'abord le code de la fonction, puis l'execution minimale qui produit
l'artefact attendu. Pour `train`, c'est le JSON. Pour `predict`, c'est le CSV.
Quand le point est mathematique, je montre la ligne, la shape des operands et
le resultat attendu. Si le sujet demande un score de 98%, je dis honnetement
que ce point n'est pas prouvable ici sans `evaluate.py` et `dataset_truth.csv`.

## Question 55
Quelle est aujourd'hui la faiblesse la plus attaquable de ces deux scripts ?

### Reponse
Il y en a trois principales. Le score officiel a 98% n'est pas verifiable ici
sans les fichiers d'evaluation. Le train ne protege pas le cas `sigma == 0`
lors de la normalisation. Et le `except Exception` imprime une erreur lisible,
mais sans remonter explicitement un code shell non nul.
