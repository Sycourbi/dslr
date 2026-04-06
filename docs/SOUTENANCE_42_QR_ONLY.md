# Soutenance DSLR - Questions / Reponses

## Questions communes

## Question 1
Comment fonctionne la regression logistique dans ce projet ?

### Reponse
On part d'un score lineaire calcule a partir des notes normalisees et des
poids. Ce score passe dans une sigmoid pour produire une probabilite entre 0
et 1. Ensuite le train calcule l'erreur entre la probabilite et la cible, en
deduit un gradient, puis met a jour les poids par descente de gradient batch.
Enfin, les poids et les statistiques de normalisation sont sauvegardes dans un
JSON pour etre reutilises au predict.

## Question 2
En quoi la regression logistique se compare-t-elle a la regression lineaire ?

### Reponse
Les deux partent d'une combinaison lineaire des entrees et des poids. La
difference, ici, est qu'on applique une sigmoid pour obtenir une probabilite,
puis qu'on choisit une classe. Une regression lineaire renverrait une valeur
continue non bornee. Donc la base reste lineaire, mais l'usage final est la
classification et pas la prediction d'une valeur.

## Question 3
Pourquoi normaliser les donnees dans ce projet ?

### Reponse
La normalisation remet toutes les matieres sur une echelle comparable avant
l'entrainement. Sans elle, une matiere avec de grandes valeurs peut dominer le
gradient et deformer l'apprentissage. Ici, elle sert aussi a garantir que
`predict` utilise exactement le meme repere numerique que `train`, grace a
`mu` et `sigma` sauvegardes dans le JSON.

## Question 4
Qu'est-ce que la methode un-contre-tous dans ton code ?

### Reponse
Ici, un-contre-tous veut dire qu'on entraine un classifieur binaire par
maison. Pour une maison donnee, la cible vaut 1 si l'eleve est dans cette
maison, sinon 0. On obtient donc une ligne de poids par maison. En prediction,
on calcule les scores de toutes les maisons et on retient celle dont la
probabilite est la plus haute avec `argmax`.

## Question 5
Quel est le flux complet de `logreg_train.py` du CSV jusqu'au JSON ?

### Reponse
Le script parse les arguments, charge le CSV, extrait les colonnes numeriques
utiles, filtre les lignes incompletes, encode les maisons en entiers, puis
normalise les notes. Ensuite il ajoute une colonne de biais, entraine le
one-vs-all, puis serialise dans un JSON les poids, les stats de
normalisation, les features et les mappings de classes.

## Question 6
Quel est le flux complet de `logreg_predict.py` du JSON jusqu'a `houses.csv` ?

### Reponse
Le script parse les arguments, recharge le fichier de poids, relit `thetas`,
`mu`, `sigma`, `features` et le mapping des maisons. Ensuite il charge le CSV
de test, verifie le schema, remet les colonnes dans l'ordre attendu,
normalise avec les stats du train, ajoute le biais, calcule les probabilites
par maison, choisit la meilleure avec `argmax`, puis ecrit `houses.csv`.

## Questions sur `logreg_train.py`

## Question 7
Pourquoi `pd.read_csv(...)` est appele au tout debut du train ?

### Reponse
Parce que tout le pipeline depend d'abord de la materialisation du CSV dans un
DataFrame. Tant que le fichier n'est pas charge, on ne peut ni detecter les
colonnes utiles, ni filtrer les lignes, ni encoder la cible. C'est donc une
etape d'entree obligatoire, pas juste un detail d'implementation.

## Question 8
Que fait exactement `get_discipline_names(dataset)` ?

### Reponse
Cette fonction parcourt les colonnes du DataFrame, garde uniquement celles
dont le type est numerique, puis retire `Index`. Son role est de definir le
schema reel des features du modele. En pratique, elle decide quelles colonnes
seront apprises, sauvegardees, puis attendues plus tard par `predict`.

## Question 9
Pourquoi retirer `Index` avec `remove("Index")` ?

### Reponse
Parce que `Index` est un identifiant technique, pas une information metier sur
l'eleve. Si on le garde, le modele risque d'apprendre un faux signal lie a un
numero de ligne et pas aux notes. Le retirer protege donc le sens statistique
du modele.

## Question 10
Pourquoi faire `dropna(...)` dans le train ?

### Reponse
Ici, le choix est d'entrainer uniquement sur des lignes completes pour eviter
d'apprendre sur des donnees partielles ou incoherentes. Le gradient est alors
calcule sur des exemples totalement observes. C'est un compromis simple et
coherent, meme si ce n'est pas la seule strategie possible.

## Question 11
Pourquoi `dropna(...)` utilise `subset=["Hogwarts House"] + discipline_names` ?

### Reponse
Parce qu'on veut supprimer uniquement les lignes qui manquent sur la cible ou
sur une feature effectivement utilisee par le modele. Si on ne cible pas ce
sous-ensemble, on pourrait supprimer trop de lignes pour des colonnes sans
importance, ou au contraire garder des lignes inutilisables pour le calcul.

## Question 12
Pourquoi faire `reset_index(drop=True)` apres le filtrage du train ?

### Reponse
Apres un `dropna`, les index pandas peuvent rester discontinus. Ici,
`reset_index(drop=True)` remet un index propre et dense sur les features
retenues. Le `drop=True` evite de transformer l'ancien index en nouvelle
colonne parasite. Ce n'est pas le coeur mathematique du modele, mais c'est une
protection de coherence structurelle.

## Question 13
Pourquoi creer `house_code_by_name` avec une comprehension et `enumerate` ?

### Reponse
Parce qu'il faut un mapping stable entre les noms de maisons et des codes
entiers. `enumerate` permet d'associer chaque maison a un entier dans un ordre
canonique fixe. Cet ordre est central, car il conditionne ensuite le sens des
lignes de `thetas` et le remappage en prediction.

## Question 14
Pourquoi avoir aussi `house_name_by_code` ?

### Reponse
Parce que le train travaille plus facilement avec des codes entiers, mais la
sortie finale doit rester lisible et conforme au sujet. Il faut donc garder la
traduction inverse pour revenir d'un code numerique vers un nom de maison au
moment d'ecrire le CSV de prediction.

## Question 15
Pourquoi faire `.map(...).to_numpy(dtype=int)` sur `Hogwarts House` ?

### Reponse
`.map(...)` convertit les noms des maisons en codes numeriques stables selon
le mapping defini juste avant. Ensuite `to_numpy(dtype=int)` transforme cette
serie pandas en vecteur dense d'entiers directement exploitable par numpy.
L'apprentissage et les comparaisons one-vs-all deviennent ainsi plus simples
et plus robustes que si la cible restait en texte.

## Question 16
Que fait exactement `to_numpy(dtype=float)` sur les features du train ?

### Reponse
Ca convertit le DataFrame pandas en matrice numpy de flottants. Le but est de
sortir du monde pandas pour passer a des operations vectorisees stables avec
numpy. Le `dtype=float` garantit que les calculs de moyenne, d'ecart-type, de
sigmoid et de gradient se font sur un type numerique coherent.

## Question 17
Explique la formule de normalisation `(X - mu) / sigma`.

### Reponse
`mu` est la moyenne du train pour chaque matiere et `sigma` son ecart-type. On
retire la moyenne pour recentrer la colonne, puis on divise par l'ecart-type
pour remettre toutes les matieres sur une echelle comparable. Le resultat est
une representation ou les variations des differentes colonnes deviennent
comparables pour le calcul du gradient.

## Question 18
Pourquoi `mean(axis=0)` et `std(axis=0, ddof=0)` ?

### Reponse
`axis=0` veut dire qu'on calcule une statistique par colonne, donc par
matiere, et pas par eleve. C'est exactement ce qu'il faut pour normaliser les
features. `ddof=0` correspond a l'ecart-type calcule directement sur les
donnees observees du train, sans correction d'echantillon.

## Question 19
Explique exactement la formule de la sigmoid `1 / (1 + exp(-z))`.

### Reponse
Cette formule prend un score lineaire quelconque et le transforme en valeur
entre 0 et 1. Dans ce projet, `z` est le logit calcule a partir des notes et
des poids. La sigmoid sert donc de pont entre un score lineaire brut et une
probabilite interpretable pour la classification.

## Question 20
Pourquoi faire `np.unique(...)` sur les codes de maisons ?

### Reponse
Parce qu'il faut connaitre les classes effectivement presentes dans le train
avant de boucler dessus. `np.unique(...)` sert ici a lister les codes de
maisons presents dans la cible. C'est a partir de cette liste que le script
entraine un classifieur binaire par maison.

## Question 21
Pourquoi initialiser les poids avec `np.zeros(...)` ?

### Reponse
Le script choisit ici une initialisation simple, deterministe et lisible. Avec
des poids a zero, on part d'un point neutre avant les mises a jour du
gradient. Ce n'est pas la seule strategie possible, mais elle est facile a
auditer et suffisante pour cette implementation.

## Question 22
Pourquoi convertir la cible binaire avec `.astype(float)` ?

### Reponse
Parce que la cible binaire va etre soustraite a une probabilite, puis
reutilisee dans un calcul vectorise de gradient. La forcer en float evite de
melanger des types numeriques differentes et garde une semantique claire :
`0.0` ou `1.0` dans le meme espace que les probabilites predites.

## Question 23
Explique la formule du score lineaire `X.dot(w)`.

### Reponse
Ici, chaque eleve est une ligne de la matrice `X`, et `w` est le vecteur de
poids de la maison courante. Le produit donne un score lineaire par eleve,
avant la sigmoid. C'est ce score qui resume l'influence combinee du biais et
des notes sur la maison consideree.

## Question 24
Explique la formule de l'erreur `prediction - cible`.

### Reponse
Une fois la probabilite calculee, on la compare a la cible binaire de la
maison courante. Si la prediction est trop haute ou trop basse, cette
difference porte le signe et l'amplitude de l'erreur. C'est elle qui sert
ensuite a orienter le gradient dans le bon sens.

## Question 25
Explique exactement le role de `.T.dot(...)` dans le calcul du gradient.

### Reponse
Le vecteur d'erreur contient une erreur par eleve. La matrice des notes a pour
forme `nombre_eleves x nombre_features_avec_biais`. En la transposant, on
obtient `nombre_features_avec_biais x nombre_eleves`, ce qui permet la
multiplication matricielle avec le vecteur d'erreur. Le resultat est une somme
d'erreurs ponderees par feature, donc exactement ce qu'il faut pour le
gradient.

## Question 26
Pourquoi diviser par `students_count` dans le gradient ?

### Reponse
Parce qu'on ne veut pas une somme brute d'erreurs dependante du nombre
d'eleves, mais un gradient moyen par coefficient. Cette division stabilise
l'echelle de la mise a jour et rend le pas d'apprentissage plus interpretable.
Sans elle, le comportement du gradient changerait mecaniquement avec la taille
du dataset.

## Question 27
Explique la formule d'update `weights -= learning_rate * gradient`.

### Reponse
Le gradient indique dans quelle direction la sortie augmente le plus. Comme on
veut reduire l'erreur, on se deplace dans la direction opposee, d'ou le signe
`-=`. `learning_rate` regle la taille du pas. Si on inverse le signe, on ne
descend plus, on risque au contraire d'eloigner les poids d'une bonne
solution.

## Question 28
Pourquoi ajouter une colonne de `1` avec `np.hstack(...)` ?

### Reponse
Cette colonne represente le biais, donc l'intercept du modele. Sans elle, le
modele ne pourrait apprendre que des poids par matiere, mais pas un decalage
global independant des features. L'ajouter dans la matrice permet de traiter
le biais comme un coefficient appris dans la meme operation matricielle que le
reste.

## Question 29
Pourquoi stocker `thetas`, `mu`, `sigma` et `features` dans un JSON ?

### Reponse
Parce que `predict` doit pouvoir reconstruire exactement le meme contrat que
celui du train. Les poids seuls ne suffisent pas : il faut aussi connaitre
l'ordre des colonnes et la normalisation utilisee. Le JSON sert donc de
bundle minimal pour rendre l'inference coherente et reproductible.

## Questions sur `logreg_predict.py`

## Question 30
Pourquoi `predict` fait `json.load(...)` puis `np.array(...)` sur `thetas` ?

### Reponse
`json.load(...)` relit le fichier en structure Python standard. Mais les
calculs suivants sont vectorises avec numpy. Il faut donc transformer
`thetas`, qui sort du JSON comme listes imbriquees, en matrice numpy pour
pouvoir faire les produits matriciels du predict.

## Question 31
Pourquoi faire `.items()` et `int(house_code_text)` sur `inv_house_map` ?

### Reponse
Dans le JSON, les cles de dictionnaire reviennent en texte. Or ensuite le code
de maison sorti par `argmax` est utilise comme entier. Cette comprehension
reconstruit donc un mapping `int -> str` coherent avec les codes du modele. Si
on garde les cles en texte, le remappage final devient fragile ou faux.

## Question 32
Pourquoi `load_observations()` verifie d'abord `Index` puis les features manquantes ?

### Reponse
Parce que le fichier final doit conserver un `Index` par eleve, et parce que
le modele ne peut predire correctement que si toutes les colonnes attendues
sont presentes. Ces deux verifications fixent donc le contrat minimal de la
prediction avant tout calcul numerique.

## Question 33
Pourquoi `students_discipline_scores = raw_students_dataset[discipline_names].copy()` ?

### Reponse
La selection par `discipline_names` force l'ordre exact appris au train, et
`.copy()` evite de modifier indirectement le DataFrame source plus tard. Ici,
ce n'est pas juste une question de style: on preserve l'alignement des
colonnes et on evite des effets de bord pandas inutiles.

## Question 34
Pourquoi `predict` relit `mu`, `sigma` et `features` depuis le JSON au lieu de recalculer sur le CSV test ?

### Reponse
Parce que les poids appris n'ont de sens que dans le meme espace que celui du
train. Si je change l'ordre des colonnes ou si je renormalise sur le test, je
casse la coherence entre les coefficients appris et les donnees qu'on leur
donne. Le JSON sert a figer ce contrat: quelles colonnes, dans quel ordre,
avec quelle normalisation.

## Question 35
Pourquoi faire `np.where(sigma == 0, 1.0, sigma)` dans `predict` ?

### Reponse
Parce qu'une matiere constante au train peut produire un ecart-type nul. Si on
divise par zero au moment de normaliser le test, on casse la prediction.
`np.where(...)` remplace donc uniquement les `sigma` nuls par `1.0` pour
eviter ce plantage, tout en laissant les autres colonnes intactes.

## Question 36
Pourquoi faire `astype(float).copy().fillna(...)` dans `predict` ?

### Reponse
`astype(float)` force un type numerique coherent avant les calculs. `copy()`
evite de modifier l'objet d'entree par effet de bord. Puis `fillna(...)`
remplace les valeurs manquantes avec les moyennes du train pour garder une
prediction par eleve. Le but n'est pas juste de ne pas planter, mais aussi de
garder une structure propre et un comportement explicite.

## Question 37
Pourquoi ne pas faire `dropna(...)` au predict comme au train ?

### Reponse
Parce qu'en prediction on doit rendre une reponse pour chaque `Index` du CSV
test. Si on supprimait des lignes, on casserait l'alignement entre le fichier
d'entree et le fichier de sortie. C'est pour cela que `predict` choisit
l'imputation alors que `train` choisit la suppression.

## Question 38
Explique la formule de normalisation appliquee dans `predict`.

### Reponse
Le script prend les notes du test, leur retire `mu` du train, puis les divise
par `sigma` du train, avec la protection `sigma_safe`. Donc la formule est la
meme que celle du train, mais avec des statistiques figees et non recalculees
sur le test. C'est ce qui preserve l'invariant numerique du modele.

## Question 39
Explique le produit matriciel `X_test_bias.dot(thetas.T)` dans `predict`.

### Reponse
`X_test_bias` a une ligne par eleve et une colonne par coefficient d'entree,
biais compris. `thetas.T` met les poids de chaque maison dans le bon sens pour
produire un score par maison et par eleve. Le resultat est donc une matrice
`nombre_eleves x nombre_maisons` de logits.

## Question 40
Pourquoi faire `np.clip(...)` avant la sigmoid dans `predict` ?

### Reponse
Parce que `exp` peut deborder si les logits deviennent trop grands en valeur
absolue. `np.clip(...)` borne les scores avant le passage dans la sigmoid pour
eviter les overflows numeriques. Ce n'est pas du confort: c'est une
protection de stabilite juste avant une operation potentiellement fragile.

## Question 41
Explique la formule de probabilite `1 / (1 + exp(-score))` dans `predict`.

### Reponse
Le produit matriciel donne un score brut par maison. Cette formule transforme
ce score en probabilite comparable entre 0 et 1. Le predict peut alors
comparer les maisons sur une meme echelle avant de choisir la meilleure.

## Question 42
Explique `np.argmax(..., axis=1)` dans le `predict`.

### Reponse
A ce moment-la, on a une matrice `nombre_eleves x nombre_maisons` de
probabilites. `axis=1` veut dire qu'on cherche le maximum sur chaque ligne,
donc pour chaque eleve. Le resultat est un code de maison par eleve. Si on se
trompe d'axe, on ne choisit plus une classe par eleve, donc la sortie perd son
sens metier.

## Question 43
Pourquoi faire une comprehension de liste pour `predicted_house_names_for_all_students` ?

### Reponse
Parce qu'apres `argmax`, on n'a encore que des codes numeriques. Il faut donc
les traduire un par un en noms de maisons exportables. La comprehension de
liste fait ce remappage de facon explicite, simple a relire et coherente avec
la structure finale attendue dans le CSV.

## Question 44
Pourquoi `to_csv(..., index=False)` dans la sortie finale ?

### Reponse
Parce que le sujet attend deja une colonne `Index` metier dans le fichier.
Si on laisse pandas ecrire son propre index, on ajoute une colonne parasite et
on casse le format attendu. `index=False` est donc une protection de
conformite du livrable.

## Question 45
Que signifient exactement `thetas`, `mu`, `sigma` et `features` ?

### Reponse
`thetas`, ce sont les poids appris par le modele. `mu`, ce sont les moyennes
du train pour chaque matiere. `sigma`, ce sont les ecarts-types du train pour
chaque matiere. `features`, c'est la liste ordonnee des colonnes que le
modele utilise reellement. Si je ne sais pas traduire ces noms a l'oral, le
nommage devient une faiblesse.

## Question 46
Que se passe-t-il si `Index` manque ou si une matiere attendue manque dans le CSV test ?

### Reponse
Le script refuse de continuer. `load_observations()` verifie d'abord que la
colonne `Index` existe, puis calcule les matieres manquantes par rapport a
`discipline_names`. Si l'une de ces conditions echoue, il leve une
`ValueError` avec un message explicite. Donc l'echec est controle et lisible.

## Question 47
Ton script gere-t-il vraiment bien les erreurs ?

### Reponse
Il gere bien l'affichage des erreurs pour un humain, mais pas le contrat shell
complet. Les deux scripts attrapent l'exception dans `main()` et affichent un
message lisible. En revanche, ils ne font pas de `sys.exit(1)`, donc un echec
peut apparaitre avec un code de retour `0`. La lisibilite est bonne, la
semantique d'exploitation est insuffisante.

## Question 48
Peux-tu prouver ici le seuil de 98% demande par le sujet ?

### Reponse
Non, pas serieusement dans ce depot tel qu'il est. Le sujet prevoit
`evaluate.py` et `dataset_truth.csv`, mais ces fichiers ne sont pas presents
ici. Je peux prouver que `train` produit un fichier de poids et que `predict`
produit `houses.csv`. Je ne peux pas calculer honnetement le score officiel
sans l'outil d'evaluation et la verite terrain.

## Question 49
Que se passe-t-il si une maison manque dans le dataset d'entrainement ?

### Reponse
Le comportement n'est pas totalement robuste. Le code calcule
`unique_house_codes`, alloue la matrice de poids avec
`len(unique_house_codes)`, puis ecrit les poids a l'index
`current_house_code`. Si une classe manquante n'est pas la derniere, les codes
restants peuvent depasser la taille allouee et provoquer un
`index out of bounds`. Donc le projet suppose implicitement que les classes
attendues sont bien presentes.

## Question 50
Si j'enleve une seule des briques `features`, `mu`, `sigma` ou `thetas`, qu'est-ce qui casse ?

### Reponse
Si j'enleve `features`, je perds l'ordre exact des colonnes. Si j'enleve `mu`
ou `sigma`, je ne peux plus remettre le test dans le meme repere numerique que
le train. Si j'enleve `thetas`, je n'ai plus le modele appris. Donc ces
quatre elements ne sont pas decoratifs: ils sont le minimum pour rendre le
predict coherent et reproductible.
