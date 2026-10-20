# Football Match Probability Prediction — Analyse exploratoire

**Groupe :** Hugo, Antonio, Kalil, Gaby, Siloé

**Dataset :** *Football Match Probability Prediction* sur Kaggle. Les données sont chargées depuis les fichiers CSV du dossier `data/` (une API peut être utilisée pour les récupérer ou les mettre à jour).

**Objectifs du projet :**

- comprendre et vérifier les données de matchs de football ;
- explorer les relations entre les variables ;
- répondre à une question précise : **peut-on prédire qui va gagner le prochain match en regardant les points accumulés lors des 5 derniers matchs ?**
- à terme, construire un modèle de prévision des résultats de matchs et analyser les résultats à l'échelle des ligues ;
- étudier des indicateurs d'équipes (meilleur buteur, séries sans défaite, corrélations et causalité possible).

Le notebook actuel est une **analyse exploratoire (EDA)** suivie d'un premier test de prédiction (partie 4), volontairement simple : il ne construit pas encore de modèle de production.

---

## 1. Le jeu de données

Le notebook fusionne deux fichiers :

- `train.csv` : les matchs et l'historique des équipes ;
- `train_target_and_scores.csv` : le résultat (`target`) et le score exact.

La fusion se fait sur la clé `id`, avec la garantie qu'un identifiant correspond à un seul match.

Après fusion, la table contient :

- **110 938 matchs** ;
- **191 colonnes** ;
- une ligne = **un match** entre une équipe à domicile et une équipe à l'extérieur ;
- pour chaque équipe, les **10 matchs précédents** : dates, buts marqués, buts encaissés, notes, adversaires, etc. ;
- la période couverte : du **1er décembre 2019** au **1er mai 2021** (période Covid) ;
- **727 ligues** différentes ;
- **10 214 équipes** uniques ;
- **9 081 matchs de coupe**.

La variable `target` vaut `home` (victoire à domicile), `draw` (match nul) ou `away` (victoire à l'extérieur).

---

## 2. Comment lancer le notebook

Le notebook se trouve dans `Notebook/eda_jointure.ipynb`.

Les données doivent être placées dans `data/` :

```text
data/
├── train.csv
└── train_target_and_scores.csv
```

Bibliothèques utilisées : `pandas`, `numpy`, `matplotlib`, `seaborn` et `scikit-learn`.

Le notebook se lit dans l'ordre : chaque partie commence par une question, montre les analyses, puis se termine par une réponse explicite.

---

## Partie 1 — Qu'ai-je ?

**Question :** que contient la table, et à quoi ressemble une observation ?

Cette partie sert à prendre connaissance des données avant de les interpréter.

### Ce que l'on regarde

- les dimensions de la table (`shape`) ;
- quelques lignes (`head`, `sample`) pour voir à quoi ressemble un match ;
- les ligues présentes et leur fréquence ;
- le nombre d'équipes uniques ;
- le nombre de matchs de coupe ;
- la liste des variables et leur type (`info`, `dtypes`) ;
- les premiers ordres de grandeur (`describe`).

### Ce que l'on apprend

- Chaque ligne décrit un match complet : équipe à domicile, équipe à l'extérieur, compétition, date, résultat et historique récent des deux équipes.
- Les variables sont de plusieurs types : nombres (buts, notes), texte (noms d'équipes, ligues), dates et booléens (coupe, match à domicile).
- L'historique de chaque équipe est déjà présent dans la table : les colonnes `_1` à `_10` correspondent aux 10 matchs précédents.
- La couverture est très large : 727 ligues et plus de 10 000 équipes. Toutes les ligues n'ont donc pas le même poids.

**Réponse :** nous avons une table de matchs enrichie par l'historique récent des deux équipes. Elle est assez grande pour une EDA, mais très hétérogène.

---

## Partie 2 — Puis-je faire confiance aux données ?

**Question :** les données sont-elles complètes, cohérentes et utilisables ?

Avant de tirer des conclusions, il faut chercher les défauts qui pourraient fausser l'analyse.

### Ce que l'on vérifie

1. **Les valeurs manquantes** : combien de colonnes en ont, et dans quelles proportions ?
2. **Les doublons** : y a-t-il des lignes ou des identifiants répétés ?
3. **Les catégories** : les valeurs de `target` sont-elles bien celles attendues ? Les noms d'équipes sont-ils vides ?
4. **Les valeurs impossibles** : buts négatifs, indicateurs binaires hors de `[0, 1]`, scores illisibles.
5. **Les dates** : sont-elles convertibles, cohérentes, et les matchs historiques sont-ils bien antérieurs au match à prédire ?

### Ce que l'on apprend

- **186 colonnes sur 191** contiennent au moins une valeur manquante, soit environ **8,10 %** de toutes les cellules. Cela concerne surtout les historiques incomplets : les équipes n'ont pas toujours 10 matchs disponibles.
- Il y a **0 doublon** et **0 identifiant dupliqué** : la clé `id` est fiable.
- Le score et la cible ne manquent pas.
- Les valeurs de `target` sont uniquement `home`, `draw` et `away`.
- Il n'y a **aucun but négatif**, **aucun indicateur binaire invalide** et **aucune date historique future**.
- Les dates sont cohérentes, mais la période correspond à la **Covid-19** : le calendrier peut être perturbé et la couverture varier dans le temps.

**Réponse :** les lignes sont cohérentes et utilisables pour une EDA et un premier modèle. La réserve principale concerne les **valeurs manquantes dans les historiques** : il faudra les traiter explicitement avant de modéliser.

---

## Partie 3 — Que racontent ces variables ?

**Question :** quelles sont les distributions, les tendances et les relations entre les variables ?

Cette partie explore les données sous plusieurs angles.

### 3.1 Les buts

On calcule les moyennes, médianes, quantiles, histogrammes et boxplots des buts.

- **1,52 but** en moyenne pour l'équipe à domicile ;
- **1,24 but** pour l'équipe à l'extérieur ;
- **2,76 buts** par match au total.

Les distributions sont asymétriques : beaucoup de matchs avec peu de buts, quelques matchs avec beaucoup de buts.

### 3.2 Les résultats

La répartition de `target` est la suivante :

| Résultat | Part |
|---|---|
| Victoire à domicile (`home`) | **43,37 %** |
| Match nul (`draw`) | **24,92 %** |
| Victoire à l'extérieur (`away`) | **31,71 %** |

Il existe donc un **avantage du domicile** : gagner à la maison est le résultat le plus fréquent.

On observe aussi une forte hétérogénéité entre ligues : **727 ligues**, dont **105** avec moins de 10 matchs. Les comparaisons entre ligues doivent donc être prudentes.

### 3.3 L'évolution dans le temps

Le notebook trace :

- le nombre de matchs par mois ;
- la proportion mensuelle de chaque résultat.

Cela permet de repérer les changements de couverture (moins de matchs à certaines périodes, compétitions qui démarrent ou s'arrêtent) et de ne pas confondre un phénomène sportif avec un artefact de collecte.

**Conséquence :** pour évaluer un futur modèle, il faudra faire une **séparation chronologique** (entraîner sur le passé, tester sur le futur), et non une séparation aléatoire.

### 3.4 Les comparaisons entre ligues

Avec `groupby()`, on compare les ligues qui ont au moins 100 matchs :

- nombre de matchs ;
- buts moyens à domicile et à l'extérieur ;
- buts moyens au total ;
- proportions de victoires, de nuls et de défaites.

Les ligues ne se ressemblent pas : certaines sont plus offensives, d'autres plus défensives, et l'avantage du domicile varie.

### 3.5 Les relations entre variables

Le notebook croise d'abord deux variables catégorielles : **coupe ou championnat** et **résultat**. Il calcule ensuite des corrélations entre variables numériques :

- la forme offensive récente et les buts du match ;
- la différence de note récente et les buts ;
- les buts à domicile, à l'extérieur et au total.

Quelques résultats :

- la forme offensive récente est corrélée à **0,209** avec les buts marqués à domicile ;
- la différence de note récente est corrélée à **0,252** avec les buts à domicile ;
- les relations sont **réelles mais modérées** : la forme récente aide un peu, mais elle n'explique pas tout.

### 3.6 La matrice de corrélation

Une matrice de corrélation et une carte de chaleur (heatmap) résument les relations entre 8 variables :

- buts récents à domicile et à l'extérieur ;
- notes récentes à domicile et à l'extérieur ;
- écart de note récent ;
- buts du match à domicile, à l'extérieur et au total.

Comment la lire :

- **+1** = relation positive parfaite ;
- **0** = pas de relation linéaire ;
- **−1** = relation négative parfaite.

Les corrélations les plus fortes sont en partie **mécaniques** :

- `home_goals` et `away_goals` avec `total_goals` : **0,72** et **0,63** (le total est la somme des deux) ;
- buts récents et note récente : **0,68** à domicile et **0,67** à l'extérieur.

Les relations entre forme récente et buts du match sont plus modestes : **0,19 à 0,25**. L'écart de note récent est corrélé à **−0,23** avec les buts marqués par l'équipe à l'extérieur.

Enfin, les formes récentes à domicile et à l'extérieur ne sont corrélées qu'à **0,18** : ce sont deux informations différentes, à traiter séparément.

**Réponse de la partie 3 :** les variables montrent un avantage du domicile, une forte hétérogénéité entre ligues et des relations prédictives réelles mais modérées. Une corrélation ne prouve pas une causalité.

---

## Partie 4 — Peut-on prédire le gagnant du prochain match ?

**Question :** *Peut-on prédire qui va gagner le prochain match en regardant les points accumulés lors des 5 derniers matchs ?*

L'idée est simple : une équipe qui gagne souvent ses derniers matchs devrait être « en forme », et donc gagner le match suivant. On teste cette idée avec des règles brutales, puis avec un petit modèle.

### 4.1 La méthode : compter les points des derniers matchs

Pour chaque équipe, on compte les points de ses matchs précédents : **3 points par victoire, 1 point par match nul, 0 point par défaite**.

- **Méthode brute** : on additionne les points des matchs 1 à 5 (colonnes `_1` à `_5`). L'équipe avec le plus de points est prédite gagnante. En cas d'égalité, on prédit « domicile » (il gagne un peu plus souvent).
- **Ajustement « forme habituelle »** : on refait la même prédiction avec les matchs 6 à 10, pour comparer.
- On garde les matchs où les deux équipes ont au moins 3 matchs joués dans chaque fenêtre : **95 313 matchs analysés**.

### 4.2 Résultat des deux règles simples

| Méthode (toute la table) | Bonnes prédictions |
|---|---|
| Règle brute (matchs 1 à 5) | **45,0 %** |
| Règle forme habituelle (matchs 6 à 10) | **44,1 %** |
| Référence : toujours prédire « domicile » | **43,1 %** |

Les deux règles font à peine mieux que parier « domicile » à chaque match, sans rien regarder.

### 4.3 Un modèle entraîné sur 70 % des matchs

On entraîne une régression logistique sur 70 % des matchs et on la teste sur les 30 % restants. Ses seules informations : les points des matchs 1 à 5 et 6 à 10 des deux équipes.

| Méthode (sur le test de 30 %) | Bonnes prédictions |
|---|---|
| Modèle entraîné (70 %) | **47,2 %** |
| Règle brute (matchs 1 à 5) | **44,9 %** |
| Règle forme habituelle (matchs 6 à 10) | **44,3 %** |
| Référence : toujours prédire « domicile » | **43,4 %** |

### 4.4 Réponse de la partie 4

**Non, on ne peut pas prédire de façon fiable qui va gagner le prochain match avec les seuls points des 5 derniers matchs.**

- La règle brute ne prédit bien qu'environ **1 match sur 2** (45,0 %), à peine mieux que « toujours domicile » (43,1 %).
- La forme habituelle (matchs 6 à 10) donne le même résultat (44,1 %) : quelle que soit la fenêtre choisie, le signal reste faible.
- Le modèle entraîné sur 70 % des matchs atteint **47,2 %** sur le test : une petite amélioration, mais largement insuffisante pour être utile.
- Les matchs nuls (environ **25 %** des matchs) sont quasiment impossibles à prévoir : les règles ne prédisent jamais « nul », et le modèle se trompe presque toujours sur eux.
- Conclusion : les points des 5 derniers matchs ne suffisent pas à prédire un gagnant. Le résultat du prochain match reste trop incertain.

---

## 5. Les résultats clés en un coup d'œil

| Partie | Question | Réponse courte |
|---|---|---|
| 1 | Qu'ai-je ? | 110 938 matchs, 191 variables, 727 ligues, 10 214 équipes. |
| 2 | Puis-je faire confiance aux données ? | Oui, mais 186 colonnes ont des valeurs manquantes (8,10 % des cellules), surtout dans les historiques. |
| 3 | Que racontent les variables ? | Avantage du domicile (43,37 %), forte hétérogénéité entre ligues, corrélations réelles mais modérées. |
| 4 | Peut-on prédire le gagnant avec les 5 derniers matchs ? | Non : ≈ 45 % de bonnes prédictions avec la règle brute, ≈ 47 % avec un modèle entraîné sur 70 % des matchs — à peine mieux que « toujours domicile » (≈ 43 %). |

**Conclusion générale :** les données sont riches et cohérentes, mais le football reste très aléatoire. Les variables de forme récente ont un pouvoir explicatif limité. Pour progresser, il faudra construire un modèle de prédiction avec une séparation chronologique, de bonnes variables de force d'équipe et une évaluation honnête.

---

## 6. Limites de l'analyse

- **Analyse observationnelle** : on observe des associations, pas une causalité démontrée.
- **Séparation aléatoire dans la partie 4** : le test 70/30 est un tirage aléatoire, alors que la partie 3 recommande une **séparation chronologique** (entraîner sur le passé, tester sur le futur). Le score réel sur des matchs futurs serait probablement un peu différent.
- **Fenêtre courte** : la forme récente est mesurée sur 5 matchs, ce qui est peu et bruité.
- **Notes de match** : la note est un indicateur de performance par match, pas un classement Elo officiel. La force des adversaires est donc approximée.
- **Historiques incomplets** : les équipes avec moins de 3 matchs valides sont exclues, ce qui peut écarter les nouvelles équipes ou les ligues peu couvertes.
- **Période Covid** : le calendrier et la couverture peuvent être perturbés.
- **Hétérogénéité des ligues** : les niveaux et les styles de jeu ne sont pas comparables d'une ligue à l'autre.
- **Hasard du football** : même un bon modèle ne peut pas prédire chaque match ; l'objectif est d'être meilleur que le hasard, pas de devenir parfait.

---

## 7. Pistes pour la suite

- Construire un modèle de prédiction (régression logistique, gradient boosting) avec une **séparation chronologique**.
- Ajouter des variables de force d'équipe (Elo, classement, forme pondérée, jours de repos).
- Évaluer le modèle avec plusieurs indicateurs : exactitude, log loss, calibration.
- Analyser les résultats à l'échelle des **ligues** (ligues offensives/défensives, avantage du domicile).
- Étudier des indicateurs d'équipes : meilleur buteur, séries sans défaite, dynamique de classement.
- Aller plus loin sur la causalité avec des méthodes adaptées, si les données le permettent.
# Football Match Probability Prediction

Projet de prédiction du résultat de matchs de football (`home`, `draw`, `away`)
à partir des données du dataset Kaggle Football Match Probability Prediction.

## Application interactive

L'application Streamlit entraîne un classifieur à partir de `data/train.csv` lors
de son premier chargement. Comme dans la partie 4 de
`Notebook/eda_jointure.ipynb`, le modèle est une régression logistique utilisant
quatre variables : les points cumulés par les deux équipes sur les matchs 1–5 et
6–10 (3 points par victoire, 1 par nul, 0 par défaite). Il est entraîné sur les
lignes qui ont au moins 3 scores valides dans chacune de ces fenêtres pour les
deux équipes.

L'application permet de saisir ces dix scores pour chaque équipe, d'afficher les
probabilités des trois résultats et de générer des prédictions en lot pour un
CSV contenant les colonnes d'historique du projet. Le CSV téléchargé contient
`id,home,draw,away`, comme `data/sample_submission.csv`. Le notebook obtient
47,2 % de bonnes prédictions sur un test aléatoire de 30 % ; ce score est
indicatif et ne garantit pas les résultats futurs. Pour une ligne CSV dont
l'historique est trop incomplet, l'application utilise explicitement les
fréquences des trois résultats dans les données d'entraînement plutôt qu'une
prédiction du modèle.

Depuis la racine du dépôt, installez les dépendances et lancez l'application :

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

L'application nécessite `data/train.csv`. Le premier lancement peut prendre
quelques instants, le temps d'entraîner le modèle. Les entraînements suivants
pendant la même session Streamlit réutilisent le modèle en mémoire.

## Données

- `data/train.csv` : matchs historiques et cible `target`.
- `data/test.csv` : matchs sans cible, disponibles pour les prédictions en lot.
- `data/sample_submission.csv` : exemple du format de soumission.
- `EDA.ipynb` et `Notebook/eda_jointure.ipynb` : analyses exploratoires.
