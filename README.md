# Football Match Probability Prediction — Analyse exploratoire

**Groupe :** Hugo, Antonio, Kalil, Gaby, Siloé

**Dataset :** *Football Match Probability Prediction* sur Kaggle. Les données sont chargées depuis les fichiers CSV du dossier `data/` (une API peut être utilisée pour les récupérer ou les mettre à jour).

**Objectifs du projet :**

- comprendre et vérifier les données de matchs de football ;
- explorer les relations entre les variables ;
- répondre à une question précise : **la forme récente d'une équipe influence-t-elle son prochain résultat, à niveau habituel et adversaires comparables ?**
- à terme, construire un modèle de prévision des résultats de matchs et analyser les résultats à l'échelle des ligues ;
- étudier des indicateurs d'équipes (meilleur buteur, séries sans défaite, corrélations et causalité possible).

Le notebook actuel est une **analyse exploratoire (EDA)**. Il ne construit pas encore de modèle de production : les régressions de la partie 4 servent à comprendre les données, pas à prédire en conditions réelles.

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

Bibliothèques utilisées : `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy` et `scikit-learn`.

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

## Partie 4 — La forme récente influence-t-elle le résultat suivant ?

**Question :** *Dans quelle mesure les performances récentes d'une équipe influencent-elles son prochain résultat, indépendamment de son niveau habituel et de la force des adversaires rencontrés ?*

C'est la partie la plus détaillée du notebook. Voici l'idée, étape par étape.

### 4.1 Le piège à éviter

Les équipes qui viennent de bien jouer sont souvent... de bonnes équipes. Elles ont donc aussi de meilleures chances de gagner le match suivant.

Si on regarde seulement « bonne forme → bons résultats », on ne sait pas si c'est :

- la **forme** qui aide ;
- ou simplement la **qualité de l'équipe**, qui produit à la fois la bonne forme et les bons résultats.

Il faut donc séparer les deux.

### 4.2 Étape 1 — Transformer un match en deux observations

Un match Real Madrid–Barcelone ne donne pas une seule ligne, mais **deux** :

- une ligne pour le Real Madrid (équipe à domicile) ;
- une ligne pour Barcelone (équipe à l'extérieur).

Chaque ligne contient l'historique de l'équipe concernée et les points qu'elle a gagnés dans ce match.

La variable à expliquer est le **nombre de points au match suivant** :

- **3** points pour une victoire ;
- **1** point pour un match nul ;
- **0** point pour une défaite.

Au total, après nettoyage, on obtient **190 624 observations équipe-match**, correspondant à **95 312 matchs** et **8 188 équipes**. Le nombre d'observations est plus grand que le nombre de matchs, car chaque match est vu deux fois.

### 4.3 Étape 2 — Mesurer la forme récente

Pour chaque équipe, on regarde les **5 derniers matchs** (colonnes `_1` à `_5`) et on résume :

- les **points par match** (0 à 3) ;
- la **différence de buts** ;
- la **note moyenne**.

L'idée est simple : une équipe qui vient de gagner beaucoup a une forme récente élevée.

### 4.4 Étape 3 — Mesurer le niveau habituel sans mélanger avec la forme

Pour mesurer le niveau habituel, on utilise les **5 matchs précédents** (colonnes `_6` à `_10`), et non les 5 matchs de la forme récente.

Pourquoi ? Parce que si on utilisait les mêmes matchs pour mesurer le niveau et la forme, on comparerait la forme avec elle-même. En séparant les deux fenêtres :

- **forme récente** = ce qui s'est passé dans les 5 derniers matchs ;
- **niveau habituel** = ce que l'équipe faisait dans les 5 matchs d'avant.

Les deux fenêtres ne se chevauchent pas.

### 4.5 Étape 4 — Mesurer la force des adversaires

Deux contrôles supplémentaires sont ajoutés :

- la **note moyenne des adversaires rencontrés** pendant les 5 derniers matchs : une bonne forme contre des adversaires faibles vaut moins qu'une bonne forme contre des adversaires forts ;
- le **niveau habituel du prochain adversaire** : jouer contre une équipe forte ou faible change évidemment le résultat attendu.

Sans ces contrôles, on pourrait attribuer à la forme un effet qui vient en réalité du calendrier.

### 4.6 Étape 5 — Nettoyer les données

On exige :

- au moins **3 matchs valides** dans la fenêtre récente ;
- au moins **3 matchs valides** dans la fenêtre du niveau habituel ;
- les données des deux équipes et du prochain adversaire.

Les observations incomplètes sont écartées. Cela évite que le résultat dépende de quelques cas particuliers.

### 4.7 Étape 6 — Première photo, sans aucun contrôle

On classe les observations selon la forme récente, en **5 groupes (quintiles)**, du plus faible au plus fort :

- Q1 = la moins bonne forme ;
- Q5 = la meilleure forme.

Puis on regarde les points moyens au match suivant :

| Groupe de forme | Points au match suivant |
|---|---|
| Q1 (forme la plus faible) | **1,15** |
| Q5 (forme la plus forte) | **1,69** |

Soit un écart de **+0,54 point**. La corrélation brute est de **0,14**.

À ce stade, tout semble indiquer que la forme récente compte.

### 4.8 Étape 7 — Ajouter les contrôles

On compare trois modèles :

| Modèle | Ce qu'il contrôle | Coefficient de la forme |
|---|---|---|
| **M1** | la forme seule | **0,186** |
| **M2** | + niveau habituel + niveau du prochain adversaire + match à domicile | **0,133** |
| **M3** | + force des adversaires récents + forme récente de l'adversaire | **0,103** |

Les coefficients sont **standardisés** : les variables n'ont pas la même unité (points, notes), donc on les ramène à une unité commune, l'écart-type. Un coefficient plus grand signifie que la variable pèse davantage. Un coefficient de 0,10 signifie : quand la variable augmente d'un écart-type, le résultat augmente de 0,10 point, toutes choses égales par ailleurs.

Le coefficient de la forme passe de **0,186** à **0,103** : l'effet diminue presque de moitié dès que l'on contrôle le niveau et les adversaires.

Dans le modèle complet M3, les autres variables pèsent davantage :

| Variable | Coefficient standardisé |
|---|---|
| Niveau habituel de l'équipe | **0,213** |
| Niveau du prochain adversaire | **−0,166** |
| Forme récente de l'adversaire | **−0,156** |
| Force des adversaires récents | **−0,087** |
| Jouer à domicile | **0,352** |
| **Forme récente de l'équipe** | **0,103** |

Les signes négatifs sont logiques : plus le prochain adversaire est fort, plus il est en forme, ou plus les adversaires récents étaient forts, moins l'équipe gagne de points en moyenne.

En points : passer de **0 à 3 points de forme récente** est associé à **+0,46 point** au match suivant, en gardant le niveau et les adversaires constants. C'est réel, mais modeste.

### 4.9 Étape 8 — La preuve décisive : comparer chaque équipe à elle-même

Le modèle M3 reste imparfait : le niveau habituel est mesuré avec seulement 5 matchs, donc avec du bruit. Il peut rester de la qualité d'équipe non contrôlée.

Pour aller plus loin, on compare **chaque équipe à sa propre moyenne**. Autrement dit, on ne compare plus le Real Madrid à une équipe faible, mais le Real Madrid d'une bonne série au Real Madrid d'une mauvaise série.

Techniquement, on retire à chaque variable la moyenne de l'équipe. Cela élimine tout ce qui est fixe pour l'équipe : budget, effectif, ligue, niveau structurel.

Résultats :

- le coefficient de la forme devient **−0,052** ;
- l'écart entre le quintile de forme le plus bas et le plus haut est de **−0,13 point** (le gradient disparaît, avec même une très légère baisse) ;
- la corrélation **entre équipes** est de **0,69**, mais la corrélation **à l'intérieur des équipes** est de **−0,04**.

Autrement dit : la relation observée au départ est presque entièrement une relation **entre équipes**, pas une relation **au sein d'une même équipe**.

Si on mesure la performance par la **note** plutôt que par les points, l'effet interne devient très légèrement positif (**+0,056**), mais il reste négligeable.

### 4.10 Pourquoi l'effet disparaît-il ?

Prenons une analogie simple : les élèves qui ont eu de bonnes notes aux derniers contrôles ont souvent de bonnes notes au suivant. Mais c'est surtout parce qu'ils sont bons élèves. Si on compare **le même élève** à lui-même, une bonne série ne garantit pas la note suivante : il peut très bien revenir vers sa moyenne.

C'est exactement ce que l'on observe :

- les bonnes équipes ont de bonnes séries **et** de bons résultats ;
- mais quand une même équipe a une série meilleure que d'habitude, elle ne gagne pas significativement plus de points au match suivant ;
- on observe même un léger **retour à la moyenne** après une série exceptionnelle.

### 4.11 Réponse de la partie 4

**La forme récente, par elle seule, influence très peu le résultat suivant, et probablement pas de façon causale.**

- Sans contrôle, l'écart entre la pire et la meilleure forme est de **+0,54 point**.
- En contrôlant le niveau et les adversaires, l'effet se réduit à **+0,46 point** entre 0 et 3 points de forme (coefficient standardisé de 0,103).
- En comparant chaque équipe à elle-même, l'effet positif disparaît, voire s'inverse très légèrement (coefficient de **−0,052**, écart de **−0,13 point** entre Q1 et Q5).

La forme récente peut servir de **signal faible**, mais elle ne suffit pas à prédire le résultat. L'essentiel de l'association brute vient de la **qualité habituelle des équipes**, pas de la forme récente elle-même.

---

## 5. Les résultats clés en un coup d'œil

| Partie | Question | Réponse courte |
|---|---|---|
| 1 | Qu'ai-je ? | 110 938 matchs, 191 variables, 727 ligues, 10 214 équipes. |
| 2 | Puis-je faire confiance aux données ? | Oui, mais 186 colonnes ont des valeurs manquantes (8,10 % des cellules), surtout dans les historiques. |
| 3 | Que racontent les variables ? | Avantage du domicile (43,37 %), forte hétérogénéité entre ligues, corrélations réelles mais modérées. |
| 4 | La forme récente influence-t-elle le résultat ? | Très peu : l'effet brut diminue avec les contrôles et disparaît quand on compare chaque équipe à elle-même. |

**Conclusion générale :** les données sont riches et cohérentes, mais le football reste très aléatoire. Les variables de forme récente ont un pouvoir explicatif limité. Pour progresser, il faudra construire un modèle de prédiction avec une séparation chronologique, de bonnes variables de force d'équipe et une évaluation honnête.

---

## 6. Limites de l'analyse

- **Analyse observationnelle** : on observe des associations, pas une causalité démontrée.
- **Pas de test hors échantillon dans la partie 4** : les régressions servent à expliquer les données observées, pas à mesurer une performance de prédiction sur des matchs futurs.
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
