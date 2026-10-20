# Football Match Probability Prediction

Projet de prédiction du résultat de matchs de football (`home`, `draw`, `away`)
à partir des données du dataset Kaggle Football Match Probability Prediction.

## Application interactive

L'application Streamlit entraîne un classifieur à partir de `data/train.csv` lors
de son premier chargement. Elle permet de saisir les scores et le lieu des cinq
derniers matchs de chaque équipe, d'afficher les probabilités des trois résultats,
et de générer des prédictions pour un fichier CSV contenant les colonnes
d'historique du projet. Le CSV téléchargé contient `id,home,draw,away`, comme
`data/sample_submission.csv`. Les noms des équipes sont uniquement utilisés pour
l'affichage ; les variables d'entrée du modèle sont les scores et le lieu des
matchs récents. Le modèle prend également en compte les moyennes de buts, de
buts encaissés et de points obtenus par l'équipe à domicile dans ses matchs
passés à domicile, ainsi que par l'équipe visiteuse dans ses matchs passés à
l'extérieur.

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
