# Prédiction du prix d'une voiture d'occasion

Projet de Machine Learning : estimation du prix d'une voiture d'occasion à partir de ses
caractéristiques (marque, modèle, puissance, kilométrage, âge, etc.), à partir d'un jeu de
données d'annonces automobiles en Allemagne.

## Application

Application déployée : https://mlprojectcarpriceprediction-pq7rnsr3rwhb8wfrujchcm.streamlit.app/

## Structure du dépôt

- `main.py` — nettoyage des données brutes (`autos.csv` → `autos_clean.csv`)
- `eda.py` — analyse exploratoire des données
- `linear_regression.py`, `ridge.py`, `lasso.py`, `elastic_net.py` — modèles linéaires
- `decision_tree.py`, `random_forest.py`, `extra_trees.py` — modèles à base d'arbres
- `gradient_boosting.py`, `xgboost_model.py`, `lightgbm_model.py` — boosting
- `knn.py` — k plus proches voisins
- `tuning.py` — optimisation des hyperparamètres (RandomizedSearchCV) sur les 3 meilleurs modèles
- `train_final_model.py` — entraînement du modèle final (LightGBM) + modèles de quantile,
  sérialisation en `model_pipeline.joblib` / `app_options.joblib`
- `app.py` — application Streamlit
- `requirements.txt` — dépendances nécessaires au déploiement

## Démarche

1. **Préprocessing** : nettoyage des colonnes inutiles, traitement des valeurs aberrantes
   (prix, année, puissance) et des valeurs manquantes, feature engineering (âge du véhicule).
2. **Construction des modèles** : comparaison de 11 algorithmes (linéaires, arbres, ensembles,
   boosting, voisinage) sur le prix en log.
3. **Tuning** : sélection des 3 meilleurs modèles, optimisation des hyperparamètres, choix du
   modèle final sur la base de la performance en test et de l'écart train/test (overfitting).
4. **Déploiement** : sérialisation du pipeline complet et mise à disposition via une application
   Streamlit.

## Exécution locale

```bash
pip install -r requirements.txt
python main.py                 # génère autos_clean.csv à partir de autos.csv
python train_final_model.py    # entraîne et sérialise le modèle final
streamlit run app.py           # lance l'application
```

## Auteur

Hajar Amine
