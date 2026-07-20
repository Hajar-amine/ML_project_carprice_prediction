# ==========================================================
# Machine Learning - Tuning des hyperparamètres
# Étape V du rapport : RandomForest, XGBoost, LightGBM
# ==========================================================

import time
import sys

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

sys.stdout.reconfigure(encoding="utf-8")

# ----------------------------------------------------------
# Charger les données nettoyées
# ----------------------------------------------------------

df = pd.read_csv("autos_clean.csv")

df["age"] = 2016 - df["yearOfRegistration"]

features = [
    "brand",
    "vehicleType",
    "gearbox",
    "fuelType",
    "powerPS",
    "kilometer",
    "age",
    "notRepairedDamage"
]

X = df[features]
y = np.log1p(df["price"])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("=" * 60)
print("TUNING DES HYPERPARAMÈTRES - 3 MODÈLES CANDIDATS")
print("=" * 60)
print("Taille X_train :", X_train.shape)
print("Taille X_test  :", X_test.shape)

# ----------------------------------------------------------
# Préprocessing (identique aux scripts précédents)
# ----------------------------------------------------------

numeric_features = ["powerPS", "kilometer", "age"]

categorical_features = [
    "brand",
    "vehicleType",
    "gearbox",
    "fuelType",
    "notRepairedDamage"
]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# ----------------------------------------------------------
# Échantillon pour la recherche d'hyperparamètres
# ----------------------------------------------------------
# RandomizedSearchCV sur les 274k lignes de X_train, avec
# plusieurs folds et plusieurs modèles, serait beaucoup trop
# long (un seul entraînement RandomForest prend déjà ~18 min).
# On cherche donc les hyperparamètres sur un échantillon,
# puis on ré-entraîne le meilleur jeu d'hyperparamètres trouvé
# sur l'intégralité de X_train pour l'évaluation finale.

SEARCH_SAMPLE_SIZE = 20000

X_search = X_train.sample(n=SEARCH_SAMPLE_SIZE, random_state=42)
y_search = y_train.loc[X_search.index]

print(f"\nRecherche d'hyperparamètres sur un échantillon de {SEARCH_SAMPLE_SIZE} lignes")
print(f"(le meilleur modèle trouvé sera ensuite ré-entraîné sur les "
      f"{X_train.shape[0]} lignes complètes de X_train)")

# ----------------------------------------------------------
# Espaces d'hyperparamètres
# ----------------------------------------------------------

candidates = {
    "RandomForest": {
        "estimator": RandomForestRegressor(random_state=42, n_jobs=1),
        "param_distributions": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [10, 15, 20, 25, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2"]
        }
    },
    "XGBoost": {
        "estimator": XGBRegressor(random_state=42, n_jobs=1),
        "param_distributions": {
            "model__n_estimators": [100, 200, 300, 400],
            "model__max_depth": [3, 5, 7, 9],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__subsample": [0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0]
        }
    },
    "LightGBM": {
        "estimator": LGBMRegressor(random_state=42, n_jobs=1, verbose=-1),
        "param_distributions": {
            "model__n_estimators": [100, 200, 300, 400],
            "model__max_depth": [-1, 5, 10, 15],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__num_leaves": [15, 31, 63, 127],
            "model__subsample": [0.7, 0.8, 0.9, 1.0]
        }
    }
}

results = []

for name, cfg in candidates.items():
    print("\n" + "=" * 60)
    print(f"TUNING - {name}")
    print("=" * 60)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", cfg["estimator"])
    ])

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=cfg["param_distributions"],
        n_iter=8,
        cv=3,
        scoring="r2",
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    start = time.time()
    search.fit(X_search, y_search)
    search_time = time.time() - start

    print(f"Meilleurs hyperparamètres : {search.best_params_}")
    print(f"Meilleur score CV (R2_log) sur échantillon : {search.best_score_:.4f}")
    print(f"Temps de recherche : {search_time:.1f}s")

    # Ré-entraînement du meilleur modèle sur X_train complet
    # (on remet n_jobs=-1 pour accélérer ce dernier entraînement)
    best_pipeline = search.best_estimator_
    try:
        best_pipeline.set_params(model__n_jobs=-1)
    except ValueError:
        pass

    start = time.time()
    best_pipeline.fit(X_train, y_train)
    fit_time = time.time() - start

    y_train_pred_log = best_pipeline.predict(X_train)
    y_test_pred_log = best_pipeline.predict(X_test)

    r2_log_train = r2_score(y_train, y_train_pred_log)
    r2_log_test = r2_score(y_test, y_test_pred_log)

    y_test_pred = np.expm1(y_test_pred_log)
    y_test_real = np.expm1(y_test)

    mae_eur = mean_absolute_error(y_test_real, y_test_pred)
    rmse_eur = mean_squared_error(y_test_real, y_test_pred) ** 0.5
    r2_eur_test = r2_score(y_test_real, y_test_pred)

    results.append({
        "Modele": f"{name} (tuné)",
        "R2_log_train": r2_log_train,
        "R2_log_test": r2_log_test,
        "MAE_eur": mae_eur,
        "RMSE_eur": rmse_eur,
        "R2_eur_test": r2_eur_test,
        "Overfitting_gap": r2_log_train - r2_log_test,
        "Temps_s": round(search_time + fit_time, 1)
    })

    print(f"\nR2_log_train : {r2_log_train:.3f}")
    print(f"R2_log_test  : {r2_log_test:.3f}")
    print(f"MAE_eur      : {mae_eur:.2f} €")
    print(f"RMSE_eur     : {rmse_eur:.2f} €")
    print(f"R2_eur_test  : {r2_eur_test:.3f}")

# ----------------------------------------------------------
# Tableau comparatif (comme dans le rapport, section V.2)
# ----------------------------------------------------------

results_df = pd.DataFrame(results).sort_values("R2_log_test", ascending=False)

print("\n" + "=" * 60)
print("TABLEAU COMPARATIF - MODÈLES TUNÉS")
print("=" * 60)
print(results_df.to_string(index=False))

# ----------------------------------------------------------
# Choix du modèle final (section V.3 du rapport) :
# meilleur R2_log_test, en départageant les modèles proches
# par le plus faible écart train/test (overfitting)
# ----------------------------------------------------------

best_test_score = results_df["R2_log_test"].max()
close_candidates = results_df[
    results_df["R2_log_test"] >= best_test_score - 0.005
]
final_choice = close_candidates.sort_values("Overfitting_gap").iloc[0]

print("\n" + "=" * 60)
print("MODÈLE FINAL RETENU")
print("=" * 60)
print(
    f"{final_choice['Modele']} — "
    f"R2_log_test = {final_choice['R2_log_test']:.3f}, "
    f"écart train/test = {final_choice['Overfitting_gap']:.3f}"
)

print("\nTuning terminé avec succès !")
