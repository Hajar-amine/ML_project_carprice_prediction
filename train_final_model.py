# ==========================================================
# Machine Learning - Entraînement du modèle final
# Étape VI du rapport : sérialisation du pipeline
# ==========================================================
#
# Modèle retenu à l'étape V : LightGBM (meilleur compromis
# performance / overfitting). On entraîne ici :
#   - le modèle "point" (prédiction médiane du prix)
#   - deux modèles de quantile (10% / 90%) pour donner un
#     intervalle de confiance, comme décrit dans le rapport (VI.2)
# sur l'intégralité des données nettoyées (le split train/test
# a déjà servi à valider la performance aux étapes IV et V).
# ==========================================================

import sys

import joblib
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from lightgbm import LGBMRegressor

sys.stdout.reconfigure(encoding="utf-8")

# ----------------------------------------------------------
# Meilleurs hyperparamètres trouvés par tuning.py
# ----------------------------------------------------------

BEST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.2,
    "num_leaves": 127,
    "subsample": 0.7,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1
}

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

print("=" * 60)
print("ENTRAÎNEMENT DU MODÈLE FINAL (LightGBM)")
print("=" * 60)
print("Dimensions de X :", X.shape)

# ----------------------------------------------------------
# Préprocessing (identique à tous les scripts précédents)
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
# Modèle "point" (médiane / moyenne du log-prix)
# ----------------------------------------------------------

print("\nEntraînement du modèle principal...")

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LGBMRegressor(**BEST_PARAMS))
])

model.fit(X, y)

print("Modèle principal entraîné.")

# ----------------------------------------------------------
# Modèles de quantile (10% et 90%) pour l'intervalle de confiance
# ----------------------------------------------------------

print("\nEntraînement du modèle de quantile 10%...")

model_lower = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LGBMRegressor(
        **{**BEST_PARAMS, "objective": "quantile", "alpha": 0.1}
    ))
])
model_lower.fit(X, y)

print("Entraînement du modèle de quantile 90%...")

model_upper = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LGBMRegressor(
        **{**BEST_PARAMS, "objective": "quantile", "alpha": 0.9}
    ))
])
model_upper.fit(X, y)

print("Modèles de quantile entraînés.")

# ----------------------------------------------------------
# Sérialisation du pipeline complet
# ----------------------------------------------------------

joblib.dump(
    {
        "model": model,
        "model_lower": model_lower,
        "model_upper": model_upper
    },
    "model_pipeline.joblib"
)

print("\nPipeline sauvegardé dans model_pipeline.joblib")

# ----------------------------------------------------------
# Options des menus déroulants pour l'application Streamlit
# ----------------------------------------------------------

brand_model = (
    df[["brand", "model"]]
    .dropna()
    .drop_duplicates()
    .groupby("brand")["model"]
    .apply(lambda s: sorted(s.unique().tolist()))
    .to_dict()
)

app_options = {
    "brand": sorted(df["brand"].dropna().unique().tolist()),
    "brand_model": brand_model,
    "vehicleType": sorted(df["vehicleType"].dropna().unique().tolist()),
    "gearbox": sorted(df["gearbox"].dropna().unique().tolist()),
    "fuelType": sorted(df["fuelType"].dropna().unique().tolist()),
    "notRepairedDamage": sorted(df["notRepairedDamage"].dropna().unique().tolist()),
    "powerPS": {
        "min": int(df["powerPS"].quantile(0.01)),
        "max": int(df["powerPS"].quantile(0.99)),
        "default": int(df["powerPS"].median())
    },
    "kilometer": {
        "min": int(df["kilometer"].min()),
        "max": int(df["kilometer"].max()),
        "default": int(df["kilometer"].median())
    },
    "age": {
        "min": int(df["age"].min()),
        "max": int(df["age"].max()),
        "default": int(df["age"].median())
    }
}

joblib.dump(app_options, "app_options.joblib")

print("Options des menus sauvegardées dans app_options.joblib")
print("\nEntraînement du modèle final terminé avec succès !")
