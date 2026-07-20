# ==========================================================
# Machine Learning - Préprocessing + Régression Linéaire
# Conforme au rapport
# ==========================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ----------------------------------------------------------
# Charger les données nettoyées
# ----------------------------------------------------------

df = pd.read_csv("autos_clean.csv")

print("=" * 60)
print("PRÉPARATION DES DONNÉES")
print("=" * 60)

# ----------------------------------------------------------
# Feature engineering : création de l'âge
# ----------------------------------------------------------

df["age"] = 2016 - df["yearOfRegistration"]

# ----------------------------------------------------------
# Variables explicatives
# ----------------------------------------------------------

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

# ----------------------------------------------------------
# Variable cible : log(price)
# ----------------------------------------------------------

y = np.log1p(df["price"])

print("\\nDimensions de X :", X.shape)
print("Dimensions de y :", y.shape)

# ----------------------------------------------------------
# Séparation Train / Test
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\\n==============================")
print("DÉCOUPAGE DES DONNÉES")
print("==============================")

print("Taille de X_train :", X_train.shape)
print("Taille de X_test :", X_test.shape)
print("Taille de y_train :", y_train.shape)
print("Taille de y_test :", y_test.shape)

# ==========================================================
# Préprocessing
# ==========================================================

# ----------------------------------------------------------
# Colonnes numériques
# ----------------------------------------------------------

numeric_features = [
    "powerPS",
    "kilometer",
    "age"
]

# ----------------------------------------------------------
# Colonnes catégorielles
# ----------------------------------------------------------

categorical_features = [
    "brand",
    "vehicleType",
    "gearbox",
    "fuelType",
    "notRepairedDamage"
]

# ----------------------------------------------------------
# Pipeline numérique
# ----------------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

# ----------------------------------------------------------
# Pipeline catégoriel
# ----------------------------------------------------------

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="Unknown"
            )
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

# ----------------------------------------------------------
# ColumnTransformer
# ----------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)

# ==========================================================
# Pipeline complet + Régression Linéaire
# ==========================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        ("model", LinearRegression())
    ]
)

# ----------------------------------------------------------
# Entraînement
# ----------------------------------------------------------

print("\nEntraînement du Decision Tree...")

model.fit(X_train, y_train)

print("Entraînement terminé.")

# ----------------------------------------------------------
# Prédictions (échelle logarithmique)
# ----------------------------------------------------------

y_pred_log = model.predict(X_test)

# ----------------------------------------------------------
# Retour à l'échelle réelle (euros)
# ----------------------------------------------------------

y_pred = np.expm1(y_pred_log)
y_test_real = np.expm1(y_test)

# ----------------------------------------------------------
# Évaluation
# ----------------------------------------------------------

mae = mean_absolute_error(y_test_real, y_pred)
rmse = mean_squared_error(y_test_real, y_pred) ** 0.5
r2 = r2_score(y_test_real, y_pred)

print("\\n==============================")
print("RÉSULTATS - RÉGRESSION LINÉAIRE")
print("==============================")

print(f"MAE  : {mae:.2f} €")
print(f"RMSE : {rmse:.2f} €")
print(f"R²   : {r2:.3f}")
print("\nPréprocessing + baseline terminés avec succès !")
