# ==========================================================
# Analyse Exploratoire des Données (EDA)
# Projet : Prédiction du prix des voitures d'occasion
# ==========================================================

import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.stdout.reconfigure(encoding="utf-8")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# ----------------------------------------------------------
# Charger le dataset nettoyé
# ----------------------------------------------------------

df = pd.read_csv("autos_clean.csv")

print("=" * 60)
print("ANALYSE EXPLORATOIRE DES DONNÉES")
print("=" * 60)

# ----------------------------------------------------------
# Informations générales
# ----------------------------------------------------------

print("\n=== Aperçu du dataset ===")
print(df.head())

print("\n=== Dimensions du dataset ===")
print(df.shape)

print("\n=== Informations générales ===")
df.info()

print("\n=== Statistiques descriptives ===")
print(df.describe())

print("\n=== Statistiques des variables catégorielles ===")
print(df.describe(include="object"))

print("\n=== Valeurs manquantes ===")
print(df.isnull().sum())

# ----------------------------------------------------------
# Top 10 des marques
# ----------------------------------------------------------

print("\n=== Top 10 des marques ===")
print(df["brand"].value_counts().head(10))

plt.figure(figsize=(8,5))
df["brand"].value_counts().head(10).plot(kind="bar", color="steelblue")
plt.title("Top 10 des marques")
plt.xlabel("Marque")
plt.ylabel("Nombre de voitures")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# Distribution du prix
# ----------------------------------------------------------

plt.figure(figsize=(8,5))
plt.hist(df["price"], bins=50, edgecolor="black")
plt.title("Distribution du prix")
plt.xlabel("Prix (€)")
plt.ylabel("Nombre de voitures")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# Distribution du prix après transformation logarithmique
# ----------------------------------------------------------

df["log_price"] = np.log1p(df["price"])

plt.figure(figsize=(8,5))
plt.hist(df["log_price"], bins=50, edgecolor="black")
plt.title("Distribution du log(price)")
plt.xlabel("log(price)")
plt.ylabel("Nombre de voitures")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# Création de la variable âge
# ----------------------------------------------------------

df["age"] = 2016 - df["yearOfRegistration"]

# ----------------------------------------------------------
# Prix selon l'âge
# ----------------------------------------------------------

plt.figure(figsize=(8,5))
plt.scatter(df["age"], df["price"], alpha=0.2)
plt.title("Prix selon l'âge")
plt.xlabel("Âge (années)")
plt.ylabel("Prix (€)")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# Prix selon le kilométrage
# ----------------------------------------------------------

plt.figure(figsize=(8,5))
plt.scatter(df["kilometer"], df["price"], alpha=0.2)
plt.title("Prix selon le kilométrage")
plt.xlabel("Kilométrage")
plt.ylabel("Prix (€)")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# Marques les plus chères (prix médian)
# ----------------------------------------------------------

median_price = (
    df.groupby("brand")["price"]
      .median()
      .sort_values(ascending=False)
      .head(10)
)

plt.figure(figsize=(10,5))
median_price.plot(kind="bar", color="darkorange")
plt.title("Top 10 des marques selon le prix médian")
plt.xlabel("Marque")
plt.ylabel("Prix médian (€)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# Matrice de corrélation
# ----------------------------------------------------------

numeric = df[[
    "price",
    "powerPS",
    "kilometer",
    "age"
]]

plt.figure(figsize=(8,6))
sns.heatmap(
    numeric.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Matrice de corrélation")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# Valeurs aberrantes : Prix
# ----------------------------------------------------------

plt.figure(figsize=(8,4))
plt.boxplot(df["price"], vert=False)
plt.title("Valeurs aberrantes du prix")
plt.xlabel("Prix (€)")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# Valeurs aberrantes : Puissance
# ----------------------------------------------------------

plt.figure(figsize=(8,4))
plt.boxplot(df["powerPS"], vert=False)
plt.title("Valeurs aberrantes de la puissance (powerPS)")
plt.xlabel("Puissance (PS)")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# Valeurs aberrantes : Année d'immatriculation
# ----------------------------------------------------------

plt.figure(figsize=(8,4))
plt.boxplot(df["yearOfRegistration"], vert=False)
plt.title("Valeurs aberrantes de l'année d'immatriculation")
plt.xlabel("Année")
plt.tight_layout()
plt.show()

print("\nAnalyse exploratoire terminée avec succès !")