import pandas as pd
print("Bonjour Hajar, c'est le nouveau programme !")

# Lire le fichier
df = pd.read_csv("autos.csv")

# Informations générales
print("Dimensions avant nettoyage :", df.shape)
print("\nValeurs manquantes :")
print(df.isnull().sum())

# Supprimer les doublons
df.drop_duplicates(inplace=True)

# Supprimer les colonnes constantes/quasi constantes et inutiles à la prédiction
df.drop(columns=[
    "index",
    "nrOfPictures",
    "seller",
    "offerType",
    "dateCrawled",
    "dateCreated",
    "lastSeen",
    "postalCode",
    "monthOfRegistration"
], inplace=True)

# Remplacer les valeurs manquantes
df["vehicleType"] = df["vehicleType"].fillna("Unknown")
df["gearbox"] = df["gearbox"].fillna("Unknown")
df["model"] = df["model"].fillna("Unknown")
df["fuelType"] = df["fuelType"].fillna("Unknown")
df["notRepairedDamage"] = df["notRepairedDamage"].fillna("Unknown")

# Supprimer les années incohérentes
df = df[
    (df["yearOfRegistration"] >= 1950) &
    (df["yearOfRegistration"] <= 2016)
]

# Supprimer les prix incohérents
df = df[
    (df["price"] > 100) &
    (df["price"] < 150000)
]

# Transformer les puissances nulles en valeurs manquantes
df.loc[df["powerPS"] == 0, "powerPS"] = pd.NA

# Supprimer uniquement les puissances irréalistes (on garde les valeurs
# manquantes pour qu'elles soient imputées plus tard par le pipeline)
df = df[(df["powerPS"] < 1000) | (df["powerPS"].isna())]

# Sauvegarder le fichier nettoyé
df.to_csv("autos_clean.csv", index=False)

print("\nNettoyage terminé !")
print("Dimensions après nettoyage :", df.shape)
df.head()    