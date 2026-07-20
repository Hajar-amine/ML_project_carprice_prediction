# ==========================================================
# Application Streamlit - Estimation du prix d'une voiture d'occasion
# Étape VI du rapport : déploiement du modèle
# ==========================================================

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Estimation du prix d'une voiture d'occasion",
    page_icon="🚗"
)

# ----------------------------------------------------------
# Chargement du pipeline et des options des menus
# ----------------------------------------------------------

@st.cache_resource
def load_pipeline():
    return joblib.load("model_pipeline.joblib")


@st.cache_resource
def load_options():
    return joblib.load("app_options.joblib")


pipeline = load_pipeline()
options = load_options()

# ----------------------------------------------------------
# En-tête
# ----------------------------------------------------------

st.title("🚗 Estimation du prix d'une voiture d'occasion")
st.write("Renseignez les caractéristiques du véhicule pour obtenir une estimation de prix.")

st.header("Caractéristiques du véhicule")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Marque", options["brand"])

    models_for_brand = options["brand_model"].get(brand, ["Unknown"])
    model_name = st.selectbox("Modèle", models_for_brand)

    vehicle_type = st.selectbox("Type de véhicule", options["vehicleType"])

    gearbox = st.selectbox("Boîte de vitesses", options["gearbox"])

with col2:
    fuel_type = st.selectbox("Carburant", options["fuelType"])

    not_repaired_damage = st.selectbox("Dégât non réparé", options["notRepairedDamage"])

    power_ps = st.slider(
        "Puissance (ch)",
        min_value=options["powerPS"]["min"],
        max_value=options["powerPS"]["max"],
        value=options["powerPS"]["default"]
    )

kilometer = st.slider(
    "Kilométrage (km)",
    min_value=options["kilometer"]["min"],
    max_value=options["kilometer"]["max"],
    value=options["kilometer"]["default"],
    step=5000
)

age = st.slider(
    "Âge du véhicule (années)",
    min_value=options["age"]["min"],
    max_value=options["age"]["max"],
    value=options["age"]["default"]
)

# ----------------------------------------------------------
# Récapitulatif des données saisies
# ----------------------------------------------------------

input_data = pd.DataFrame([{
    "vehicleType": vehicle_type,
    "gearbox": gearbox,
    "powerPS": power_ps,
    "model": model_name,
    "kilometer": kilometer,
    "fuelType": fuel_type,
    "brand": brand,
    "notRepairedDamage": not_repaired_damage,
    "age": age
}])

st.header("Données saisies")
st.dataframe(
    input_data[[
        "vehicleType", "gearbox", "powerPS", "model",
        "kilometer", "fuelType", "brand", "notRepairedDamage"
    ]],
    hide_index=True
)

# ----------------------------------------------------------
# Prédiction
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

if st.button("Estimer le prix"):
    X_input = input_data[features]

    price_log = pipeline["model"].predict(X_input)[0]
    price_low_log = pipeline["model_lower"].predict(X_input)[0]
    price_high_log = pipeline["model_upper"].predict(X_input)[0]

    price = np.expm1(price_log)
    price_low = np.expm1(price_low_log)
    price_high = np.expm1(price_high_log)

    st.header("Résultat")

    st.metric("Prix estimé", f"{price:,.0f} €".replace(",", " "))

    st.write(
        f"**Intervalle de confiance (~80%)** : "
        f"{price_low:,.0f} € — {price_high:,.0f} €".replace(",", " ")
    )

    st.progress(
        min(max((price - price_low) / (price_high - price_low), 0.0), 1.0)
    )

    st.caption(
        "Estimation basée sur un modèle LightGBM entraîné sur "
        f"~{342_682:,} annonces.".replace(",", " ")
    )
