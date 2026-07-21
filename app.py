# ==========================================================
# Application Streamlit - Estimation du prix d'une voiture d'occasion
# Étape VI du rapport : déploiement du modèle
# ==========================================================

from datetime import date

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
from fpdf import FPDF

st.set_page_config(
    page_title="Estimation du prix d'une voiture d'occasion",
    page_icon="🚗"
)

FRIENDLY_NUMERIC = {
    "powerPS": "Puissance (ch)",
    "kilometer": "Kilométrage",
    "age": "Âge"
}

FRIENDLY_CATEGORICAL = {
    "brand": "Marque",
    "vehicleType": "Type de véhicule",
    "gearbox": "Boîte de vitesses",
    "fuelType": "Carburant",
    "notRepairedDamage": "Dégât non réparé"
}


def get_feature_labels(preprocessor):
    """Reconstruit un libellé lisible pour chaque colonne produite par le
    ColumnTransformer (dans le même ordre que les colonnes transformées),
    en s'appuyant sur les catégories réellement apprises par l'encodeur
    plutôt que sur un découpage de chaîne de caractères (les marques comme
    'land_rover' contiennent déjà des underscores)."""
    labels = []

    numeric_features = preprocessor.transformers_[0][2]
    for feature in numeric_features:
        labels.append(FRIENDLY_NUMERIC.get(feature, feature))

    categorical_features = preprocessor.transformers_[1][2]
    encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    for column, categories in zip(categorical_features, encoder.categories_):
        column_label = FRIENDLY_CATEGORICAL.get(column, column)
        for category in categories:
            labels.append(f"{column_label} = {category}")

    return labels


def build_pdf_report(input_row, price, price_low, price_high, top_contrib):
    """Construit un PDF récapitulatif de l'estimation. Les polices de base
    de fpdf2 sont encodées en latin-1, qui ne contient pas le symbole '€' :
    on utilise donc 'EUR' dans le PDF (les accents français, eux, passent)."""
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Estimation du prix d'une voiture d'occasion", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(110, 110, 110)
    pdf.cell(0, 8, f"Genere le {date.today().strftime('%d/%m/%Y')}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Caracteristiques du vehicule", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)

    field_labels = {
        "brand": "Marque",
        "model": "Modele",
        "vehicleType": "Type de vehicule",
        "gearbox": "Boite de vitesses",
        "fuelType": "Carburant",
        "powerPS": "Puissance (ch)",
        "kilometer": "Kilometrage (km)",
        "notRepairedDamage": "Degat non repare"
    }
    for key, label in field_labels.items():
        pdf.cell(0, 7, f"{label} : {input_row[key]}", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Resultat", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, f"Prix estime : {price:,.0f} EUR".replace(",", " "), new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(
        0, 8,
        f"Intervalle de confiance (~80%) : {price_low:,.0f} EUR - {price_high:,.0f} EUR".replace(",", " "),
        new_x="LMARGIN", new_y="NEXT"
    )

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Pourquoi ce prix ?", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)

    for label, value in top_contrib.sort_values(ascending=False).items():
        sign = "+" if value >= 0 else ""
        pdf.cell(0, 6, f"{label} : {sign}{value:,.0f} EUR".replace(",", " "), new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())

# ----------------------------------------------------------
# Chargement du pipeline et des options des menus
# ----------------------------------------------------------

@st.cache_resource
def load_pipeline():
    return joblib.load("model_pipeline.joblib")


@st.cache_resource
def load_options():
    return joblib.load("app_options.joblib")


@st.cache_resource
def load_explainer(_pipeline):
    return shap.TreeExplainer(_pipeline["model"].named_steps["model"])


pipeline = load_pipeline()
options = load_options()
explainer = load_explainer(pipeline)

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

    vehicle_types_for_brand = options["brand_vehicleType"].get(brand, options["vehicleType"])
    vehicle_type = st.selectbox("Type de véhicule", vehicle_types_for_brand)

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

    # ------------------------------------------------------
    # Explication de la prédiction (SHAP)
    # ------------------------------------------------------

    preprocessor = pipeline["model"].named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X_input)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    shap_values = explainer.shap_values(X_transformed)[0]
    feature_labels = get_feature_labels(preprocessor)

    # Approximation : autour du point prédit, une variation de shap_value
    # (en échelle log-prix) se traduit par ~ prix * shap_value en euros.
    contrib = pd.DataFrame({
        "Caractéristique": feature_labels,
        "Impact (€)": price * shap_values
    })
    contrib["abs_impact"] = contrib["Impact (€)"].abs()
    top_contrib = (
        contrib.sort_values("abs_impact", ascending=False)
        .head(8)
        .drop(columns="abs_impact")
        .sort_values("Impact (€)")
        .set_index("Caractéristique")
    )

    st.subheader("Pourquoi ce prix ?")
    st.caption(
        "Impact estimé (approximatif) de chaque caractéristique du véhicule "
        "sur le prix prédit, par rapport à un véhicule moyen."
    )
    st.bar_chart(top_contrib, horizontal=True)

    # ------------------------------------------------------
    # Export du résultat (CSV / PDF)
    # ------------------------------------------------------

    export_data = input_data.copy()
    export_data["prix_estime_eur"] = round(price)
    export_data["intervalle_bas_eur"] = round(price_low)
    export_data["intervalle_haut_eur"] = round(price_high)

    pdf_bytes = build_pdf_report(
        input_data.iloc[0],
        price,
        price_low,
        price_high,
        top_contrib["Impact (€)"]
    )

    col_csv, col_pdf = st.columns(2)

    with col_csv:
        st.download_button(
            label="📥 Télécharger en CSV",
            data=export_data.to_csv(index=False).encode("utf-8"),
            file_name="estimation_prix.csv",
            mime="text/csv"
        )

    with col_pdf:
        st.download_button(
            label="📄 Télécharger en PDF",
            data=pdf_bytes,
            file_name="estimation_prix.pdf",
            mime="application/pdf"
        )
