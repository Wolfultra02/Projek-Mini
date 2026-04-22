import streamlit as st
import numpy as np
import joblib

# Load model & scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Prediksi Penyakit Kulit")

st.write("Masukkan nilai klinis (0–3) sesuai kondisi pasien")

# =========================
# INPUT SEMUA FITUR (34)
# =========================
features = []

features.append(st.slider("erythema", 0, 3, 1))
features.append(st.slider("scaling", 0, 3, 1))
features.append(st.slider("definite_borders", 0, 3, 1))
features.append(st.slider("itching", 0, 3, 1))
features.append(st.slider("koebner_phenomenon", 0, 3, 0))
features.append(st.slider("polygonal_papules", 0, 3, 0))
features.append(st.slider("follicular_papules", 0, 3, 0))
features.append(st.slider("oral_mucosal_involvement", 0, 3, 0))
features.append(st.slider("knee_and_elbow_involvement", 0, 3, 0))
features.append(st.slider("scalp_involvement", 0, 3, 0))
features.append(st.selectbox("family_history", [0,1]))
features.append(st.slider("melanin_incontinence", 0, 3, 0))
features.append(st.slider("eosinophils_infiltrate", 0, 3, 0))
features.append(st.slider("PNL_infiltrate", 0, 3, 0))
features.append(st.slider("fibrosis_papillary_dermis", 0, 3, 0))
features.append(st.slider("exocytosis", 0, 3, 0))
features.append(st.slider("acanthosis", 0, 3, 0))
features.append(st.slider("hyperkeratosis", 0, 3, 0))
features.append(st.slider("parakeratosis", 0, 3, 0))
features.append(st.slider("clubbing_rete_ridges", 0, 3, 0))
features.append(st.slider("elongation_rete_ridges", 0, 3, 0))
features.append(st.slider("thinning_suprapapillary_epidermis", 0, 3, 0))
features.append(st.slider("spongiform_pustule", 0, 3, 0))
features.append(st.slider("munro_microabcess", 0, 3, 0))
features.append(st.slider("focal_hypergranulosis", 0, 3, 0))
features.append(st.slider("disappearance_granular_layer", 0, 3, 0))
features.append(st.slider("vacuolisation_damage_basal_layer", 0, 3, 0))
features.append(st.slider("spongiosis", 0, 3, 0))
features.append(st.slider("saw_tooth_appearance_retes", 0, 3, 0))
features.append(st.slider("follicular_horn_plug", 0, 3, 0))
features.append(st.slider("perifollicular_parakeratosis", 0, 3, 0))
features.append(st.slider("inflammatory_mononuclear_infiltrate", 0, 3, 1))
features.append(st.slider("band_like_infiltrate", 0, 3, 0))
features.append(st.number_input("age", 0, 100, 30))

# =========================
# LABEL PENYAKIT
# =========================
label_penyakit = {
    1: "Psoriasis",
    2: "Seborrheic Dermatitis",
    3: "Lichen Planus",
    4: "Pityriasis Rosea",
    5: "Chronic Dermatitis",
    6: "Pityriasis Rubra Pilaris"
}

# =========================
# PREDIKSI
# =========================
if st.button("Prediksi"):
    input_data = np.array([features])
    input_scaled = scaler.transform(input_data)

    pred = model.predict(input_scaled)[0]

    st.success(f"Hasil Prediksi: {label_penyakit[int(pred)]}")
