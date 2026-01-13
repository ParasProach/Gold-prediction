import streamlit as st
import numpy as np
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium

# ----------------------------------------------------------
# MUST BE FIRST COMMAND
# ----------------------------------------------------------
st.set_page_config(page_title="Gold Predictor", page_icon="⛏️")

# ----------------------------------------------------------
# Load Feature List
# ----------------------------------------------------------
feature_list = joblib.load("model_columns.pkl")

# ----------------------------------------------------------
# Model Selection Dropdown
# ----------------------------------------------------------
model_choice = st.selectbox(
    "Choose Model",
    ["Random Forest", "XGBoost", "Logistic Regression", "Decision Tree"]
)

if model_choice == "Random Forest":
    model = joblib.load("model_rf.pkl")
elif model_choice == "XGBoost":
    model = joblib.load("model_xgb.pkl")
elif model_choice == "Logistic Regression":
    model = joblib.load("model_logreg.pkl")
else:
    model = joblib.load("model_dtree.pkl")

st.title("⛏️ Gold Deposit Prediction System")
st.write("Click a location on the map or enter manually.")

# ----------------------------------------------------------
# MAP LOCATION INPUT
# ----------------------------------------------------------
st.subheader("🌍 Select Location on Map")

m = folium.Map(location=[20, 0], zoom_start=2)
m.add_child(folium.LatLngPopup())
map_click = st_folium(m, width=700, height=450)

lat = 20.0
lon = 0.0

if map_click and map_click.get("last_clicked"):
    lat = map_click["last_clicked"]["lat"]
    lon = map_click["last_clicked"]["lng"]
    st.write(f"📍 Selected: **{lat:.6f}, {lon:.6f}**")

# Manual override
lat = st.number_input("Latitude", value=lat, format="%.6f")
lon = st.number_input("Longitude", value=lon, format="%.6f")

# ----------------------------------------------------------
# GEOLOGICAL INPUTS
# ----------------------------------------------------------
st.subheader("🪨 Geological Info")

dev_stat = st.selectbox("Development Status", ["Occurrence", "Producer"])

commodities = ["silver","copper","lead","zinc","iron","chromium","manganese","uranium","tungsten"]
commodity_input = {c: st.checkbox(f"{c.capitalize()} Present?") for c in commodities}

hrock = st.selectbox("Host Rock Type",
    ["unknown","granite","andesite","rhyolite","pegmatite","diorite",
     "basalt","gabbro","diabase","greenstone","limestone","dolomite",
     "sandstone","shale","siltstone","schist","slate","phyllite",
     "gneiss","quartzite","marble"])

arock = st.selectbox("Associated Rock Type",
    ["unknown","granite","monzonite","quartz monzonite","diorite",
     "gabbro","diabase","pegmatite","mafic intrusive rock",
     "andesite","rhyolite","basalt","tuff","dacite",
     "volcanic rock (aphanitic)","latite","quartz latite",
     "gneiss","greenstone"])

# ----------------------------------------------------------
# ROCK ENCODING
# ----------------------------------------------------------
def convert_arock(r):
    feats = {f"arock_class_{c}":0 for c in ["igneous_extrusive","igneous_intrusive","metamorphic","other","unknown"]}
    if r in ["granite","monzonite","quartz monzonite","diorite","gabbro","diabase","pegmatite","mafic intrusive rock"]:
        feats["arock_class_igneous_intrusive"] = 1
    elif r in ["andesite","rhyolite","basalt","tuff","dacite","volcanic rock (aphanitic)","latite","quartz latite"]:
        feats["arock_class_igneous_extrusive"] = 1
    elif r in ["gneiss","greenstone"]:
        feats["arock_class_metamorphic"] = 1
    elif r == "unknown":
        feats["arock_class_unknown"] = 1
    else:
        feats["arock_class_other"] = 1
    return feats

def convert_hrock(r):
    feats = {f"hrock_class_h_{c}":0 for c in
             ["igneous_felsic","igneous_intermediate","igneous_mafic",
              "sed_carbonate","sed_clastic","meta_foliated","meta_nonfoliated","unknown"]}

    if r in ["granite","rhyolite","pegmatite"]:
        feats["hrock_class_h_igneous_felsic"] = 1
    elif r in ["diorite","andesite","monzonite"]:
        feats["hrock_class_h_igneous_intermediate"] = 1
    elif r in ["basalt","gabbro","diabase","greenstone"]:
        feats["hrock_class_h_igneous_mafic"] = 1
    elif r in ["limestone","dolomite"]:
        feats["hrock_class_h_sed_carbonate"] = 1
    elif r in ["sandstone","shale","siltstone"]:
        feats["hrock_class_h_sed_clastic"] = 1
    elif r in ["schist","slate","phyllite","gneiss"]:
        feats["hrock_class_h_meta_foliated"] = 1
    elif r in ["quartzite","marble"]:
        feats["hrock_class_h_meta_nonfoliated"] = 1
    else:
        feats["hrock_class_h_unknown"] = 1
    return feats

# ----------------------------------------------------------
# FEATURE VECTOR BUILDING
# ----------------------------------------------------------
input_data = {
    "latitude": lat,
    "longitude": lon,
    "dev_stat": 1 if dev_stat == "Producer" else 0,
    "ctype_M": 1, "ctype_N": 0, "ctype_B": 0, "ctype_E": 0,
}

for c in commodities:
    input_data[f"c_{c}"] = 1 if commodity_input[c] else 0

input_data.update(convert_arock(arock))
input_data.update(convert_hrock(hrock))

input_df = pd.DataFrame([input_data])

# Ensure matching columns
missing = set(feature_list) - set(input_df.columns)
for col in missing:
    input_df[col] = 0

input_df = input_df[feature_list]

# ----------------------------------------------------------
# PREDICTION
# ----------------------------------------------------------
if st.button("🔍 Predict"):
    prob = model.predict_proba(input_df)[0][1] * 100
    pred = model.predict(input_df)[0]

    st.write("---")
    st.write(f"🤖 Model Used: **{model_choice}**")
    st.write(f"📊 Gold Probability: **{prob:.2f}%**")

    if pred == 1:
        st.success("🔥 High Potential for Gold")
    else:
        st.error("⚠️ Low Potential for Gold")
