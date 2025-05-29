import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("xgboost_final_model.pkl")
df_raw = pd.read_csv("autoscout_clean.csv")
expected_columns = pd.read_csv("X_encoded.csv", nrows=1).columns.tolist()


make_model_split = df_raw['make_model'].dropna().str.extract(r'^(\w+)\s+(.+)$')
df_raw['make'] = make_model_split[0]
df_raw['model'] = make_model_split[1]

st.set_page_config(page_title=" Used Car Price Estimator", page_icon="🚗", layout="wide")


st.title("Used Car Price Estimator")
st.markdown("Enter vehicle details to estimate its **market price** using a machine learning model.")


make_list = sorted(df_raw['make'].dropna().unique())
make = st.selectbox("Brand", make_list)

model_list = sorted(df_raw[df_raw['make'] == make]['model'].dropna().unique())
model_selected = st.selectbox("Model", model_list)


filtered_df = df_raw[(df_raw['make'] == make) & (df_raw['model'] == model_selected)]


st.subheader("🛠️ Technical Features")
col1, col2, col3 = st.columns(3)

with col1:
    body_type = st.selectbox("Body Type", sorted(filtered_df['body_type'].dropna().unique()))
    fuel = st.selectbox("Fuel Type", sorted(filtered_df['fuel'].dropna().unique()))
with col2:
    gearing = st.selectbox("Transmission", sorted(filtered_df['gearing_type'].dropna().unique()))
    upholstery = st.selectbox("Upholstery", sorted(filtered_df['upholstery_type'].dropna().unique()))
with col3:
    inspection_new = st.checkbox("Recently Inspected (Tech Control)", value=True)


st.subheader("📊 Numeric Information")
col4, col5, col6 = st.columns(3)

with col4:
    age = st.number_input("Vehicle Age (years)", 0, 30, 3)
    gears = st.number_input("Number of Gears", 3, 8, 6)

with col5:
    km = st.number_input("Mileage (km)", 0, 500000, 60000, step=5000)
    weight = st.number_input("Weight (kg)", 800, 3000, 1200)

with col6:
    hp_kw = st.number_input("Power (kW)", 20, 500, 85)
    displacement = st.number_input("Displacement (cc)", 800, 5000, 1600)
    owners = st.number_input("Previous Owners", 1, 10, 1)


if st.button("🔍 Estimate Price"):
    input_data = {
        'age': age,
        'km': km,
        'hp_kw': hp_kw,
        'gears': gears,
        'weight_kg': weight,
        'displacement_cc': displacement,
        'previous_owners': owners,
        'inspection_new': int(inspection_new),
    }

    
    for col, val in {
        **{f"body_type_{x}": x == body_type for x in df_raw['body_type'].dropna().unique()},
        **{f"fuel_{x}": x == fuel for x in df_raw['fuel'].dropna().unique()},
        **{f"gearing_type_{x}": x == gearing for x in df_raw['gearing_type'].dropna().unique()},
        **{f"upholstery_type_{x}": x == upholstery for x in df_raw['upholstery_type'].dropna().unique()},
        **{f"make_{x}": x == make for x in make_list},
        **{f"model_{x}": x == model_selected for x in df_raw['model'].dropna().unique()},
    }.items():
        input_data[col] = int(val)

    df_input = pd.DataFrame([input_data])

    
    for col in expected_columns:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[expected_columns]

    
    log_price = model.predict(df_input)[0]
    predicted_price = np.exp(log_price)

    
    st.success(f"💰 **Estimated Price: €{predicted_price:,.0f}**")
