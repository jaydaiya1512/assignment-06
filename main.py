import streamlit as st
import numpy as np
import pickle
import os

# Load the trained model from the same repository
model_url = os.path.join(os.path.dirname(__file__), "xgboost_model.pkl")

st.set_page_config(page_title="Estimate Delivery Time", layout="wide")

@st.cache_resource()
def load_model():
    with open(model_url, "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()



# Main Container
with st.container():
    st.title("📦 Estimate Delivery Time Prediction")

    # Input fields
    col1, col2 = st.columns(2)

    with col1:
        purchase_day = st.number_input("Purchase Day", min_value=1, max_value=7, value=2)
        purchase_month = st.number_input("Purchase Month", min_value=1, max_value=12, value=6)
        year = st.number_input("Year", min_value=2000, max_value=2025, value=2025)
        geolocation_state_seller = st.number_input("Geolocation State (Seller)", min_value=0, max_value=26, value=20)

    with col2:
        product_size_cm3 = st.number_input("Product Size (cm³)", min_value=1, value=10000)
        product_weight_g = st.number_input("Product Weight (g)", min_value=1, value=500)
        distance = st.number_input("Distance (km)", min_value=0.1, value=300.5)
        geolocation_state_customer = st.number_input("Geolocation State (Customer)", min_value=0, max_value=26, value=10)

    
    
    

    # Button to predict estimated wait time
    if st.button("Calculate Estimated Time"):
        input_features = np.array([[
            purchase_day, purchase_month, year, product_size_cm3,
            product_weight_g, geolocation_state_customer,
            geolocation_state_seller, distance
        ]])
        
        prediction = model.predict(input_features)
        estimated_days = round(prediction[0], 2)
        
        st.success(f"📅 Estimated Delivery Time: {estimated_days} days")
        
