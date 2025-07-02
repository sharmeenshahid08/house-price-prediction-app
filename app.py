#Importing libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#Loading dataset
@st.cache_data
def load_data():
    data = pd.read_csv("C:/Users/Sharmeen Shahid/Downloads/archive/Housing.csv")
    #Dropping less relavant featues
    data.drop(columns=["prefarea", "hotwaterheating", "airconditioning"], inplace=True)
    #Dropping missing values
    data.dropna(inplace=True)
    return data

data = load_data()

#Separating target(y) and X(features)
X = data.drop(columns=["price"])
y = data["price"]

#Converting categorical columns to numeric
X = pd.get_dummies(X, drop_first=True)

#Selecting the model
model = LinearRegression()

#Training the model
model.fit(X, y)

# --- App Title ---
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")
st.title("🏡 **House Price Prediction App**")
st.markdown("Welcome! Fill in the details below to estimate your house price using a trained ML model.")

# --- Sidebar Branding ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/616/616408.png", width=100)
    st.title("🔧 Settings")
    st.info("This app uses a Linear Regression model trained on real housing data.")

# --- Layout Using Columns ---
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("📐 Area (sq ft)", min_value=100.0, step=10.0, help="Total area of the house")
    bedrooms = st.number_input("🛏 Bedrooms", min_value=1, max_value=10, step=1)
    bathrooms = st.number_input("🛁 Bathrooms", min_value=1, max_value=10, step=1)
    stories = st.number_input("🏢 Stories", min_value=1, max_value=5, step=1)

with col2:
    parking = st.number_input("🚗 Parking spaces", min_value=0, max_value=5, step=1)
    mainroad = st.selectbox("🚦 Main Road?", ['yes', 'no'])
    guestroom = st.selectbox("🛋 Guest Room?", ['yes', 'no'])
    basement = st.selectbox("🏗 Basement?", ['yes', 'no'])
    furnish = st.selectbox("🪑 Furnishing Status", ['furnished', 'semi-furnished', 'unfurnished'])

# --- Prediction Button ---
st.markdown("### 🔮 Predict Price")
if st.button("✨ Show Estimated Price"):
    input_data = pd.DataFrame([{
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': stories,
        'parking': parking,
        'mainroad': mainroad,
        'guestroom': guestroom,
        'basement': basement,
        'furnishingstatus': furnish
    }])

    # Encode like training data
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    prediction = model.predict(input_data)[0]

    st.success(f"💰 **Estimated House Price: Rs. {round(prediction, 2):,}**")

# --- Footer ---
st.markdown("---")
st.markdown("Made | By **Sharmeen**")

