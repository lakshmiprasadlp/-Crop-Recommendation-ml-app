import streamlit as st
import numpy as np
import joblib

# Load the model and scalers
model = joblib.load('model.pkl')
sc = joblib.load('standscaler.pkl')
mx = joblib.load('minmaxscaler.pkl')

# Crop dictionary
crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
             8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
             14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
             19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

# Streamlit app
st.title('Crop Prediction App')

# Input fields for the features
N = st.number_input('Nitrogen', min_value=0, max_value=140, value=50)
P = st.number_input('Phosphorus', min_value=0, max_value=145, value=50)
K = st.number_input('Potassium', min_value=0, max_value=205, value=50)
temp = st.number_input('Temperature (°C)', min_value=0.0, max_value=50.0, value=25.0)
humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0, value=50.0)
ph = st.number_input('pH', min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input('Rainfall (mm)', min_value=0.0, max_value=300.0, value=100.0)

# Predict button
if st.button('Predict Crop'):
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)
    
    # Apply the scalers
    mx_features = mx.transform(single_pred)
    sc_mx_features = sc.transform(mx_features)
    
    # Make prediction
    prediction = model.predict(sc_mx_features)
    predicted_crop = crop_dict.get(prediction[0], "Unknown Crop")
    
    st.write(f'The recommended crop is: **{predicted_crop}**')

