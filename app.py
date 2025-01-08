import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the LabelEncoder
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Load the OneHotEncoder
with open('onehot_encoded_season.pkl', 'rb') as file:
    onehot_encoder_season = pickle.load(file)
    if not isinstance(onehot_encoder_season, OneHotEncoder):
        raise ValueError("The file 'onehot_encoded.pkl' does not contain a valid OneHotEncoder instance.")

# Load the StandardScaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app title
st.title('Crop Recommendation System')

# User inputs
Nitrogen = st.slider('Nitrogen (N)', 0, 140)
Phosphorous = st.slider('Phosphorous (P)', 5, 145)
Potassium = st.slider('Potassium (K)', 5, 205)
Humidity = st.number_input('Humidity (%)', min_value=14.25, max_value=100.00, step=0.01)
season = st.selectbox('Season', list(onehot_encoder_season.categories_[0]))
ph = st.number_input('pH Level', min_value=0.00, max_value=14.00, step=0.01)
rainfall = st.number_input('Rainfall (mm)', min_value=20.00, max_value=300.00, step=0.01)

# Prepare input data
input_data = pd.DataFrame({
    'N': [Nitrogen],
    'P': [Phosphorous],
    'K': [Potassium],
    'humidity': [Humidity],
    'ph': [ph],
    'rainfall': [rainfall]
})

season_encoder=onehot_encoder_season.transform([[season]]).toarray()
season_encoder_df=pd.DataFrame(season_encoder,columns=onehot_encoder_season.get_feature_names_out(['season']))

input_data=pd.concat([input_data.reset_index(drop=True),season_encoder_df],axis=1)


# Scale the input data
input_scaled = scaler.transform(input_data)

# Make predictions
prediction = model.predict(input_scaled)

# Get the predicted class
predicted_index = np.argmax(prediction)

# Map the predicted index to the class name
class_names = [
    "apple", "banana", "blackgram", "chickpea", "coconut", "coffee", "cotton", 
    "grapes", "jute", "kidneybeans", "lentil", "maize", "mango", "mothbeans", 
    "mungbeans", "muskmelon", "orange", "papaya", "pigeonpeas", "pomegranate", 
    "rice", "watermelon"
]
predicted_class = class_names[predicted_index]

# Display the predicted class
st.markdown(
    f"<h2 style='color:green;'>Predicted Crop: <b>{predicted_class}</b></h2>",
    unsafe_allow_html=True
)

