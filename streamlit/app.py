import streamlit as st
import numpy as np
from joblib import load

# Function to convert alphanumeric characters into numerical form
def encode_data(s):
    return np.array([ord(c) for c in s]).reshape(1, -1)

# Load the trained model
model = load('trained_model.joblib')

# Load the LabelEncoder used during training
le = load('label_encoder.joblib')  # Adjust the filename if needed

# Streamlit app
st.title('PAN Card Last Digit Predictor')

# User input
user_input = st.text_input('Enter the first digits of the PAN card number:', '')

if user_input:
    # Convert input to lowercase
    user_input = user_input.lower()
    
    # Encode the input and predict
    encoded_input = encode_data(user_input)
    prediction = model.predict(encoded_input)
    
    # Decode the predicted label back to the original class name, convert to string, and then to uppercase
    decoded_prediction = str(le.inverse_transform(prediction)[0]).upper()
    
    st.write(f'Predicted last digit: {decoded_prediction}')
else:
    st.write('Please enter the PAN card number.')
