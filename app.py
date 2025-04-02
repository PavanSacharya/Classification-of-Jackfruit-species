import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model("trainedmodel.h5")
classes = ['DengSurya', 'IAHSManmohan', 'Prakashchandra', 'VietnamEarly']  # Replace with your actual classes

def load_and_predict_image(image_data, model, classes):
    byte_stream = np.asarray(bytearray(image_data.read()), dtype=np.uint8)
    original_image = cv2.imdecode(byte_stream, 1)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    processed_image = cv2.resize(original_image, (32, 32)) / 255.0

    prediction = model.predict(np.expand_dims(processed_image, axis=0))
    predicted_class_index = np.argmax(prediction)
    predicted_class = classes[predicted_class_index]

    return original_image, predicted_class

st.title('IMAGE CLASSIFICATION APP')
st.markdown("""
    <style>
    .stFileUploader {
        width: 60%; 
        margin-left: auto; 
        margin-right: auto;
        text-align: center;
    }
    .img-container {
        display: block; 
        margin-left: auto; 
        margin-right: auto; 
        width: 50%; 
        border: 2px solid black; 
        padding: 10px;
    }
    .prediction {
        color: #FFFFFF;
        font-size: 20px;
        font-family: Arial, Helvetica, sans-serif;
        text-align: center; 
        background-color: #333333;
        border: 2px solid black;
        margin-top: 5px;
        padding: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    original_image, predicted_class = load_and_predict_image(uploaded_file, model, classes)
    st.image(original_image, width=300, caption='Uploaded Image', output_format='PNG')
    st.markdown(f'<div class="prediction">Predicted Class: {predicted_class}</div>', unsafe_allow_html=True)

