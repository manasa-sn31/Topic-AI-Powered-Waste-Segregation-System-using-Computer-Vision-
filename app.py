import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load trained model
model = load_model("waste_model.h5")

# Class labels
classes = [
    "biological", "cardboard", "clothes",
    "glass", "metal", "organic",
    "paper", "plastic", "shoes", "trash"
]

# Page title
st.title("AI Powered Waste Segregation System")

st.write("Upload a waste image and the model will classify it.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Read image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = np.array(image)

    # Preprocess image
    img_resized = cv2.resize(img, (224, 224))
    img_normalized = img_resized / 255.0
    img_input = np.reshape(img_normalized, (1, 224, 224, 3))

    # Predict
    prediction = model.predict(img_input)
    class_index = np.argmax(prediction)
    result = classes[class_index]
    confidence = prediction[0][class_index] * 100

    # Show result
    st.success(f"Prediction: {result}")
    st.write(f"Confidence: {confidence:.2f}%")