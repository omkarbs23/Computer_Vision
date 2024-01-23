import streamlit as st
import cv2
import numpy as np
import pickle
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Loading Model
from keras.models import load_model
model = load_model('my_model.keras')

# Loading LabelEncoder()
with open('label_encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)


st.set_page_config(page_title="Fruit vs Vegetable Recognition App", page_icon="🍒")  #
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

def process_image(image_data):    
    # Check if the image is loaded properly
    if image_data is None:
        return "Error: The image could not be processed."
    
    p1 = cv2.resize(image_data, (300, 300))
    p1 = p1 / 255
    p1 = np.array([p1])
    
    # Predicting the image is Fruit or Vegetable
    prediction = model.predict(p1)
    probability = np.argmax(prediction)
    output = encoder.classes_[probability]
    
    return output

st.title('Fruit vs Vegetable Recognition App')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


st.divider()
if uploaded_file is not None:
    # Display the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    st.image(opencv_image, caption=f'{uploaded_file.name}',  width=500 , channels='BGR')
        
    # get classes
    result = process_image(opencv_image)

    # Display the output of the function
    st.header(f"Fruit in Image is {result}")
       
            
