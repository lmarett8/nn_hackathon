import streamlit as st
import pickle
import pandas as pd 
import numpy as np
import pathlib
import random
import tensorflow as tf
from tensorflow import keras

from PIL import Image

st.title('Hotdog or not Hotdog?')

st.subheader('\~To hotdog or not to hotdog that is the question\~')
       
# Grabs a test image

def get_random_image():
    """randomly generate an image from one of the many images in the test folders
"""
    path_list = ['test/hotdog/', 'test/nothotdog/']
    image_list = []
    
    for path in path_list:
        image_dir = pathlib.Path(f'./data/hotdog-nothotdog/{path}')

        image_list.extend(list(image_dir.glob('*.jpg')))
        
    return random.choice(image_list)

#Loads the model

def load_model():
    the_model = keras.models.load_model("model.p")
    return the_model


model = load_model()

# Predicting the image

if st.button('Is it a hotdog?'):
    
    # Getting random image to predict and displaying it
    
    image_path = get_random_image()
    image = Image.open(image_path)
    st.image(image, caption ='random image')
    st.write(image_path)

    # transform that data into something the model can use
    
    image_matrix = np.asarray(image)
    image_matrix = np.reshape(image_matrix, newshape=(1,299, 299, 3))
    
    # makes a prediction and reports back to the page
    pred = int(model.predict(image_matrix)[0][0])
    prob = '' if pred == 1 else 'not '
    st.write(f'Looks like it\'s {prob}a hotdog')



