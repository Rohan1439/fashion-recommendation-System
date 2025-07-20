import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Load precomputed embeddings and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load the pre-trained ResNet50 model without the top layers
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Add a global max pooling layer for feature extraction
model = tensorflow.keras.Sequential([model, GlobalMaxPooling2D()])

st.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        os.makedirs('uploads', exist_ok=True)  # Ensure the upload directory exists
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# File upload and display
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display the uploaded image
        display_image = Image.open(uploaded_file)
        st.image(display_image)

        # Feature extraction from the uploaded image
        features = feature_extraction(os.path.join('uploads', uploaded_file.name), model)

        # Recommendation
        indices = recommend(features, feature_list)

        # Display recommended images
        cols = st.columns(5)  # Using `st.columns` instead of `st.beta_columns`
        for i, col in enumerate(cols):
            with col:
                st.image(filenames[indices[0][i]])
    else:
        st.header("Some error occurred in file upload")
