import os.path

import streamlit as st
import numpy as np
import torch
import train_save
from torchvision import datasets
from PIL import Image

# We do this because when we are deploying the model, the location of the file will depend on when it's located.
# This is due also to how the docker files work
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/model.pth"

model = train_save.CNN()
model.load_state_dict(torch.load(model_path))
model.eval()

mnist_classes = datasets.MNIST.classes

def preprocess_image(image):
    # Resizing needed for mnist input
    img = image.resize((28, 28))
    img = img.convert("L")
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 1, 28, 28))
    img_tensor = torch.tensor(img_array, dtype=torch.float32)
    return img_tensor

# Start the definition of a streamlit app
st.title("Mnist digit classifier")

# We let the user upload an image
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
     image = Image.open(uploaded_image)

     # Creates 2 columns in the webpage
     col1, col2 = st.columns(2)

     # This is a view showing the user what he added to the webpage
     with col1:
         resized_img = image.resize((100, 100))
         st.image(resized_img)

     with col2:
         # if the user pushed the button
         if st.button("Classify"):
             # Preprocess the image
             img_array = preprocess_image(image)

             # Make a prediction using a pre-trained model
             result = model(img_array)[0]
             predicted_class = torch.argmax(torch.softmax(result, dim=-1))
             prediction = mnist_classes[predicted_class]

             st.success(f"Prediction: {prediction}")