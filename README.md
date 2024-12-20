# 🍎 Fruit Recognition Website
link: https://fruitrecognition.streamlit.app/

This repository hosts the code for a Fruit Recognition Website built with Streamlit. Users can upload an image of a fruit, and a Convolutional Neural Network (CNN) deep learning model classifies the fruit into one of the predefined categories.

## Features
* Upload images of fruits via the web interface or take a picture using your device camera
* Get instant predictions using a trained CNN model
* User-friendly interface powered by Streamlit

## Tech Stack
* Frontend: Streamlit for an interactive web interface.
* Backend: PyTorch-based CNN for image classification.
* Languages: Python

## Model Architecture
This classification model is implemented using a CNN that contains 5 convolutional layers each followed by a max pooling layer, and one fully connected output layer at the end

## Setup Instructions
1. Go to your working directory and clone this repository: git clone https://github.com/bingdong45/FruitRecognition.git
2. Create a virtual environment and run in the working directory: pip install -r requirements.txt 
3. Run this to run the streamlit app: streamlit run ./website/main.py
4. Run the train.py file to train the model, a pth file for model weight will be generated in a ./weights folder

## Dataset
We used this dataset from kaggle: https://www.kaggle.com/datasets/karimabdulnabi/fruit-classification10-class?select=MY_data
It contains fruit images from 10 categories
