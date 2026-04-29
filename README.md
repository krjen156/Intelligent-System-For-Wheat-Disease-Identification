# Intelligent-System-For-Wheat-Disease-Identification
An interactive web-based application for detecting wheat diseases using deep learning models. The system allows users to upload images, compare multiple CNN models, and visualize how predictions are made using Grad-CAM.


## 📌 Project Overview

Wheat diseases can significantly impact crop yield and food production. This project applies Convolutional Neural Networks (CNNs) to classify wheat diseases from images and provides an accessible tool for both detection and education.

The system integrates multiple trained models and allows users to:

Upload wheat images
Compare predictions across models
View confidence scores
Understand model decisions through visual explanation

## Project Structure:
- WPD contains the jupyter notebooks for training ResNet50, VGG16, InceptionV3, MobileNetV2 and WheatNetwork on the WPD dataset. ResNet50, VGG16 and WheatNetwork are also tested on a final evaluation dataset in FINAL_EVALUATION. The images used for FINAL_EVALUATION are stored in a excel shcema within the folder. 
- LWDC contrains the jupyter notebooks for training ResNet50, VGG16, InceptionV3, MobileNetV2 and WheatNetwork on the LWDC dataset. ResNet50, VGG16 and WheatNetwork are also tested on a final evaluation dataset in FINAL_EVALUATION. Images used to evaluate the models on the FINAL_EVALUATION were random images from WPD dataset (within the same classes)
- system APPLICATION contrains the .py files for Intelligent System for Wheat Disease Identification
- Intelligent System for Wheat Disease Identification.mp4 is a video review of the application. 


##  Features of the application
- Multi-model comparison (ResNet50, VGG16, WheatNetwork)
- Image upload (JPG, PNG, JPEG)
- Prediction confidence and class probabilities
- Grad-CAM visualization (model interpretability)
- Web-based interface (Streamlit)

##  Datasets
WPD Dataset – Multi-class wheat disease dataset (Kaggle) - https://www.kaggle.com/datasets/kushagra3204/wheat-plant-diseases
LWDC Dataset – Simplified dataset with fewer classes - https://www.kaggle.com/datasets/taibariaz/large-wheat-disease-classification-dataset

These datasets allow comparison between:

Complex classification (many diseases)
Generalized classification (fewer disease groups)


## Installation (Bash)
1. git clone https://github.com/krjen156/Intelligent-System-For-Wheat-Disease-Identification.git
2. cd Intelligent-System-For-Wheat-Disease-Identification/system APPLICATION
4. python -m venv .venv
5. .venv\Scripts\activate   # Windows
6. pip install -r requirements.txt
7. streamlit run app.py
Nb. You must train the models first (.keras). They were to big to include on GitHub.

## Limitations
❗ Only works with wheat plant images
❗ Does not verify if the image contains wheat
❗ Performance depends on image quality and similarity to training data
❗ Models are static (no real-time training)

## Future Work
Real-time mobile deployment
Integration with weather/soil data
Larger and more diverse datasets
Improved generalization across environments

This project is part of a bachelor thesis.
The full paper (methodology, models, and results) can be found in the repository.

