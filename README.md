# Meesho_Attributes_Predictions
Visual Taxonomy Classification and Submission Pipeline
This repository contains a pipeline for solving the Visual Taxonomy Classification problem, involving attribute prediction for images based on their category. The project leverages MobileNetV2 for feature extraction and a Random Forest Classifier for attribute prediction.

Table of Contents
Features
Requirements
Setup
Dataset
Usage
Model Training and Evaluation
Submission
Acknowledgments
Features
Custom Data Generator: Handles large-scale image datasets efficiently.
Feature Extraction: Uses pre-trained MobileNetV2 for extracting image features.
Multi-Label Classification: Predicts multiple attributes for each image.
Random Forest Classifier: Used for final classification based on extracted features.
Kaggle Submission Formatting: Generates a valid submission file for Kaggle competitions.
Requirements
Make sure you have the following installed:

Python 3.8 or later
Libraries:
TensorFlow
scikit-learn
pandas
numpy
matplotlib
Pillow
kagglehub
You can install the required libraries using:

bash
Copy code
pip install tensorflow scikit-learn pandas numpy matplotlib pillow kagglehub
Setup
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/visual-taxonomy-pipeline.git
cd visual-taxonomy-pipeline
Download the required datasets:

Visual Taxonomy Dataset: Available on Kaggle (link).
MobileNetV2 Weights: Download from MobileNetV2 Weights on Kaggle.
Place the datasets and weights in the following structure:

kotlin
Copy code
visual-taxonomy-pipeline/
├── data/
│   ├── train_images/
│   ├── test_images/
│   ├── train.csv
│   ├── test.csv
│   ├── category_attributes.parquet
│   └── mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128_no_top.h5
└── visual_taxonomy_pipeline.py
Dataset
This project uses the Visual Taxonomy Dataset, which includes:

Train and test images
Attribute information for categories in train.csv and category_attributes.parquet.
Usage
Run the Pipeline: Execute the Python script to preprocess data, extract features, train the model, and generate predictions:

bash
Copy code
python visual_taxonomy_pipeline.py
Verify Output:

Feature extraction and model training logs will appear in the console.
Submission file is generated at: submission.csv.
Model Training and Evaluation
Feature Extraction:
Uses pre-trained MobileNetV2 to extract feature vectors for images.
Classification:
A MultiOutputClassifier with a Random Forest base model is trained on the extracted features.
Evaluation Metrics:
Accuracy and F1-Score (Micro, Macro) are computed for validation data.
Submission
The final submission file is formatted as required for the Kaggle competition, ensuring:

Each product has the correct number of attributes predicted.
Default values (dummy_value) fill missing predictions.
Acknowledgments
Kaggle for hosting the competition and providing datasets.
TensorFlow for enabling deep learning workflows.
MobileNetV2 for pre-trained weights used in feature extraction.
Feel free to reach out via GitHub Issues for any questions or feedback.
