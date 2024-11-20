Visual Taxonomy Classification and Submission Pipeline

This repository contains a pipeline for solving the Visual Taxonomy Classification problem, involving attribute prediction for images based on their category. The project leverages MobileNetV2 for feature extraction and a Random Forest Classifier for attribute prediction.

Table of Contents

1.	Features
   
2.	Requirements
   
3.	Setup
   
4.	Dataset
   
5.	Usage

6.	Model Training and Evaluation
	
7.	Submission
	
8.	Acknowledgments
________________________________________

Features

•	Custom Data Generator: Handles large-scale image datasets efficiently.
•	Feature Extraction: Uses pre-trained MobileNetV2 for extracting image features.
•	Multi-Label Classification: Predicts multiple attributes for each image.
•	Random Forest Classifier: Used for final classification based on extracted features.
•	Kaggle Submission Formatting: Generates a valid submission file for Kaggle competitions.
________________________________________
Requirements
Make sure you have the following installed:
•	Python 3.8 or later
•	Libraries:
o	TensorFlow
o	scikit-learn
o	pandas
o	numpy
o	matplotlib
o	Pillow
o	kagglehub
You can install the required libraries using:
pip install tensorflow scikit-learn pandas numpy matplotlib pillow kagglehub________________________________________
Setup
1.	Clone the repository:
git clone https://github.com/yourusername/visual-taxonomy-pipeline.git
cd visual-taxonomy-pipeline
2.	Download the required datasets:
o	Visual Taxonomy Dataset: Available on Kaggle (link).
o	MobileNetV2 Weights: Download from MobileNetV2 Weights on Kaggle or from my repository.
3.	Place the datasets and weights in the following structure:
visual-taxonomy-pipeline/
├── data/
│   ├── train_images/
│   ├── test_images/
│   ├── train.csv
│   ├── test.csv
│   ├── category_attributes.parquet
│   └── mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128_no_top.h5
└── visual_taxonomy_pipeline.py
________________________________________
Dataset
This project uses the Visual Taxonomy Dataset, which includes:
•	Train and test images
•	Attribute information for categories in train.csv and category_attributes.parquet.
________________________________________
Usage
1.	Run the Pipeline: Execute the Python script to preprocess data, extract features, train the model, and generate predictions:
python visual_taxonomy_pipeline.py
2.	Verify Output:
o	Feature extraction and model training logs will appear in the console.
o	Submission file is generated at: submission.csv.
________________________________________
Model Training and Evaluation
•	Feature Extraction:
o	Uses pre-trained MobileNetV2 to extract feature vectors for images.
•	Classification:
o	A MultiOutputClassifier with a Random Forest base model is trained on the extracted features.
•	Evaluation Metrics:
o	Accuracy and F1-Score (Micro, Macro) are computed for validation data.
________________________________________
Submission
The final submission file is formatted as required for the Kaggle competition, ensuring:
•	Each product has the correct number of attributes predicted.
•	Default values (dummy_value) fill missing predictions.
________________________________________
Acknowledgments
•	Kaggle for hosting the competition and providing datasets.
•	TensorFlow for enabling deep learning workflows.
•	MobileNetV2 for pre-trained weights used in feature extraction.
________________________________________
Feel free to reach out via GitHub Issues for any questions or feedback.
________________________________________
Replace yourusername in the GitHub URL with your actual username. Update file paths or links if necessary. This README ensures a smooth experience for users cloning and running your code!

