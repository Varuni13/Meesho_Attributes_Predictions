# Visual Taxonomy Classification and Submission Pipeline

This repository contains a pipeline for solving the Visual Taxonomy Classification problem, involving attribute prediction for images based on their category. The project leverages MobileNetV2 for feature extraction and a Random Forest Classifier for attribute prediction.

## Table of Contents
1. [Features](#features)
2. [Requirements](#requirements)
3. [Setup](#setup)
4. [Dataset](#dataset)
5. [Usage](#usage)
6. [Model Training and Evaluation](#model-training-and-evaluation)
7. [Inference](#inference)
8. [Submission](#submission)
9. [Acknowledgments](#acknowledgments)
________________________________________

## Features

- **Custom Data Generator:** Efficiently handles large-scale image datasets for training.
- **Feature Extraction:** Uses pre-trained MobileNetV2 for extracting image features.
- **Multi-Label Classification:** Predicts multiple attributes for each image (multi-output classification).
- **Random Forest Classifier:** Final classifier for attribute prediction based on extracted features.
- **Kaggle Submission Formatting:** Generates a valid submission file formatted for Kaggle competitions.
________________________________________

## Requirements

Make sure you have the following installed:

- Python 3.8 or later

### Libraries:
- TensorFlow
- scikit-learn
- pandas
- numpy
- matplotlib
- Pillow

To install all dependencies, run:


pip install -r requirements.txt



### Setup

1.	Clone the repository:
   
	git clone https://github.com/Varuni13/Meesho_Attributes_Predictions.git

	cd Meesho_Attributes_Predictions


3. Download the required datasets:
   
- Visual Taxonomy Dataset: Available on Kaggle.
	
- MobileNetV2 Weights: Download from google or from my repository.


4. Place the datasets and weights in the following structure:
   
	Meesho_Attributes_Predictions/

	├── data/
	
	│   ├── train_images/
	
	│   ├── test_images/
	
	│   ├── train.csv
	
	│   ├── test.csv
	
	│   ├── category_attributes.parquet
	
	│   └── mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128_no_top.h5
	
	└── visual_taxonomy_pipeline.py


## Dataset

This project uses the Visual Taxonomy Dataset, which includes:

- Train and test images

- Attribute information for categories in train.csv and category_attributes.parquet.


## Usage

1. Run the Pipeline:

   Execute the Python script to preprocess data, extract features, train the model, and generate  predictions:

 	```bash
	python visual_taxonomy_pipeline.py


2. Verify Output:
    
    - Feature extraction and model training logs will appear in the console.
	
    - Submission file is generated at: submission.csv.

3. Notes on Model Weights:

   The pre-trained MobileNetV2 weights (mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128_no_top.h5) must be downloaded separately and placed in the appropriate directory (/data/).


## Model Training and Evaluation

1. Feature Extraction : We use a pre-trained MobileNetV2 model (with the top layers removed)
to extract feature vectors from the images. These features are then used for classification.
	
2. Classification: A MultiOutputClassifier with a Random Forest base model is trained on the extracted features.This 	allows  the model to predict multiple attributes for each image.

3. Evaluation Metrics :Accuracy and F1-Score (both Micro and Macro) are computed on the validation data to evaluate 	model performance.

##Inference

1. Once the model is trained, the following steps are used to generate predictions for the test dataset:
	
2. Feature Extraction: The test images are processed using the same feature extraction pipeline.
	
3. Prediction: The trained Random Forest classifier predicts the attributes for each product.
	
4. Submission Formatting: The predictions are formatted into a submission.csv file, as required by the Kaggle competition.

To run the inference script, use:
	```bash 
          Python inference.py

This will:

- Load the pre-trained model and the test data.

- Process the test images.
	
- Predict the attributes for each product.
	
- Format the predictions into a Kaggle-compatible CSV file (submission.csv).

## Submission

The generated submission file is formatted according to the competition requirements:

- Each product in the test set has the correct number of attributes predicted.
	
- Missing attributes are filled with the default value (dummy_value).

The final submission file (submission.csv) will be saved in the current directory.


### Acknowledgments

 1. Kaggle for hosting the competition and providing the datasets.
	
 2. TensorFlow for enabling deep learning workflows.
	
 3. MobileNetV2 for providing pre-trained weights used in feature extraction.
	
 4. scikit-learn for supporting multi-output classification.

Feel free to reach out via GitHub Issues for any questions or feedback.



