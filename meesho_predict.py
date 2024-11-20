# -*- coding: utf-8 -*-

# IMPORTANT: SOME KAGGLE DATA SOURCES ARE PRIVATE
# RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES.
import kagglehub
kagglehub.login()

# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

visual_taxonomy_path = kagglehub.competition_download('visual-taxonomy')
varuni13_mobilenetv2_path = kagglehub.dataset_download('varuni13/mobilenetv2')

print('Data source import complete.')

# Required Libraries
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import Sequence
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, f1_score



# Define folder paths for train and test images and paths for CSV files
train_images_folder = '/kaggle/input/visual-taxonomy/train_images'
test_images_folder = '/kaggle/input/visual-taxonomy/test_images'
train_csv_path = '/kaggle/input/visual-taxonomy/train.csv'
test_csv_path = '/kaggle/input/visual-taxonomy/test.csv'

# Load CSV files into DataFrames
train_data = pd.read_csv(train_csv_path)
test_data = pd.read_csv(test_csv_path)

print(train_data.head())
print(test_data.head())

# Custom Image Data Generator
class CustomImageDataGenerator(Sequence):
    def __init__(self, image_folder, image_ids, batch_size=32, target_size=(12_8, 128)):
        self.image_folder = image_folder
        self.image_ids = image_ids
        self.batch_size = batch_size
        self.target_size = target_size

    def __len__(self):
        return (len(self.image_ids) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        batch_ids = self.image_ids[idx * self.batch_size: (idx + 1) * self.batch_size]
        images = []
        for img_id in batch_ids:
            img_id_str = f"{int(img_id):06d}"
            img_path = os.path.join(self.image_folder, f"{img_id_str}.jpg")
            try:
                img = Image.open(img_path).resize(self.target_size)
                img = np.array(img) / 255.0  # Normalize pixel values
                if img.shape == (self.target_size[0], self.target_size[1], 3):
                    images.append(img)
                else:
                    print(f"[Warning] Image {img_path} has unexpected shape {img.shape}. Skipping.")
            except FileNotFoundError:
                print(f"[Warning] Image {img_path} not found.")
                continue
        print(f"Processed {len(images)} images in batch {idx + 1}")
        return np.array(images) if images else np.zeros((self.batch_size, *self.target_size, 3))

train_image_ids = train_data['id'].tolist()
train_generator = CustomImageDataGenerator(train_images_folder, train_image_ids, batch_size=32)

# Verify the generator
sample_images = train_generator[0]  # Get the first batch
print("Sample images shape:", sample_images.shape)  # Should be (batch_size, 128, 128, 3)

# Load and preprocess the entire training dataset
train_image_ids = train_data['id'].tolist()
train_labels = train_data[['attr_1', 'attr_2', 'attr_3', 'attr_4', 'attr_5',
                            'attr_6', 'attr_7', 'attr_8', 'attr_9', 'attr_10']].values

# Clean labels
train_labels_cleaned = []
for row in train_labels:
    cleaned_row = [str(x) for x in row if pd.notna(x) and x not in ['default', 'dummy_value', 'no attribute', '']]
    train_labels_cleaned.append(cleaned_row if cleaned_row else ['no attribute'])

# One-hot encode the cleaned labels
mlb = MultiLabelBinarizer()
y_encoded = mlb.fit_transform(train_labels_cleaned)

print("Cleaned labels example:", train_labels_cleaned[:5])
print("One-hot encoded shape:", y_encoded.shape)  # Should match the number of training samples

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
# Specify the path to the downloaded weights file
local_weights_path = '/kaggle/input/mobilenetv2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128_no_top.h5'
print("MobileNetV2 model loaded successfully")

# Load the MobileNetV2 model with local weights
base_model = MobileNetV2(weights=local_weights_path, include_top=False, input_shape=(128, 128, 3))
pooled_output = GlobalAveragePooling2D()(base_model.output)
feature_extractor_model = Model(inputs=base_model.input, outputs=pooled_output)

feature_extractor_model.summary()  # Should show the layers and output shape

# Initialize the Custom Image Data Generator for training
train_generator = CustomImageDataGenerator(train_images_folder, train_image_ids, batch_size=32)

sample_images = train_generator[0]  # Fetch the first batch
print("Sample images shape:", sample_images.shape)

# Extract features using the generator
X_train = []
for i in range(len(train_generator)):
    batch_images = train_generator[i]
    features = feature_extractor_model.predict(preprocess_input(batch_images))
    X_train.append(features)

X_train = np.vstack(X_train)  # Combine all batches into a single array

print("Extracted features shape:", X_train.shape)  # Should match (num_samples, feature_dim)

# Split the training data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_encoded, test_size=0.2, random_state=42)

print("Training features shape:", X_train_split.shape)
print("Validation features shape:", X_val_split.shape)

import time
# Step 11: Train the Random Forest Classifier
start_time = time.time()  # Start time
print("Model training started")
# Train the Random Forest Classifier using MultiOutputClassifier
clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
clf.fit(X_train_split, y_train_split)

end_time = time.time()  # End time
execution_time = end_time - start_time  # Calculate the time taken
print("Model training complete.")
print(f"Time taken for training: {execution_time:.2f} seconds")

# Evaluate on validation set
y_val_pred = clf.predict(X_val_split)
accuracy = accuracy_score(y_val_split, y_val_pred)
f1 = f1_score(y_val_split, y_val_pred, average='micro')
print("Validation Accuracy:", accuracy)
print("Validation F1 Score (Micro):", f1)

y_val_pred = clf.predict(X_val_split)
accuracy = accuracy_score(y_val_split, y_val_pred)
f1_micro = f1_score(y_val_split, y_val_pred, average='micro')
f1_macro = f1_score(y_val_split, y_val_pred, average='macro')

print("Validation Accuracy:", accuracy)
print("Validation F1 Score (Micro):", f1_micro)
print("Validation F1 Score (Macro):", f1_macro)
print("Time to evaluate model on validation set: {:.2f} seconds".format(end_time - start_time))

# Calculate harmonic mean of F1 scores
attribute_f1_score = (2 * f1_micro * f1_macro) / (f1_micro + f1_macro)

# Calculate final score across categories (assuming total_categories is known)
total_categories = 5  # Adjust this based on your data
final_score = sum([attribute_f1_score] * total_categories) / total_categories
print("Final Average Score Across Categories:", final_score)

# Load test data and process with a new generator
test_image_ids = test_data['id'].tolist()
test_generator = CustomImageDataGenerator(test_images_folder, test_image_ids, batch_size=32)

# Extract features for test images
X_test = []
for i in range(len(test_generator)):
    batch_images = test_generator[i]
    features = feature_extractor_model.predict(preprocess_input(batch_images))
    X_test.append(features)

X_test = np.vstack(X_test)  # Combine all batches into a single array
print("Test features shape:", X_test.shape)
# Predict attributes using the trained classifier
y_test_pred = clf.predict(X_test)

# Inverse transform predictions back to their original labels
test_predictions = mlb.inverse_transform(y_test_pred)

print("Sample test predictions:", test_predictions[:5])

import pandas as pd

category_attributes_path = '/kaggle/input/visual-taxonomy/category_attributes.parquet'
category_attributes_df = pd.read_parquet(category_attributes_path)

# Display the first few rows to understand the structure
print("Category Attributes DataFrame:")
print(category_attributes_df.head())

# Create a dictionary to map each category to its number of required attributes
category_attributes = {
    row['Category']: row['No_of_attribute']
    for _, row in category_attributes_df.iterrows()
}

# Print out the mapping for verification
print("Category to Number of Attributes Mapping:")
print(category_attributes)

# Format predictions for submission
def format_predictions(test_data, predictions, category_attributes):
    formatted_rows = []
    for i, row in test_data.iterrows():
        product_id = row['id']
        category = row['Category']
        num_attributes = category_attributes.get(category, 10)  # Default to 10 attributes if not found
        pred_attrs = list(predictions[i])
        if len(pred_attrs) < num_attributes:
            pred_attrs += ['dummy_value'] * (num_attributes - len(pred_attrs))
        pred_attrs = pred_attrs[:num_attributes]
        formatted_row = [product_id, category, num_attributes] + pred_attrs
        while len(formatted_row) < 13:
            formatted_row.append('dummy_value')
        formatted_rows.append(formatted_row)

    columns = ['id', 'Category', 'len'] + [f'attr_{i+1}' for i in range(10)]
    submission_df = pd.DataFrame(formatted_rows, columns=columns)
    return submission_df

# Apply the formatting function
submission_df = format_predictions(test_data, test_predictions, category_attributes)
print(submission_df.head())

# Specify the path to save the submission file in Kaggle's environment
submission_path = '/kaggle/working/sample_submission.csv'

# Save the submission DataFrame to the specified path
submission_df.to_csv(submission_path, index=False)
print(f"Submission file saved at: {submission_path}")