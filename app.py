import pandas as pd
import numpy as np
import joblib
import re
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

# Load the saved preprocessing pipeline and model
preprocessing_pipeline = joblib.load('preprocessing_pipeline.joblib')
pipeline = joblib.load('sarcasm_detection_model.joblib')

# Load and preprocess datasets
train_df = pd.read_csv('reddit_training.csv')
test_df = pd.read_csv('reddit_test.csv')
train_df['body'] = train_df['body'].apply(preprocess_text)
test_df['body'] = test_df['body'].apply(preprocess_text)

# Extract features and labels
X = train_df['body']
y = train_df['sarcasm_tag'].apply(lambda x: 1 if x == 'yes' else 0)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Transform the text data into numeric features
X_train_transformed = preprocessing_pipeline.fit_transform(X_train)
X_val_transformed = preprocessing_pipeline.transform(X_val)

# Apply SMOTE to balance the training set
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_transformed, y_train)

# Train the classifier on resampled data
pipeline.fit(X_resampled, y_resampled)

# Predictions for validation data
y_val_pred = pipeline.predict(X_val_transformed)

# Evaluate the model on validation data
val_accuracy = accuracy_score(y_val, y_val_pred)
val_classification_report = classification_report(y_val, y_val_pred, output_dict=True)
val_classification_report_text = classification_report(y_val, y_val_pred)

# Transform the text data into numeric features and predict for test data
X_test_transformed = preprocessing_pipeline.transform(test_df['body'])
y_test_pred = pipeline.predict(X_test_transformed)

# Output predictions for test data
test_df['sarcasm_prediction'] = y_test_pred

# Streamlit app
st.title("Sarcasm Detection Results")

st.write("## Validation Results")
st.write(f"Validation Accuracy: {val_accuracy:.2f}")
st.text("Validation Classification Report:\n")
st.text(val_classification_report_text)

# Display the confusion matrix
st.write("## Confusion Matrix")
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_val, y_val_pred, ax=ax)
st.pyplot(fig)

# Display classification report as a heatmap
st.write("## Classification Report Heatmap")
fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(pd.DataFrame(val_classification_report).iloc[:-1, :].T, annot=True, cmap='Blues', ax=ax)
st.pyplot(fig)

st.write("## Test Data with Sarcasm Predictions")
st.write(test_df[['body', 'sarcasm_prediction']])

if st.button('Refresh Data'):
    st.experimental_rerun()
