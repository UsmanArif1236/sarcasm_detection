import pandas as pd
import numpy as np
import re
import nltk
import spacy
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Import custom transformers
from custom_transformers import LexicalFeatures, PragmaticFeatures, ExplicitIncongruityFeatures, ImplicitIncongruityFeatures

# Load datasets
train_df = pd.read_csv('reddit_training.csv')
test_df = pd.read_csv('reddit_test.csv')

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

# Apply preprocessing to the 'body' column
train_df['body'] = train_df['body'].apply(preprocess_text)
test_df['body'] = test_df['body'].apply(preprocess_text)

# Combine all feature extractors
features = FeatureUnion([
    ('lexical', LexicalFeatures()),
    ('pragmatic', PragmaticFeatures()),
    ('explicit_incongruity', ExplicitIncongruityFeatures()),
    ('implicit_incongruity', ImplicitIncongruityFeatures())
])

# Extract features and labels
X = train_df['body']
y = train_df['sarcasm_tag'].apply(lambda x: 1 if x == 'yes' else 0)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the feature extraction and SMOTE pipeline
preprocessing_pipeline = Pipeline([
    ('features', features)
])

# Transform the text data into numeric features
X_train_transformed = preprocessing_pipeline.fit_transform(X_train)
X_val_transformed = preprocessing_pipeline.transform(X_val)

# Apply SMOTE to balance the training set
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_transformed, y_train)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Create the final classification pipeline
pipeline = Pipeline([
    ('classifier', SVC(kernel='rbf', class_weight=class_weights_dict))
])

# Train the classifier on resampled data
pipeline.fit(X_resampled, y_resampled)

# Save the preprocessing pipeline and model
joblib.dump(preprocessing_pipeline, 'preprocessing_pipeline.joblib')
joblib.dump(pipeline, 'sarcasm_detection_model.joblib')
