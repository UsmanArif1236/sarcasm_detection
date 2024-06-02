import pandas as pd
import numpy as np
import re
import nltk
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

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

# Define feature extractors
class LexicalFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vectorizer = CountVectorizer(ngram_range=(1, 2))

    def fit(self, X, y=None):
        self.vectorizer.fit(X)
        return self

    def transform(self, X):
        return self.vectorizer.transform(X)

class PragmaticFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = np.array([
            [len(re.findall(r'[A-Z]', text)),
             len(re.findall(r'[!?.]', text)),
             len(re.findall(r'[:;=][oO\-]?[D\)\]\(\/\\OpP]', text)),
             len(re.findall(r'\b(lol|haha|hehe)\b', text))]
            for text in X])
        return features

class ExplicitIncongruityFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.sid = SentimentIntensityAnalyzer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def explicit_incongruity(text):
            tokens = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(tokens)
            polarities = [self.sid.polarity_scores(word)['compound'] for word, tag in pos_tags if tag.startswith('JJ') or tag.startswith('VB')]
            pos_count = sum(1 for p in polarities if p > 0.5)
            neg_count = sum(1 for p in polarities if p < -0.5)
            return [pos_count, neg_count, len(polarities)]

        return np.array([explicit_incongruity(text) for text in X])

class ImplicitIncongruityFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.sid = SentimentIntensityAnalyzer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def implicit_incongruity(text):
            doc = nlp(text)
            phrases = [chunk.text for chunk in doc.noun_chunks]
            sentence_sentiment = self.sid.polarity_scores(text)['compound']
            incongruity_score = 0

            for phrase in phrases:
                phrase_sentiment = self.sid.polarity_scores(phrase)['compound']
                if (sentence_sentiment > 0 and phrase_sentiment < -0.5) or (sentence_sentiment < 0 and phrase_sentiment > 0.5):
                    incongruity_score += 1

            return [incongruity_score]

        return np.array([implicit_incongruity(text) for text in X])

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

# Predictions for validation data
y_val_pred = pipeline.predict(X_val_transformed)

# Evaluate the model on validation data
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Validation Classification Report:\n", classification_report(y_val, y_val_pred))

# # Cross-Validation
# scores = cross_val_score(Pipeline([
#     ('features', features),
#     ('classifier', SVC(kernel='rbf', class_weight=class_weights_dict))
# ]), X, y, cv=5, scoring='f1')
# print("Cross-Validation F1 Scores:", scores)
# print("Average F1 Score:", np.mean(scores))

# Predictions for test data
X_test_transformed = preprocessing_pipeline.transform(test_df['body'])
y_test_pred = pipeline.predict(X_test_transformed)

# Output predictions
test_df['sarcasm_prediction'] = y_test_pred
print(test_df[['body', 'sarcasm_prediction']])
test_df.to_csv('sarcasm_predictions.csv', index=False)