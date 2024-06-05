# custom_transformers.py
import re
import numpy as np
import nltk
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

# Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

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

            # Calculate the longest positive/negative subsequence
            longest_pos_seq = longest_neg_seq = 0
            current_pos_seq = current_neg_seq = 0

            for p in polarities:
                if p > 0.5:
                    current_pos_seq += 1
                    current_neg_seq = 0
                elif p < -0.5:
                    current_neg_seq += 1
                    current_pos_seq = 0
                else:
                    current_pos_seq = current_neg_seq = 0
                
                longest_pos_seq = max(longest_pos_seq, current_pos_seq)
                longest_neg_seq = max(longest_neg_seq, current_neg_seq)

            return [pos_count, neg_count, len(polarities), longest_pos_seq, longest_neg_seq]

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