from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
import joblib

class EmailClassifier:
    def __init__(self):
        self.text_features = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english'))
        ])
        
        self.preprocessor = ColumnTransformer([
            ('text', self.text_features, 'text'),
            ('numeric', 'passthrough', ['urls', 'exclamations', 'caps_ratio'])
        ])
        
        self.pipeline = Pipeline([
            ('features', self.preprocessor),
            ('classifier', MultinomialNB())
        ])
    
    def train(self, emails, labels):
        # Convert emails to DataFrame if it's not already
        if not isinstance(emails, pd.DataFrame):
            emails = pd.DataFrame(emails)
        self.pipeline.fit(emails, labels)
    
    def predict(self, emails):
        # Convert single email to DataFrame
        if not isinstance(emails, pd.DataFrame):
            emails = pd.DataFrame(emails)
        return self.pipeline.predict(emails)
    
    def save_model(self, filepath):
        joblib.dump(self.pipeline, filepath)
    
    def load_model(self, filepath):
        self.pipeline = joblib.load(filepath)
