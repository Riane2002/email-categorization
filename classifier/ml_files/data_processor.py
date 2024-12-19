import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

class EmailPreprocessor:
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))
    
    def count_urls(self, text):
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return len(re.findall(url_pattern, text))
    
    def count_exclamation_marks(self, text):
        return text.count('!')
    
    def calculate_caps_ratio(self, text):
        if not text:
            return 0
        caps_chars = sum(1 for c in text if c.isupper())
        return caps_chars / len(text)
    
    def clean_text(self, text):
        # Convert input to DataFrame if it's a single text
        if isinstance(text, str):
            features = {
                'text': [text.lower()],
                'urls': [self.count_urls(text)],
                'exclamations': [self.count_exclamation_marks(text)],
                'caps_ratio': [self.calculate_caps_ratio(text)]
            }
            return pd.DataFrame(features)
        
        # Process batch of texts
        features = {
            'text': [t.lower() for t in text],
            'urls': [self.count_urls(t) for t in text],
            'exclamations': [self.count_exclamation_marks(t) for t in text],
            'caps_ratio': [self.calculate_caps_ratio(t) for t in text]
        }
        return pd.DataFrame(features)
