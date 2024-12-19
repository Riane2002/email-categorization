from .email_classifier import EmailClassifier
from .data_processor import EmailPreprocessor

class EmailPredictionService:
    def __init__(self, model_path='classifier/ml_files/email_classifier.joblib'):
        self.classifier = EmailClassifier()
        self.classifier.load_model(model_path)
        self.preprocessor = EmailPreprocessor()
    
    def predict_category(self, email_content):
        # Clean and extract features from the input email
        processed_email = self.preprocessor.clean_text(email_content)
        # Make prediction using the DataFrame directly
        prediction = self.classifier.predict(processed_email)[0]
        return prediction
