import pandas as pd
from sklearn.model_selection import train_test_split
from classifier.ml_files.email_classifier import EmailClassifier
from classifier.ml_files.data_processor import EmailPreprocessor

def train_and_save_model():
    # Load the enhanced dataset with absolute path
    data = pd.read_csv('C:/Users/HP/email_categorization/data/enhanced_mail_data.csv')
    
    # Initialize preprocessor
    preprocessor = EmailPreprocessor()
    
    # Process all emails
    processed_data = pd.DataFrame([preprocessor.clean_text(text) for text in data['Message']])
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(processed_data, data['Category'], test_size=0.2, random_state=42)
    
    # Initialize and train classifier
    classifier = EmailClassifier()
    classifier.train(X_train, y_train)
    
    # Save the trained model
    classifier.save_model('classifier/ml_files/email_classifier.joblib')
    
    # Print accuracy on test set
    accuracy = (classifier.predict(X_test) == y_test).mean()
    print(f"Model accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    train_and_save_model()
