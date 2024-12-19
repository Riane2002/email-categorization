from django.test import TestCase
from .ml_files.prediction_service import EmailPredictionService

class EmailClassifierTests(TestCase):
    def setUp(self):
        self.service = EmailPredictionService()

    def test_business_email_classification(self):
        business_email = "Meeting tomorrow at 2 PM to discuss Q4 results. Please bring your reports."
        result = self.service.predict_category(business_email)
        self.assertEqual(result, 'business')

    def test_personal_email_classification(self):
        personal_email = "Hey mom, thanks for the birthday wishes! Love you!"
        result = self.service.predict_category(personal_email)
        self.assertEqual(result, 'personal')

    def test_spam_email_classification(self):
        spam_email = "CONGRATULATIONS! You've won $1,000,000! Click here now!!!"
        result = self.service.predict_category(spam_email)
        self.assertEqual(result, 'spam')
