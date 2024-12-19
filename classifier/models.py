from django.db import models

class EmailPrediction(models.Model):
    email_content = models.TextField()
    predicted_category = models.CharField(max_length=50)
    prediction_date = models.DateTimeField(auto_now_add=True)
    confidence_score = models.FloatField(default=0.0)

    class Meta:
        ordering = ['-prediction_date']
        verbose_name = 'Email Prediction'
        verbose_name_plural = 'Email Predictions'

    def __str__(self):
        return f"{self.predicted_category}: {self.email_content[:50]}..."

