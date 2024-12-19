from django.contrib import admin
from .models import EmailPrediction

@admin.register(EmailPrediction)
class EmailPredictionAdmin(admin.ModelAdmin):
    list_display = ('email_content', 'predicted_category', 'prediction_date')
    list_filter = ('predicted_category', 'prediction_date')
    search_fields = ('email_content', 'predicted_category')
    ordering = ('-prediction_date',)
