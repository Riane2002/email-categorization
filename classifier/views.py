from django.shortcuts import render
from django.http import JsonResponse
from .ml_files.prediction_service import EmailPredictionService
from .models import EmailPrediction

def classify_email(request):
    result = None
    if request.method == 'POST':
        email_content = request.POST.get('email_content', '')
        service = EmailPredictionService()
        result = service.predict_category(email_content)
        
        # Store prediction in database
        EmailPrediction.objects.create(
            email_content=email_content,
            predicted_category=result
        )
    
    return render(request, 'classifier/index.html', {
        'result': result,
        'recent_predictions': EmailPrediction.objects.all()[:5]
    })

def prediction_history(request):
    predictions = EmailPrediction.objects.all().order_by('-prediction_date')
    return render(request, 'classifier/history.html', {'predictions': predictions})

def api_classify_email(request):
    if request.method == 'POST':
        email_content = request.POST.get('email_content', '')
        service = EmailPredictionService()
        result = service.predict_category(email_content)
        return JsonResponse({'category': result})

