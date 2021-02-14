from django.shortcuts import render
from django.http import HttpResponseNotFound
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings

import uuid
import os

from .scripts.imageprocessing import imagePreprocess
from .scripts.training import trainModel

# Create your views here.
def test(request):
    if request.method == 'POST':
        model = request.POST.get('model')
        image = request.FILES['image']
        location = settings.MEDIA_ROOT + '/' + str(model) + '/'
        fs = FileSystemStorage(location=location)
        filename = fs.save(image.name, image)
        response = {
            'model': model,
            'file': filename
        }
        return JsonResponse(response, safe=False)
    return HttpResponseNotFound()

def train(request):
    if request.method == 'POST':
        model = uuid.uuid4()
        files = request.FILES.getlist('images')
        location = settings.MEDIA_ROOT + '/' + str(model) + '/'
        fs = FileSystemStorage(location=location)
        filenames = list()
        for file in files:
            filename = fs.save(file.name, file)
            imagePreprocess(location + filename)
            filenames.append(filename)
        trainModel(location)
        for file in filenames:
            fs.delete(file)
        response = { "model": model }
        return JsonResponse(response, safe=False)
    return HttpResponseNotFound()
