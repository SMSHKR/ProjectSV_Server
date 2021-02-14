from django.shortcuts import render
from django.http import HttpResponse, HttpResponseNotFound
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings

import uuid
import os

from .scripts.imageprocessing import imagePreprocess
from .scripts.training import trainModel

# Create your views here.
def test(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        return HttpResponse(fs.url(filename))
    return HttpResponseNotFound()

def train(request):
    if request.method == 'POST':
        model = uuid.uuid4()
        files = request.FILES.getlist('images')
        location = settings.MEDIA_ROOT + '/' + str(model) + '/'
        fs = FileSystemStorage(location=location)
        for file in files:
            filename = fs.save(file.name, file)
            imagePreprocess(location + filename)
        trainModel(location)
        response = { "model": model }
        return JsonResponse(response, safe=False)
    return HttpResponseNotFound()
