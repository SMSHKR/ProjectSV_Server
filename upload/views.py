from django.shortcuts import render
from django.http import HttpResponse, HttpResponseNotFound
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings

import uuid
import os

from .train.imageprocessing import imagePreprocess
from .train.training import trainModel

# Create your views here.
def test(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        return HttpResponse(fs.url(filename))
    return HttpResponse("Test Page")

def train(request):
    if request.method == 'POST':
        model = uuid.uuid4()
        files = request.FILES.getlist('images')
        fs = FileSystemStorage()
        path = settings.MEDIA_ROOT + '/' + str(model) + '/'
        for file in files:
            fs.save(str(model) + '/' + file.name, file)
            imagePreprocess(path + file.name)
        trainModel(path)
        response = { "model": model }
        return JsonResponse(response, safe=False)
    return HttpResponseNotFound()
