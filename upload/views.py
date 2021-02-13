from django.shortcuts import render
from django.http import HttpResponse, HttpResponseNotFound
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage

import uuid

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
        files = request.FILES.getlist('images')
        fs = FileSystemStorage()
        for file in files:
            fs.save(file.name, file)
        response = {
            "test": 1,
            "model": uuid.uuid4()
        }
        return JsonResponse(response, safe=False)
    return HttpResponseNotFound()
