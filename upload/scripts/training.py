import argparse
import imutils
from cv2 import cv2
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display
from matplotlib import style
import pandas as pd
import numpy as np
from PIL import Image
from skimage.color import rgb2grey
from sklearn.metrics import accuracy_score
import matplotlib.cm as cm
import math
from scipy import ndimage
import functools
from os.path import basename
from imutils import paths
import pickle
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier
from sklearn.utils.estimator_checks import check_estimator
from .fakesign import grid
from .fakesign import masking
from .fakesign import rotate_image
from .fakesign import gauss
def image_to_feature_vector(image, size=(32, 32)):
	return cv2.resize(image,size).flatten()

def image_to_feature_vector2(image):
	pixels = image.flatten()
	return pixels
	
def trainModel(path):
    """ ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
    args = vars(ap.parse_args()) """
    imagePaths = list(paths.list_images(path))
    #imagePathss = list(paths.list_images('testfeature'))
    features = []
    labels = []
    #testfeatures = []
    """ for (i, imagePath) in enumerate(imagePaths):
    	image = cv2.imread(imagePath,0)
    	grid(path,image) """
    
    print('Fake signatures generating...')
    for (i, imagePath) in enumerate(imagePaths):
    	# load the image and extract the class label (assuming that our
    	# path as the format: /path/to/dataset/{class}.{image_num}.jpg
    	image = cv2.imread(imagePath,0)
    	fimage = grid(image,90,32)
    	""" outfile = '%sall%s.jpg' % ('Preprocess\\', str(i))
    	image = all(imagePath,outfile) """
    	label = imagePath.split(os.path.sep)[-1].split(".")[0]

    	pixels = image_to_feature_vector(image)
    	fpixels = image_to_feature_vector(fimage)
    	features.append(pixels)
    	labels.append(label)
    	features.append(fpixels)
    	labels.append('fake')
    	# show an update every 1,000 images
    	# if i > 0 and i % 1000 == 0:
    		# print("[INFO] processed {}/{}".format(i, len(imagePaths)))
    """ for (i, imagePath) in enumerate(imagePathss):
    	# load the image and extract the class label (assuming that our
    	# path as the format: /path/to/dataset/{class}.{image_num}.jpg
    	#image = cv2.imread(imagePath)
    	outfile = '%sall%s.jpg' % ('testpre\\', str(i))
    	image = all(imagePath,outfile)
    	pixels = image_to_feature_vector(image)
    	testfeatures.append(pixels) """
    # show some information on the memory consumed by the raw images
    # matrix and features matrix
    features = np.array(features)
    labels = np.array(labels)
    estimators = []
    # print(features.shape)
    """ from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=1)
    from sklearn.metrics import classification_report """
    # print(X_train)
    print(features.shape)
    # SVM
    print('SVM Training...')
    from sklearn.svm import SVC
    svclassifier = SVC(kernel='linear')
    params_svm = {"kernel":"rbf", "C":10, "gamma":0.000001}
    svclassifier.set_params(**params_svm)
    estimators.append(('svm', svclassifier))
    svclassifier.fit(features, labels)
    filename = 'SVM_finalized_model.sav'
    pickle.dump(svclassifier, open(path + filename, 'wb'))
    #SVM_predict_result = svclassifier.predict(testfeatures)
    """ score = svclassifier.score
    y_pred = svclassifier.predict(X_test) """
    # print(classification_report(y_test,y_pred))
    # print ("svm predict")
    # print(SVM_predict_result)
    # print(score)

    # MLP
    print('MLP Training...')
    from sklearn.neural_network import MLPClassifier
    from sklearn import preprocessing
    mlp = MLPClassifier(hidden_layer_sizes=(88,48,28,8), activation='relu', solver='lbfgs', max_iter=50 ,random_state=42) #48 72 80 r 70
    estimators.append(('mlp', mlp))
    
    mlp.fit(features,labels)
    filename = 'MLP_finalized_model.sav'
    pickle.dump(mlp, open(path + filename, 'wb'))
    #mlp_predict_result = mlp.predict(testfeatures)
    """ y_pred = mlp.predict(X_test) """
    # print(classification_report(y_test,y_pred,zero_division=1))
    # print ("mlp predict")
    # print(mlp_predict_result)

    print('KNN Training...')
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=5)
    estimators.append(('knn', neigh))
    neigh.fit(features, labels)
    filename = 'KNN_finalized_model.sav'
    pickle.dump(neigh, open(path + filename, 'wb'))
    #knn_predict_result = neigh.predict(testfeatures)
    # y_pred = neigh.predict(X_test)
    
    # print(classification_report(y_test,y_pred))
    # print ("knn predict")
    # print(knn_predict_result)

    # print('Ensemble Creating...')
    # ensemble = VotingClassifier(estimators)
    # ensemble.fit(X_scaled, y_train)
    # X_scaled = preprocessing.scale(X_train)
    # filename = 'voted_finalized_model.sav'
    # pickle.dump(ensemble, open(path + filename, 'wb'))
    
    #result = ensemble.predict(testfeatures)
    # y_pred = ensemble.predict(X_test)
    # print(classification_report(y_test,y_pred))
    # print("Voted")
    # print(result) 
