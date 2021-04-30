from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pickle
import argparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier
from sklearn.utils.estimator_checks import check_estimator
from imutils import paths
import math
from scipy import ndimage
import functools
from cv2 import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
# import json

def image_to_feature_vector(image, size=(32, 32)):
	return cv2.resize(image, size).flatten()

def testSignature(path, image):
    """ ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--Predictfeature", required=True,
    	help="path to Predict feature")
    args = vars(ap.parse_args()) """
    #testPaths = list(paths.list_images(path + image))
    testfeatures = []
    estimators = []
    image = cv2.imread(path + image,0)
    pixels = image_to_feature_vector(image)
    testfeatures.append(pixels)
    testfeatures = np.array(testfeatures)
    """ for (i, imagePath) in enumerate(testPaths):
    	# load the image and extract the class label (assuming that our
    	# path as the format: /path/to/dataset/{class}.{image_num}.jpg
    	image = cv2.imread(imagePath)
    	#outfile = '%sall%s.jpg' % ('Preprocess\\', str(i))
    	#image = all(imagePath,outfile)
    	pixels = image_to_feature_vector(image)
    	testfeatures.append(pixels)

    testfeatures = np.array(testfeatures) """

    filename = 'SVM_finalized_model.sav'
    loaded_model = pickle.load(open(path + filename, 'rb'))
    print(testfeatures.shape)
    SVM_result = loaded_model.predict(testfeatures)
    estimators.append(('svm', loaded_model))
    print ("SVM_predict")
    print(SVM_result)
    SVM_prob = loaded_model.predict_proba(testfeatures)
    """ filename = 'MLP_finalized_model.sav'
    loaded_model = pickle.load(open(path + filename, 'rb'))
    MLP_result = loaded_model.predict(testfeatures)
    estimators.append(('mlp', loaded_model))
    print ("mlp predict")
    print(MLP_result)

    filename = 'KNN_finalized_model.sav'
    loaded_model = pickle.load(open(path + filename, 'rb'))
    KNN_result = loaded_model.predict(testfeatures)
    estimators.append(('knn', loaded_model))
    print ("knn predict")
    print(KNN_result) """

    """ ensemble = VotingClassifier(estimators,voting='soft') #fit_base_estimators=False
    #ensemble.fit
    ensemble.estimators_ = estimators
    ensemble.le_ = LabelEncoder().fit(testfeatures)
    ensemble.classes_ = seclf.le_.classes_
    filename = 'voted_finalized_model.sav'
    pickle.dump(ensemble, open(filename, 'wb'))
    result = ensemble.predict(testfeatures)
    print("Voted")
    print(result)  """

    # ensemble = VotingClassifier(estimators)
    # filename = 'voted_finalized_model.sav'
    # loaded_model = pickle.load(open(path + filename, 'rb'))
    # vote_result = loaded_model.predict(testfeatures)
    # print("Voted")
    # print(vote_result) 
    perse = [SVM_result.tolist()]

    my_details = {
        'vote': perse[0][0] != 'fake',
        'fake_ratio': SVM_prob[0][1]
    }
    """ with open('personal.json', 'w') as json_file:
        json.dump(my_details, json_file) """
    return my_details

    """ with open('personal.json', 'wb') as json_file:
    	json.dumps(my_details) """
    """ pickle.dump(ensemble, open(filename, 'wb'))
    result = ensemble.predict(testfeatures)
    print("Voted")
    print(result)  """

    """json 2 paramiter  """