import argparse
import imutils
import cv2
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
from PIL import Image
from random import randint
def grid(img,cap,gridsize):
    for m in range(0,1):
        img=gauss(img,cap)
        test_image = img
        pixel = img
        windowsize_r = test_image.shape[0]/gridsize
        windowsize_r = int(windowsize_r)
        windowsize_c = test_image.shape[1]/gridsize
        windowsize_c = int(windowsize_c)

        score=0
        for r in range(0,test_image.shape[0] - windowsize_r, windowsize_r):
            for c in range(0,test_image.shape[1] - windowsize_c, windowsize_c):
                window = test_image[r:r+windowsize_r,c:c+windowsize_c]
                w=window.shape[0]
                h=window.shape[1]
                pixel = window

                for i in range(w):
                    for j in range(h):
                        #print(pixel[i, j])
                        randomrange = randint(0,255)
                        if (pixel[i, j] < (128,128)).all(): #128
                            score += 1
                            #print(score)
                        elif (pixel[i, j] == (255, 255)).all():
                            if(score <= 0):
                                score = 0
                            else:
                                score += 0
                            
                if(score > 50):            
                    
                    window = masking(window,cap)
                    window=rotate_image(window,cap)
                    score=0

                img[r:r+windowsize_r,c:c+windowsize_c] = window
        return img

def masking(img,cap):
    mask=np.array( [False]*9)
    inds=np.random.choice(np.arange(9),size=9)
    mask[inds]=True
    mask = np.array(mask)
    mask = mask.reshape(3,3)
    """ print(mask.shape)
    print(mask) """
    kernel = [[1,1,1],[1,1,1],[1,1,1]]
    kernel = np.array(kernel)
    """ print(kernel.shape) """
    kernels = np.ones((1,1),np.uint8)
    randomprocess = randint(0,100)##0 100  <=70 80 90
    if(randomprocess <= 80):
        img = cv2.dilate(img,kernel,iterations = 1) #random
    
    randomprocess = randint(0,100)#
    if(randomprocess <= 20):
        img = cv2.erode(img,mask.astype('uint8'),iterations = 1)
    
    return img
def rotate_image(img,cap):
    randomprocess = randint(0,100)#
    if(randomprocess <= cap):
        num_rows, num_cols = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 5, 1)
        img = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows),borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(255,255,255))
    return img
def gauss(img,cap):
    randomprocess = randint(0,100)#
    if(randomprocess <= cap):
        img = cv2.GaussianBlur(img,(5,5),0)
    return img	