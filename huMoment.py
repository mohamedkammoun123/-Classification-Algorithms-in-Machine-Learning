import cv2
import numpy as np
import os
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot  as plt
from sklearn.naive_bayes import GaussianNB as bayes
import observerImages
import classifier

def sign(x):
    if x==0:
        return 0
    elif x>0:
        return 1
    else:
        return -1

def huMoment(observerImages):
    x1=[]
    y1=[]
    for img in os.listdir(observerImages.getPath()):
        image=cv2.imread(observerImages.getPath()+'/'+img,0)
        moments=cv2.moments(image)
        huMoment=cv2.HuMoments(moments)
        classe=observerImages.getClasse(img)
        x1.append(huMoment)
        y1.append(classe)
    return x1,y1

path=r'C:/Users/lenovo/Desktop/dataSet/PNG_data2'
observerImages=observerImages.observerImages(path)
x,y=huMoment(observerImages)
cl=classifier.classifier(x,y)
cl.knn()
cl.bayes()











    
    


    
        
        
        
        
        
    
    



