from sklearn import svm
import numpy
from numpy import *

def train(x,y):
    classifier = svm.LinearSVC()
    classifier.fit(x,y)
    print classifier
    return classifier

def predict(classifier,x):
    #forecast = classifier.decision_function(x)
    #print numpy.max(forecast)
    #y, = where(forecast[0]==numpy.max(forecast))
    y = classifier.predict(x)
    return y[0]
