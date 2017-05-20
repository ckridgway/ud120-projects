#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Use a smaller portion of the training set to speed up
#features_train = features_train[:int(len(features_train)/100)]
#labels_train = labels_train[:int(len(labels_train)/100)]

# Create an train the SVM classifier
classifier = SVC(kernel='rbf', C=10000.0)

t0 = time()
classifier.fit(features_train, labels_train)
print("training time:", round(time()-t0, 3), "s")

t1 = time()
predictions = classifier.predict(features_test)
print("prediction time:", round(time()-t1, 3), "s")

print("Accuracy Score=", accuracy_score(labels_test, predictions))

WHO = ['Sara', 'Chris']

print('Item 10=', WHO[predictions[10]])
print('Item 26=', WHO[predictions[26]])
print('Item 50=', WHO[predictions[50]])

import numpy
unqiue, counts = numpy.unique(predictions, return_counts=True)

print('Sara wrote {} emails.'.format(counts[0]))
print('Chris wrote {} email.'.format(counts[1]))

#########################################################
### your code goes here ###

#########################################################
