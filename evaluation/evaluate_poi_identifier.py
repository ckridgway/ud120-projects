#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson14_keys.pkl')
labels, features = targetFeatureSplit(data)

### it's all yours from here forward!
from time import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Create and train the Decision Tree classifier
classifier = DecisionTreeClassifier()

t0 = time()
classifier.fit(X_train, y_train)
print("training time:", round(time()-t0, 3), "s")

t1 = time()
predictions = classifier.predict(X_test)

print("prediction time:", round(time()-t1, 3), "s")
print("Accuracy Score=", accuracy_score(y_test, predictions))
print("Number of POIs predicted: ", int(sum(predictions)))
print("Number of people in test set: ", len(X_test))
print("Accuracy if all predictions are not POI: ", accuracy_score(y_test, [0.0] * len(y_test)))

tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

print("Number of true positives: ", tp)


rscore = recall_score(y_test, predictions)
pscore = precision_score(y_test, predictions)
print("Recall score: ", rscore)
print("Precision score: ", pscore)


fake_predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 
fake_truth = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

tn, fp, fn, tp = confusion_matrix(fake_truth, fake_predictions).ravel()

print("Fake True Positives: ", tp)
print("Fake True Negatives: ", tn)
print("Fake False Positives: ", fp)
print("Fake False Negatives: ", fn)
print("Recall score: ", recall_score(fake_truth, fake_predictions))
print("Precision score: ", precision_score(fake_truth, fake_predictions))