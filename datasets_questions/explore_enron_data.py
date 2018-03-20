#!/usr/bin/python

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

print('There are {} people in the dataset.'.format(len(enron_data)))

num_features = max([len(x) for x in enron_data.values()])

print('There are {} no. of features in the dataset.'.format(num_features))

num_pois = sum([x['poi'] == 1 for x in enron_data.values()])

print('Thre are {} no. of POIs in the dataset.'.format(num_pois))
