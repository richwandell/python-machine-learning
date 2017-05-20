import csv
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

clf = joblib.load('../trained/filename.pkl')


X, y = [], []
with open("data/train.csv", "r") as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in csvreader:
        try:
            vals = list(map(int, row[1:]))
            X.append(vals)
            y.append([str(row[0])])
        except:
            pass