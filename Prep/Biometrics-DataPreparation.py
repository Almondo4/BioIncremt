# Project: Cancelable Biometrics using Deep Learning and Random Convolution - Data Preparation

import time
import numpy as np
from pandas import read_csv

#Import Random Number Generation
from random import seed
from random import randint


# Data Preparation - Faces

ListofLabels = ['Akshay Kumar', 'Alexandra Daddario', 'Alia Bhatt','Amitabh Bachchan','Andy Samberg','Anushka Sharma',
                'Billie Eilish','Brad Pitt','Camila Cabello','Charlize Theron','Claire Holt','Courtney Cox',
                'Dwayne Johnson', 'Elizabeth Olsen', 'Ellen Degeneres', 'Henry Cavill', 'Hrithik Roshan', 'Hugh Jackman',
                'Jessica Alba','Kashyap','Lisa Kudrow','Margot Robbie','Marmik','Natalie Portman','Priyanka Chopra',
                'Robert Downey Jr','Roger Federer','Tom Cruise','Vijay Deverakonda','Virat Kohli','Zac Efron']

print(ListofLabels)

NumSamples = 30

SelectedFaces = []

dataset = read_csv("H:/Data/Biometrics/Faces/Datasetfaces.csv")
Faces = dataset.iloc[:, 0].values
Labels = dataset.iloc[:, 1].values

# IDs = len(ListofLabels) * NumSamples
IDs = []
IDsNums = []

seed(int(time.time()))

for i in range(len(ListofLabels)):
    faces = (Faces[Labels == ListofLabels[i]])[:len(Faces)]
    limit = len(faces)-1
    for j in range(NumSamples):
        value = randint(0,limit)
        SelectedFaces.append(faces[value])
        IDs.append(ListofLabels[i])
        IDsNums.append(i)

print(SelectedFaces)

filename = 'H:/Data/Biometrics/Faces/selectedfaces.npy'
np.save(filename, SelectedFaces)

# Data preparation - fingerprints

dataset = read_csv("H:/Data/Biometrics/Fingerprints/Datasetfingerprints.csv")
Fingerprints = dataset.iloc[:, 0].values
Labels = dataset.iloc[:, 1].values

# print(len(Fingerprints))

seed(int(time.time()))

SelectedFingerprints = []

for i in range(len(ListofLabels)):
    fingerprints = (Fingerprints[Labels == ListofLabels[i]])[:len(Fingerprints)]
    limit = len(fingerprints) - 1
    for j in range(NumSamples):
        value = randint(0, limit)
        SelectedFingerprints.append(fingerprints[value])

print(SelectedFingerprints)
filename = 'H:/Data/Biometrics/Fingerprints/selectedfingerprints.npy'
np.save(filename, SelectedFingerprints)

print(IDs)
filename = 'H:/Data/Biometrics/IDs.npy'
np.save(filename, IDs)

print(IDsNums)
filename = 'H:/Data/Biometrics/IDsNums.npy'
np.save(filename, IDsNums)

print("selected faces fingerprints and IDs saved ... ")