# identification phase:Cancelable template using Transfer learning
# (pretrained CNN as Feature Extractor) and  Random Projection

# Author: Ali Batouche

import numpy
import tensorflow
import time
from tensorflow import keras as ks
import numpy as np
import matplotlib.pyplot as plt
from sklearn import random_projection
from pandas import read_csv
from numpy.linalg import norm
#Import Random Number Generation
from random import seed
from random import randint

# Random projection

  # set the seed
np.random.seed(42)
num_features = 4096

  # Random matrix with elements drawn from a Gaussian distribution
original_dim = 4096
reduced_dim = num_features
R = np.random.randn(reduced_dim, original_dim)

def random_projection(x, Rm):
    # Project the vector x to a lower-dimensional space
    return np.dot(Rm, x)


# set the seed of random generator
seed(int(time.time()))

# Load secured DB and IDs

ListofLabels = ['Akshay Kumar', 'Alexandra Daddario', 'Alia Bhatt', 'Amitabh Bachchan', 'Andy Samberg',
                'Anushka Sharma', 'Billie Eilish', 'Brad Pitt', 'Camila Cabello', 'Charlize Theron', 'Claire Holt',
                'Courtney Cox', 'Dwayne Johnson', 'Elizabeth Olsen', 'Ellen Degeneres', 'Henry Cavill', 'Hrithik Roshan',
                'Hugh Jackman', 'Jessica Alba', 'Kashyap', 'Lisa Kudrow', 'Margot Robbie', 'Marmik', 'Natalie Portman',
                'Priyanka Chopra', 'Robert Downey Jr', 'Roger Federer', 'Tom Cruise', 'Vijay Deverakonda',
                'Virat Kohli', 'Zac Efron']

filename = "H:/Data/Biometrics/DB.npy"
DB = np.load(filename)

filename = "H:/Data/Biometrics/IDs.npy"
IDs = np.load(filename)

# Load datasets

dataset1 = read_csv("H:/Data/Biometrics/Faces/Datasetfaces.csv")
Faces = dataset1.iloc[:, 0].values
LabelsFaces = dataset1.iloc[:, 1].values

dataset2 = read_csv("H:/Data/Biometrics/Fingerprints/Datasetfingerprints.csv")
Fingerprints = dataset2.iloc[:, 0].values
LabelsFingerprints = dataset2.iloc[:, 1].values


# load the model
# Resnet50 without dense layers ... including GlobalAveragePooling2D() layer -> 2048 features
resnet50_base = ks.applications.resnet50.ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
avg = ks.layers.GlobalAveragePooling2D()(resnet50_base.output)
resnet50_modelfs = ks.Model(inputs=resnet50_base.input, outputs=avg)
resnet50_modelfs.summary()


print(" \n Start processing data ... \n ")

# select randomly a person

value = randint(0, len(ListofLabels)-1)
Person = ListofLabels[value]

print(Person)

# select randomly one case (face + fingerprints) from datasets

faces = (Faces[LabelsFaces == Person])[:len(Faces)]
value = randint(0,len(faces)-1)
SelectedFace = faces[value]

print(SelectedFace)

fingerprints = (Fingerprints[LabelsFingerprints == Person])[:len(Fingerprints)]
value = randint(0,len(fingerprints)-1)
SelectedFingerprints = fingerprints[value]

print(SelectedFingerprints)

# generate cancelable template

facefilename = "H:/Data/Biometrics/Faces/Faces/" + SelectedFace
imface = ks.preprocessing.image.load_img(facefilename)
faceimage = ks.preprocessing.image.img_to_array(imface)
fingerprintsfilename = "H:/Data/Biometrics/Fingerprints/RealandAltered/" + SelectedFingerprints
imfingerprint = ks.preprocessing.image.load_img(fingerprintsfilename)
fingerprint = ks.preprocessing.image.img_to_array(imfingerprint)
faceimage = tensorflow.image.resize(faceimage, [224, 224])
fingerprint = tensorflow.image.resize(fingerprint, [224, 224])
images_resized = np.array([faceimage, fingerprint])
# Feature Extraction using pretrained CNN - ResNet50
inputs = ks.applications.resnet50.preprocess_input(images_resized)
Y_proba = resnet50_modelfs.predict(inputs)
deepfeatures = Y_proba
print("\n deep features")
print(deepfeatures.shape)
print(deepfeatures)
# random projection of deepfeatures
X = deepfeatures.copy()
X_new = numpy.append(X[0], X[1], axis=0)
print(X_new)
X_final = random_projection(X_new,R)
print("\n Random Projection of deep features ...")
print(X_final.shape)
print(X_final)
print("\n cancelable template ...")
cancelabletemplate = X_final.copy()
print("cancelable template + Person ...")
print(cancelabletemplate.shape)
print(cancelabletemplate)
print(Person)
print("\n")

# matching ... using Euclidean Distance between Cancelable template and DB ...

print("matching process")

index = 0
mindist = norm(DB[0]-cancelabletemplate)

for i in range(len(DB)):
    dist = norm(DB[i]-cancelabletemplate)
    print(DB[i])
    print(dist)
    if (dist < mindist):
        mindist = dist
        index = i

print(IDs[index])
print(mindist)