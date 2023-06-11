# Enrollment phase: Cancelable template generation from faces and fingerprints
# using Transfer learning (pretrained CNN as Feature Extractor)  +  Random Convolution

import numpy
import tensorflow
from tensorflow import keras as ks
import numpy as np

# definition of Random Convolution

RandomKernel = [1, 2, 1]

def convolution(A, K, index):
    sum = 0
    if ((index-1) >= 0):
        sum = sum + K[0]*A[index-1]
    sum =  K[1]*A[index]
    if ((index+1) <len(A)):
        sum = sum + K[2]*A[index+1]
    return sum

def random_convolution(V, Kernel):
    R = np.zeros(len(V))
    for i in range(len(V)):
        R[i] = convolution(V, Kernel, i)
    return R

# print(random_convolution(X,RandomKernel))

# Data Preparation

ListofLabels = ['Akshay Kumar', 'Alexandra Daddario', 'Alia Bhatt', 'Amitabh Bachchan', 'Andy Samberg',
                'Anushka Sharma', 'Billie Eilish', 'Brad Pitt', 'Camila Cabello', 'Charlize Theron', 'Claire Holt',
                'Courtney Cox', 'Dwayne Johnson', 'Elizabeth Olsen', 'Ellen Degeneres', 'Henry Cavill', 'Hrithik Roshan',
                'Hugh Jackman', 'Jessica Alba', 'Kashyap', 'Lisa Kudrow', 'Margot Robbie', 'Marmik', 'Natalie Portman',
                'Priyanka Chopra', 'Robert Downey Jr', 'Roger Federer', 'Tom Cruise', 'Vijay Deverakonda',
                'Virat Kohli', 'Zac Efron']

filename = "H:/Data/Biometrics/Faces/SelectedFaces.npy"
selectedfaces = np.load(filename)

filename = "H:/Data/Biometrics/Fingerprints/SelectedFingerprints.npy"
selectedfingerprints = np.load(filename)

filename = "H:/Data/Biometrics/IDs.npy"
IDs = np.load(filename)

# load the model
# Resnet50 without dense layers ... including GlobalAveragePooling2D() layer -> 2048 features
resnet50_base = ks.applications.resnet50.ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
avg = ks.layers.GlobalAveragePooling2D()(resnet50_base.output)
resnet50_modelfs = ks.Model(inputs=resnet50_base.input, outputs=avg)
resnet50_modelfs.summary()

print(" \n Start processing data ... \n ")

cancelabletemplates = np.array([])

for i in range(len(selectedfaces)):
    print(i)
    facefilename = "H:/Data/Biometrics/Faces/Faces/" + selectedfaces[i]
    imface = ks.preprocessing.image.load_img(facefilename)
    faceimage = ks.preprocessing.image.img_to_array(imface)
    fingerprintsfilename = "H:/Data/Biometrics/Fingerprints/RealandAltered/" + selectedfingerprints[i]
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
    X_final = random_convolution(X_new,RandomKernel)
    print("\n Random Convolution of deep features ...")
    print(X_final.shape)
    print(X_final)
    print("\n cancelable template ...")
    cancelabletemplate = X_final.copy()
    print(cancelabletemplate.shape)
    print(cancelabletemplate)
    print(IDs[i])
    if i == 0:
        cancelabletemplates = np.array([cancelabletemplate])
    else:
        cancelabletemplates = np.insert(cancelabletemplates, i, cancelabletemplate, axis=0)



# save [cancelabletemplate + id] in securedDB

print("\n Saving Database ... \n")
filename = "H:/Data/Biometrics/DB.npy"
np.save(filename, cancelabletemplates)


