# identification phase using 1D-CNN

import time

from sklearn.model_selection import train_test_split
from tensorflow import keras as ks
import numpy as np
from random import seed



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

filename = "H:/Data/Biometrics/IDsNums.npy"
IDsNums = np.load(filename)

# Split into training (80%) and testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(DB, IDsNums, test_size = 0.2, random_state = 123)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

input_data_shape = (4096,1)
hidden_activation_function = 'relu'
output_activation_function = 'softmax'

model = ks.Sequential()
model.add(ks.layers.Conv1D(32, kernel_size=3, padding='same', activation=hidden_activation_function, input_shape=input_data_shape))
model.add(ks.layers.MaxPooling1D(pool_size=4))
model.add(ks.layers.BatchNormalization())
model.add(ks.layers.Conv1D(32, kernel_size=3, padding='same',activation=hidden_activation_function))
model.add(ks.layers.MaxPooling1D(pool_size=4))
model.add(ks.layers.BatchNormalization())
model.add(ks.layers.Conv1D(16, kernel_size=3, padding='same', activation=hidden_activation_function))
model.add(ks.layers.MaxPooling1D(pool_size=4))
model.add(ks.layers.Conv1D(16, kernel_size=3, padding='same', activation=hidden_activation_function))
model.add(ks.layers.MaxPooling1D(pool_size=4))
model.add(ks.layers.BatchNormalization())
model.add(ks.layers.Dropout(0.2))
model.add(ks.layers.Flatten())
model.add(ks.layers.Dense(300, activation=hidden_activation_function))
model.add(ks.layers.Dense(128, activation=hidden_activation_function))
model.add(ks.layers.Dropout(0.2))
model.add(ks.layers.Dense(31,activation=output_activation_function))
model.summary()

# parameter settings
optimizer = 'adam'
loss_function = 'sparse_categorical_crossentropy'
metric = ['accuracy']
model.compile(optimizer=optimizer, loss=loss_function, metrics=metric)

#
model.fit(X_train, y_train, epochs=50)
#
# #Training Evaluation
training_loss, training_accuracy = model.evaluate(X_train, y_train)
print('Training Data Accuracy {}'.format(round(float(training_accuracy),2)))
#
# #Testing Evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print('Test Data Accuracy {}'.format(round(float(test_accuracy),2)))

