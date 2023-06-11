import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def test_duplicates(dataframe):
    duplicates = dataframe[dataframe.duplicated()]

    if duplicates.empty:
        print("There are no duplicates in the DataFrame.")
    else:
        print("The following rows are duplicates:")
        print(duplicates)
        print('duplicates length', duplicates.__len__())


# read the Data DB


X = np.load('DB.npy')

Names = np.load('IDs.npy')
Ids = np.load('IDsNums.npy')

# Creating a dictionary association for IDs -> Names
unique_elements_ids, counts = np.unique(Ids, return_counts=True)

print(np.asarray((unique_elements_ids, counts)).T)

unique_elements_names, counts = np.unique(Names, return_counts=True)

print(np.asarray((unique_elements_names, counts)).T)

Dict = dict(zip(unique_elements_ids, unique_elements_names))

# with open('Dictionary.json', 'w') as file:
#     json.dump(Dict, file)

# for each class randomly 2/3 for train 1/3 for test for each class

## columns
columns = []

for i in range(len(X[0])):
    columns.append(f'x_{i}')
columns.append('label')

## Making The Dataframe
data = np.concatenate((X, np.atleast_2d(Ids).T), axis=1)
data = np.unique(data, axis=0)
df = pd.DataFrame(data, columns=columns)

######
## Splitting dataframes to test/train 2/3 per class
train_DATA = []
test_DATA = []
prblm_vect = []

for name, group in df.groupby('label'):

    print(name)
    # from the pack of 30 pick randomly 10 for df_test and 20 for df_train

    x = group.iloc[:, :].values
    # random indexes unique for testing
    test_vector = np.random.choice(range(0, len(group)), size=10, replace=False)
    # rest for training
    train_vector = [x for x in range(0, len(group)) if x not in test_vector]

    #
    print('train vect length: ', train_vector.__len__())
    # print(train_vector)
    print('test vect length: ', test_vector.__len__())
    # print(test_vector)

    # Check if there are any common items
    # common_elements = np.intersect1d(train_vector, test_vector)
    # prblm_vect.append(False if common_elements.size >0 else True)

    # appending to dataframes
    for i in train_vector:
        train_DATA.append(x[i], )
    for i in test_vector:
        test_DATA.append(x[i], )

df_train = pd.DataFrame(train_DATA, columns=columns)
df_test = pd.DataFrame(test_DATA, columns=columns)

# checking for duplicates after split

counts = df_train.groupby('label').size()
print('df_train Counts: ', counts)
test_duplicates(df_train)

##saving counts dataframe
counts.to_csv('counts_train.csv')

## plotting counts
plt.bar(counts.index, counts.values)
plt.show()
plt.savefig('counts_train.png')


counts = df_test.groupby('label').size()
print('df_test Counts: ', counts)
test_duplicates(df_test)

##saving counts dataframe
counts.to_csv('counts_test.csv')

## plotting counts
plt.bar(counts.index, counts.values)
plt.show()
plt.savefig('counts_train.png')
# plotting Counts

plt.bar(counts.keys(), counts.values())
plt.show()
plt.savefig('counts_train.png')


# saving Dataframes

df_train.to_csv('TRAIN_DATA.csv',index_label=False)
df_test.to_csv('TEST_DATA.csv',index_label=False)