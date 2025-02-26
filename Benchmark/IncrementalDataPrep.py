from torch import nn

import pandas as pd
from avalanche.benchmarks import dataset_benchmark
from torch.utils.data import Dataset
import torch


class TorchDataset2(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.targets = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        return self.X[item], self.targets[item]


def prepare_single_dataset(data, label):
    X = data[data['label'] == label].iloc[:, :-1].values
    targets = data[data['label'] == label].iloc[:, -1].values
    return TorchDataset2(X, targets)


def prep_incremental_benchmark(train_loc, test_loc):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    data_train = pd.read_csv(train_loc)
    data_train['label'] = data_train['label'].astype('int64')
    data_test = pd.read_csv(test_loc)
    data_test['label'] = data_test['label'].astype('int64')

    train_datasets = []
    test_datasets = []
    cumulative_test_datasets = []

    # print the size of both train and test data
    print("Train data size: ", data_train.shape)
    print("Test data size: ", data_test.shape)

    # Prepare the first dataset with labels 0 and 1

    d1 = prepare_single_dataset(data_train, 0)
    d2 = prepare_single_dataset(data_train, 1)

    dt1 = prepare_single_dataset(data_test, 0)
    dt2 = prepare_single_dataset(data_test, 1)

    # Concatenate the tensors in the datasets
    combined_X = torch.cat([d1.X, d2.X])
    combined_targets = torch.cat([d1.targets, d2.targets])
    combined_Xt = torch.cat([dt1.X, dt2.X])
    combined_t_targets = torch.cat([dt1.targets, dt2.targets])

    # Create a new dataset instance with the concatenated tensors
    d1_2 = TorchDataset2(combined_X, combined_targets)
    dt1_2 = TorchDataset2(combined_Xt, combined_t_targets)

    train_datasets.append(d1_2)
    test_datasets.append(dt1_2)

    # Prepare the remaining datasets with labels 2 to 30
    for label in range(2, 31):
        train_datasets.append(prepare_single_dataset(data_train, label))
        test_datasets.append(prepare_single_dataset(data_test, label))

    # create a cumulative test dataset for each 5th increment
    cumulative_datasets = []
    for increment in range(1, 7):
        cumulative_X = torch.cat([d.X for d in test_datasets[:increment * 5]])
        cumulative_targets = torch.cat([d.targets for d in test_datasets[:increment * 5]])
        cumulative_datasets.append(TorchDataset2(cumulative_X, cumulative_targets))

    for dataset in cumulative_test_datasets:
        print(dataset.classes_in_this_dataset)
        # Todo maybe transofrm every sub cumulative dataset into a benchmark?
    # return all the datasets within a set
    return train_datasets, test_datasets, dataset_benchmark(train_datasets=train_datasets,
                                                            test_datasets=test_datasets), cumulative_datasets


train_datasets, test_datasets, benchmark, cumulative_benchmark = prep_incremental_benchmark \
    (train_loc='../DATA/TRAIN_DATA.csv', test_loc='../DATA/TEST_DATA.csv')

# Print classes available in each set of cumulative datasets
for i, cumul_dataset in enumerate(cumulative_benchmark):
    unique_classes = torch.unique(cumul_dataset.targets)
    print(f"Classes in cumulative dataset {i + 1}: {unique_classes.tolist()}")

######

# Test model too and see how you can seperrat it?

# main function to run this file
if __name__ == '__main__':
    train_datasets, test_datasets, benchmark, cumulative_benchmark = prep_incremental_benchmark \
        (train_loc='../DATA/TRAIN_DATA.csv', test_loc='../DATA/TEST_DATA.csv')
    data = pd.read_csv('../DATA/TRAIN_DATA.csv')
    #print list of labels in data
    print('Labels in data: ', data['label'].unique())
    print('n Labels in data: ', len(data['label'].unique()))