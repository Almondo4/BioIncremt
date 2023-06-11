import pandas as pd
import torch
from avalanche.benchmarks import ni_benchmark, benchmark_with_validation_stream, nc_benchmark
from torch.utils.data import Dataset


class TorchDataset(Dataset):

    def __init__(self, filePath):
        # Read CSV
        data = pd.read_csv(filePath)

        self.X = data.iloc[:, :-1].values
        self.targets = data.iloc[:, -1].values

        # Feature Scale if you want

        # Convert to Torch Tensors
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.targets = torch.tensor(self.targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        return self.X[item], self.targets[item]


def prep_benchmark(train_loc, test_loc):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    hdata_train = TorchDataset(train_loc)
    hdata_test = TorchDataset(test_loc)

    return benchmark_with_validation_stream(nc_benchmark(train_dataset=hdata_train, test_dataset=hdata_test
                                 , shuffle=True, seed=1234, task_labels=True, n_experiences=11,
                                 per_exp_classes={0: 10}))


def prep_Joint_benchmark(train_loc, test_loc):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    hdata_train = TorchDataset(train_loc)
    hdata_test = TorchDataset(test_loc)

    return benchmark_with_validation_stream(ni_benchmark(
        train_dataset=hdata_train, test_dataset=hdata_test
        , n_experiences=1, shuffle=True, seed=1234,
    ))
