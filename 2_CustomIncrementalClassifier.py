# %%

from torch import nn

import pandas as pd
from avalanche.benchmarks import benchmark_with_validation_stream, nc_benchmark, dataset_benchmark, CLExperience
from torch.utils.data import Dataset
import torch
from torch.nn import CrossEntropyLoss

from avalanche.training.supervised import EWC, icarl, Naive, CWRStar, Replay, GDumb, LwF, GEM, AGEM, EWC
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, loss_metrics, timing_metrics, \
    cpu_usage_metrics, disk_usage_metrics
from avalanche.training.plugins import EWCPlugin, AGEMPlugin, GEMPlugin, ReplayPlugin, CWRStarPlugin, RWalkPlugin
from avalanche.logging import InteractiveLogger, TensorboardLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training import ExemplarsBuffer, ReservoirSamplingBuffer

from avalanche.benchmarks import CLExperience
from avalanche.models import DynamicModule


# %%

class TorchDataset2(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.targets = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        return self.X[item], self.targets[item]


def prep_benchmark(train_loc, test_loc):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    # Training Data

    # create subdatasets wit htheir labels the first with 2 classes then the others wih a single class
    data_train = pd.read_csv('./DATA/TRAIN_DATA.csv')
    # tansfrom the label column to long type
    data_train['label'] = data_train['label'].astype('int64')

    X1 = data_train[data_train['label'] == 0].iloc[:, :-1].values
    targets1 = data_train[data_train['label'] == 0].iloc[:, -1].values
    X2 = data_train[data_train['label'] == 1].iloc[:, :-1].values
    targets2 = data_train[data_train['label'] == 1].iloc[:, -1].values
    # Combine classes X1/Targets1 and  X2/Targets2 into a single dataset
    X1_2 = torch.cat((torch.tensor(X1, dtype=torch.float32), torch.tensor(X2, dtype=torch.float32)),
                     0)
    targets1_2 = torch.cat(
        (torch.tensor(targets1, dtype=torch.long), torch.tensor(targets2, dtype=torch.long)), 0)

    X3 = data_train[data_train['label'] == 2].iloc[:, :-1].values
    targets3 = data_train[data_train['label'] == 2].iloc[:, -1].values
    X4 = data_train[data_train['label'] == 3].iloc[:, :-1].values
    targets4 = data_train[data_train['label'] == 3].iloc[:, -1].values
    X5 = data_train[data_train['label'] == 4].iloc[:, :-1].values
    targets5 = data_train[data_train['label'] == 4].iloc[:, -1].values
    X6 = data_train[data_train['label'] == 5].iloc[:, :-1].values
    targets6 = data_train[data_train['label'] == 5].iloc[:, -1].values
    X7 = data_train[data_train['label'] == 6].iloc[:, :-1].values
    targets7 = data_train[data_train['label'] == 6].iloc[:, -1].values
    X8 = data_train[data_train['label'] == 7].iloc[:, :-1].values
    targets8 = data_train[data_train['label'] == 7].iloc[:, -1].values
    X9 = data_train[data_train['label'] == 8].iloc[:, :-1].values
    targets9 = data_train[data_train['label'] == 8].iloc[:, -1].values
    X10 = data_train[data_train['label'] == 9].iloc[:, :-1].values
    targets10 = data_train[data_train['label'] == 9].iloc[:, -1].values
    X11 = data_train[data_train['label'] == 10].iloc[:, :-1].values
    targets11 = data_train[data_train['label'] == 10].iloc[:, -1].values
    X12 = data_train[data_train['label'] == 11].iloc[:, :-1].values
    targets12 = data_train[data_train['label'] == 11].iloc[:, -1].values
    X13 = data_train[data_train['label'] == 12].iloc[:, :-1].values
    targets13 = data_train[data_train['label'] == 12].iloc[:, -1].values
    X14 = data_train[data_train['label'] == 13].iloc[:, :-1].values
    targets14 = data_train[data_train['label'] == 13].iloc[:, -1].values
    X15 = data_train[data_train['label'] == 14].iloc[:, :-1].values
    targets15 = data_train[data_train['label'] == 14].iloc[:, -1].values
    X16 = data_train[data_train['label'] == 15].iloc[:, :-1].values
    targets16 = data_train[data_train['label'] == 15].iloc[:, -1].values
    X17 = data_train[data_train['label'] == 16].iloc[:, :-1].values
    targets17 = data_train[data_train['label'] == 16].iloc[:, -1].values
    X18 = data_train[data_train['label'] == 17].iloc[:, :-1].values
    targets18 = data_train[data_train['label'] == 17].iloc[:, -1].values
    X19 = data_train[data_train['label'] == 18].iloc[:, :-1].values
    targets19 = data_train[data_train['label'] == 18].iloc[:, -1].values
    X20 = data_train[data_train['label'] == 19].iloc[:, :-1].values
    targets20 = data_train[data_train['label'] == 19].iloc[:, -1].values
    X21 = data_train[data_train['label'] == 20].iloc[:, :-1].values
    targets21 = data_train[data_train['label'] == 20].iloc[:, -1].values
    X22 = data_train[data_train['label'] == 21].iloc[:, :-1].values
    targets22 = data_train[data_train['label'] == 21].iloc[:, -1].values
    X23 = data_train[data_train['label'] == 22].iloc[:, :-1].values
    targets23 = data_train[data_train['label'] == 22].iloc[:, -1].values
    X24 = data_train[data_train['label'] == 23].iloc[:, :-1].values
    targets24 = data_train[data_train['label'] == 23].iloc[:, -1].values
    X25 = data_train[data_train['label'] == 24].iloc[:, :-1].values
    targets25 = data_train[data_train['label'] == 24].iloc[:, -1].values
    X26 = data_train[data_train['label'] == 25].iloc[:, :-1].values
    targets26 = data_train[data_train['label'] == 25].iloc[:, -1].values
    X27 = data_train[data_train['label'] == 26].iloc[:, :-1].values
    targets27 = data_train[data_train['label'] == 26].iloc[:, -1].values
    X28 = data_train[data_train['label'] == 27].iloc[:, :-1].values
    targets28 = data_train[data_train['label'] == 27].iloc[:, -1].values
    X29 = data_train[data_train['label'] == 28].iloc[:, :-1].values
    targets29 = data_train[data_train['label'] == 28].iloc[:, -1].values
    X30 = data_train[data_train['label'] == 29].iloc[:, :-1].values
    targets30 = data_train[data_train['label'] == 29].iloc[:, -1].values

    ds_train_1 = TorchDataset2(X1_2, targets1_2)
    ds_train_2 = TorchDataset2(X2, targets2)
    ds_train_3 = TorchDataset2(X3, targets3)
    ds_train_4 = TorchDataset2(X4, targets4)
    ds_train_5 = TorchDataset2(X5, targets5)
    ds_train_6 = TorchDataset2(X6, targets6)
    ds_train_7 = TorchDataset2(X7, targets7)
    ds_train_8 = TorchDataset2(X8, targets8)
    ds_train_9 = TorchDataset2(X9, targets9)
    ds_train_10 = TorchDataset2(X10, targets10)
    ds_train_11 = TorchDataset2(X11, targets11)
    ds_train_12 = TorchDataset2(X12, targets12)
    ds_train_13 = TorchDataset2(X13, targets13)
    ds_train_14 = TorchDataset2(X14, targets14)
    ds_train_15 = TorchDataset2(X15, targets15)
    ds_train_16 = TorchDataset2(X16, targets16)
    ds_train_17 = TorchDataset2(X17, targets17)
    ds_train_18 = TorchDataset2(X18, targets18)
    ds_train_19 = TorchDataset2(X19, targets19)
    ds_train_20 = TorchDataset2(X20, targets20)
    ds_train_21 = TorchDataset2(X21, targets21)
    ds_train_22 = TorchDataset2(X22, targets22)
    ds_train_23 = TorchDataset2(X23, targets23)
    ds_train_24 = TorchDataset2(X24, targets24)
    ds_train_25 = TorchDataset2(X25, targets25)
    ds_train_26 = TorchDataset2(X26, targets26)
    ds_train_27 = TorchDataset2(X27, targets27)
    ds_train_28 = TorchDataset2(X28, targets28)
    ds_train_29 = TorchDataset2(X29, targets29)
    ds_train_30 = TorchDataset2(X30, targets30)

    data_test = pd.read_csv('./DATA/TEST_DATA.csv')
    data_test['label'] = data_test['label'].astype('int64')

    Xt1 = data_test[data_test['label'] == 0].iloc[:, :-1].values
    targets_t1 = data_test[data_test['label'] == 0].iloc[:, -1].values
    Xt2 = data_test[data_test['label'] == 1].iloc[:, :-1].values
    targets_t2 = data_test[data_test['label'] == 1].iloc[:, -1].values
    # Combine classes Xt1/Targets_t1 and  Xt2/Targets_t2 into a single dataset for testing
    Xt1_2 = torch.cat((torch.tensor(Xt1, dtype=torch.float32), torch.tensor(Xt2, dtype=torch.float32)),
                      0)
    targets_t1_2 = torch.cat(
        (torch.tensor(targets_t1, dtype=torch.float32), torch.tensor(targets_t2, dtype=torch.float32))
        , dim=0)

    Xt3 = data_test[data_test['label'] == 2].iloc[:, :-1].values
    targets_t3 = data_test[data_test['label'] == 2].iloc[:, -1].values
    Xt4 = data_test[data_test['label'] == 3].iloc[:, :-1].values
    targets_t4 = data_test[data_test['label'] == 3].iloc[:, -1].values
    Xt5 = data_test[data_test['label'] == 4].iloc[:, :-1].values
    targets_t5 = data_test[data_test['label'] == 4].iloc[:, -1].values
    Xt6 = data_test[data_test['label'] == 5].iloc[:, :-1].values
    targets_t6 = data_test[data_test['label'] == 5].iloc[:, -1].values
    Xt7 = data_test[data_test['label'] == 6].iloc[:, :-1].values
    targets_t7 = data_test[data_test['label'] == 6].iloc[:, -1].values
    Xt8 = data_test[data_test['label'] == 7].iloc[:, :-1].values
    targets_t8 = data_test[data_test['label'] == 7].iloc[:, -1].values
    Xt9 = data_test[data_test['label'] == 8].iloc[:, :-1].values
    targets_t9 = data_test[data_test['label'] == 8].iloc[:, -1].values
    Xt10 = data_test[data_test['label'] == 9].iloc[:, :-1].values
    targets_t10 = data_test[data_test['label'] == 9].iloc[:, -1].values
    Xt11 = data_test[data_test['label'] == 10].iloc[:, :-1].values
    targets_t11 = data_test[data_test['label'] == 10].iloc[:, -1].values
    Xt12 = data_test[data_test['label'] == 11].iloc[:, :-1].values
    targets_t12 = data_test[data_test['label'] == 11].iloc[:, -1].values
    Xt13 = data_test[data_test['label'] == 12].iloc[:, :-1].values
    targets_t13 = data_test[data_test['label'] == 12].iloc[:, -1].values
    Xt14 = data_test[data_test['label'] == 13].iloc[:, :-1].values
    targets_t14 = data_test[data_test['label'] == 13].iloc[:, -1].values
    Xt15 = data_test[data_test['label'] == 14].iloc[:, :-1].values
    targets_t15 = data_test[data_test['label'] == 14].iloc[:, -1].values
    Xt16 = data_test[data_test['label'] == 15].iloc[:, :-1].values
    targets_t16 = data_test[data_test['label'] == 15].iloc[:, -1].values
    Xt17 = data_test[data_test['label'] == 16].iloc[:, :-1].values
    targets_t17 = data_test[data_test['label'] == 16].iloc[:, -1].values
    Xt18 = data_test[data_test['label'] == 17].iloc[:, :-1].values
    targets_t18 = data_test[data_test['label'] == 17].iloc[:, -1].values
    Xt19 = data_test[data_test['label'] == 18].iloc[:, :-1].values
    targets_t19 = data_test[data_test['label'] == 18].iloc[:, -1].values
    Xt20 = data_test[data_test['label'] == 19].iloc[:, :-1].values
    targets_t20 = data_test[data_test['label'] == 19].iloc[:, -1].values
    Xt21 = data_test[data_test['label'] == 20].iloc[:, :-1].values
    targets_t21 = data_test[data_test['label'] == 20].iloc[:, -1].values
    Xt22 = data_test[data_test['label'] == 21].iloc[:, :-1].values
    targets_t22 = data_test[data_test['label'] == 21].iloc[:, -1].values
    Xt23 = data_test[data_test['label'] == 22].iloc[:, :-1].values
    targets_t23 = data_test[data_test['label'] == 22].iloc[:, -1].values
    Xt24 = data_test[data_test['label'] == 23].iloc[:, :-1].values
    targets_t24 = data_test[data_test['label'] == 23].iloc[:, -1].values
    Xt25 = data_test[data_test['label'] == 24].iloc[:, :-1].values
    targets_t25 = data_test[data_test['label'] == 24].iloc[:, -1].values
    Xt26 = data_test[data_test['label'] == 25].iloc[:, :-1].values
    targets_t26 = data_test[data_test['label'] == 25].iloc[:, -1].values
    Xt27 = data_test[data_test['label'] == 26].iloc[:, :-1].values
    targets_t27 = data_test[data_test['label'] == 26].iloc[:, -1].values
    Xt28 = data_test[data_test['label'] == 27].iloc[:, :-1].values
    targets_t28 = data_test[data_test['label'] == 27].iloc[:, -1].values
    Xt29 = data_test[data_test['label'] == 28].iloc[:, :-1].values
    targets_t29 = data_test[data_test['label'] == 28].iloc[:, -1].values
    Xt30 = data_test[data_test['label'] == 29].iloc[:, :-1].values
    targets_t30 = data_test[data_test['label'] == 29].iloc[:, -1].values

    ds_test_1 = TorchDataset2(Xt1_2, targets_t1_2)
    ds_test_2 = TorchDataset2(Xt2, targets_t2)
    ds_test_3 = TorchDataset2(Xt3, targets_t3)
    ds_test_4 = TorchDataset2(Xt4, targets_t4)
    ds_test_5 = TorchDataset2(Xt5, targets_t5)
    ds_test_6 = TorchDataset2(Xt6, targets_t6)
    ds_test_7 = TorchDataset2(Xt7, targets_t7)
    ds_test_8 = TorchDataset2(Xt8, targets_t8)
    ds_test_9 = TorchDataset2(Xt9, targets_t9)
    ds_test_10 = TorchDataset2(Xt10, targets_t10)
    ds_test_11 = TorchDataset2(Xt11, targets_t11)
    ds_test_12 = TorchDataset2(Xt12, targets_t12)
    ds_test_13 = TorchDataset2(Xt13, targets_t13)
    ds_test_14 = TorchDataset2(Xt14, targets_t14)
    ds_test_15 = TorchDataset2(Xt15, targets_t15)
    ds_test_16 = TorchDataset2(Xt16, targets_t16)
    ds_test_17 = TorchDataset2(Xt17, targets_t17)
    ds_test_18 = TorchDataset2(Xt18, targets_t18)
    ds_test_19 = TorchDataset2(Xt19, targets_t19)
    ds_test_20 = TorchDataset2(Xt20, targets_t20)
    ds_test_21 = TorchDataset2(Xt21, targets_t21)
    ds_test_22 = TorchDataset2(Xt22, targets_t22)
    ds_test_23 = TorchDataset2(Xt23, targets_t23)
    ds_test_24 = TorchDataset2(Xt24, targets_t24)
    ds_test_25 = TorchDataset2(Xt25, targets_t25)
    ds_test_26 = TorchDataset2(Xt26, targets_t26)
    ds_test_27 = TorchDataset2(Xt27, targets_t27)
    ds_test_28 = TorchDataset2(Xt28, targets_t28)
    ds_test_29 = TorchDataset2(Xt29, targets_t29)
    ds_test_30 = TorchDataset2(Xt30, targets_t30)

    train_ds = [ds_train_1, ds_train_2, ds_train_3, ds_train_4, ds_train_5, ds_train_6, ds_train_7, ds_train_8,
                ds_train_9, ds_train_10, ds_train_11, ds_train_12, ds_train_13, ds_train_14, ds_train_15, ds_train_16,
                ds_train_17, ds_train_18, ds_train_19, ds_train_20, ds_train_21, ds_train_22, ds_train_23, ds_train_24,
                ds_train_25, ds_train_26, ds_train_27, ds_train_28, ds_train_29, ds_train_30]
    test_ds = [ds_test_1, ds_test_2, ds_test_3, ds_test_4, ds_test_5, ds_test_6, ds_test_7, ds_test_8, ds_test_9,
               ds_test_10, ds_test_11, ds_test_12, ds_test_13, ds_test_14, ds_test_15, ds_test_16, ds_test_17,
               ds_test_18, ds_test_19, ds_test_20, ds_test_21, ds_test_22, ds_test_23, ds_test_24, ds_test_25,
               ds_test_26, ds_test_27, ds_test_28, ds_test_29, ds_test_30]

    return dataset_benchmark(train_datasets=train_ds, test_datasets=test_ds, )
    # hdata_train = TorchDataset(train_loc)
    # hdata_test = TorchDataset(test_loc)
    #
    # return benchmark_with_validation_stream(nc_benchmark(train_dataset=hdata_train, test_dataset=hdata_test
    #                                                      , shuffle=True, seed=1234, task_labels=True, n_experiences=5,
    #                                                      one_dataset_per_exp=True,
    #
    #                                                      ))



class IncrementalClassifierD1(DynamicModule):
    """
    Output layer that incrementally adds units whenever new classes are
    encountered.

    Typically used in class-incremental benchmarks where the number of
    classes grows over time.
    """

    def __init__(
            self,
            in_features,
            initial_out_features=2,
            masking=True,
            mask_value=-1000,
    ):
        """
        :param in_features: number of input features.
        :param initial_out_features: initial number of classes (can be
            dynamically expanded).
        :param masking: whether unused units should be masked (default=True).
        :param mask_value: the value used for masked units (default=-1000).
        """
        super().__init__()
        self.masking = masking
        self.mask_value = mask_value
        # self.features = nn.Sequential(
        # nn.Conv1d(in_channels=1, out_channels=32, padding='same',  kernel_size=3,),
        # nn.MaxPool1d(4),
        # nn.Conv1d(in_channels=32, out_channels=32, padding='same',  kernel_size=3),
        # nn.MaxPool1d(4),
        # nn.Conv1d(in_channels=32, out_channels=16, padding='same',  kernel_size=3),
        # nn.MaxPool1d(4),
        # nn.Conv1d(in_channels=16, out_channels=16, padding='same',  kernel_size=3),
        # nn.MaxPool1d(4),
        # )
        #
        # self.fc1 = nn.Linear(256, 300)
        # self.fc2 = nn.Linear(300, 128)
        # self.fc3 = nn.Linear(128, 31)
        self.conv1D_1 = nn.Conv1d(in_channels=1, out_channels=32, padding='same', kernel_size=3, )
        self.maxPool1D_1 = nn.MaxPool1d(4)
        self.conv1D_2 = nn.Conv1d(in_channels=32, out_channels=32, padding='same', kernel_size=3)
        self.maxPool1D_2 = nn.MaxPool1d(4)
        self.conv1D_3 = nn.Conv1d(in_channels=32, out_channels=16, padding='same', kernel_size=3)
        self.maxPool1D_3 = nn.MaxPool1d(4)
        self.conv1D_4 = nn.Conv1d(in_channels=16, out_channels=16, padding='same', kernel_size=3)
        self.maxPool1D_4 = nn.MaxPool1d(4)

        self.fc1 = nn.Linear(256, 300)
        self.fc2 = nn.Linear(300, 128)
        self.fc3 = nn.Linear(128, 31)
        self.classifier = nn.Linear(256, initial_out_features)
        # self.initial_out_features = initial_out_features

        # self.classifier = torch.nn.Linear(in_features, initial_out_features)
        au_init = torch.zeros(initial_out_features, dtype=torch.bool)
        self.register_buffer("active_units", au_init)

    @torch.no_grad()
    def adaptation(self, experience: CLExperience):
        """If `dataset` contains unseen classes the classifier is expanded.

        :param experience: data from the current experience.
        :return:
        """
        in_features = self.classifier.in_features
        old_nclasses = self.classifier.out_features
        curr_classes = experience.classes_in_this_experience
        new_nclasses = max(self.classifier.out_features, max(curr_classes) + 1)

        # update active_units mask
        if self.masking:
            if old_nclasses != new_nclasses:  # expand active_units mask
                old_act_units = self.active_units
                self.active_units = torch.zeros(new_nclasses, dtype=torch.bool)
                self.active_units[: old_act_units.shape[0]] = old_act_units
            # update with new active classes
            if self.training:
                self.active_units[curr_classes] = 1

        # update classifier weights
        if old_nclasses == new_nclasses:
            return
        old_w, old_b = self.classifier.weight, self.classifier.bias
        self.classifier = torch.nn.Linear(in_features, new_nclasses)
        self.classifier.weight[:old_nclasses] = old_w
        self.classifier.bias[:old_nclasses] = old_b

    def forward(self, x, **kwargs):
        """compute the output given the input `x`. This module does not use
        the task label.

        :param x:
        :return:
        """

        # block 1
        x = x.unsqueeze(dim=1)
        x = self.conv1D_1(x)
        x = torch.relu(x)
        x = self.maxPool1D_1(x)

        # block 2
        x = self.conv1D_2(x)
        x = torch.relu(x)
        x = self.maxPool1D_2(x)

        # block 3
        x = self.conv1D_3(x)
        x = torch.relu(x)
        x = self.maxPool1D_3(x)

        # block 4
        x = self.conv1D_4(x)
        x = torch.relu(x)
        x = self.maxPool1D_4(x)

        # Flatten

        x = x.reshape(x.shape[0], -1)

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        out = torch.log_softmax(x, dim=1)

        # if self.masking:
        #     mask = torch.logical_not(self.active_units)
        #     out[torch.unsqueeze(mask, dim=-1)] = self.mask_value
        # return out
        return out


# %%
features = nn.Sequential(
    nn.Conv1d(in_channels=1, out_channels=32, padding='same', kernel_size=3, ),
    nn.MaxPool1d(4),
    nn.Conv1d(in_channels=32, out_channels=32, padding='same', kernel_size=3),
    nn.MaxPool1d(4),
    nn.Conv1d(in_channels=32, out_channels=16, padding='same', kernel_size=3),
    nn.MaxPool1d(4),
    nn.Conv1d(in_channels=16, out_channels=16, padding='same', kernel_size=3),
    nn.MaxPool1d(4),
)
sample_input = torch.randn(1, 1, 4096)

sample_output = features(sample_input)
print(sample_output.shape)
smaple_output = sample_output.view(sample_output.size(0), -1)

print(sample_output.shape)

# %%
from avalanche.models import IncrementalClassifier, DynamicModule

model4 = IncrementalClassifierD1(in_features=4096, masking=True)
# %%

benchmark = prep_benchmark(train_loc='./DATA/TRAIN_DATA.csv', test_loc='./DATA/TEST_DATA.csv')

# log to Tensorboard
tb_logger = TensorboardLogger()

# log to text file
text_logger = TextLogger(open('log.txt', 'a'))

# print to stdout
interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True, epoch_running=True),
    forgetting_metrics(experience=True, stream=True),
    cpu_usage_metrics(experience=True),
    # confusion_matrix_metrics(num_classes=benchmark['inc_bench'].n_classes, save_image=False,
    #                          stream=True),
    disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loggers=[interactive_logger, text_logger, tb_logger]
)
# %%
cl_strategy = EWC(
    model=model4,
    optimizer=torch.optim.Adam(model4.parameters(), lr=1e-3),
    criterion=CrossEntropyLoss(),
    train_mb_size=500, train_epochs=20, eval_mb_size=100,
    ewc_lambda=0.4,
    evaluator=eval_plugin,
    plugins=[ReplayPlugin(mem_size=10000, storage_policy=ReservoirSamplingBuffer(max_size=10000)),
             ]
)

# TRAINING LOOP
print('Starting experiment...')
results = []
model_incs = []
classes_exp = []
for experience in benchmark.train_stream:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)
    # model_incs.append(model4.classifier.out_features)
    classes_exp.append(experience.classes_in_this_experience)
    # train returns a dictionary which contains all the metric values
    res = cl_strategy.train(experience)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    # test also returns a dictionary which contains all the metric values
    results.append(cl_strategy.eval(benchmark.test_stream))
