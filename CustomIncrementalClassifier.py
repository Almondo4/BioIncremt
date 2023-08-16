# %%

from torch import nn

import pandas as pd
from avalanche.benchmarks import benchmark_with_validation_stream, nc_benchmark
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


# %%

class TorchDataset(Dataset):

    def __init__(self, filePath):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Read CSV
        data = pd.read_csv(filePath)
        # drop one class from dataset to make it 30 classes
        data = data[data['label'] != 30]

        self.X = data.iloc[:, :-1].values
        self.targets = data.iloc[:, -1].values

        # Feature Scale if you want

        # Convert to Torch Tensors
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.int)

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
                                                         , shuffle=True, seed=1234, task_labels=True, n_experiences=5,
                                                         one_dataset_per_exp=True,

                                                         ))


# %%
from avalanche.benchmarks import CLExperience
from avalanche.models import DynamicModule


# implement a Incremental classifier with a custom classifier
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
from avalanche.models import IncrementalClassifier

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
    train_mb_size=500, train_epochs=200, eval_mb_size=100,
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
