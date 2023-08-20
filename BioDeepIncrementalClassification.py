
from torch import nn

import pandas as pd
from avalanche.benchmarks import benchmark_with_validation_stream, nc_benchmark, dataset_benchmark, CLExperience
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
import torch
from avalanche.training.supervised import EWC, icarl, Naive, CWRStar, Replay, GDumb, LwF, GEM, AGEM, EWC
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, loss_metrics, timing_metrics, \
    cpu_usage_metrics, disk_usage_metrics
from avalanche.training.plugins import EWCPlugin, AGEMPlugin, GEMPlugin, ReplayPlugin, CWRStarPlugin, RWalkPlugin
from avalanche.logging import InteractiveLogger, TensorboardLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training import ExemplarsBuffer, ReservoirSamplingBuffer

from avalanche.benchmarks import CLExperience
from avalanche.models import DynamicModule

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


def prep_benchmark(train_loc, test_loc):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    data_train = pd.read_csv(train_loc)
    data_train['label'] = data_train['label'].astype('int64')
    data_test = pd.read_csv(test_loc)
    data_test['label'] = data_test['label'].astype('int64')

    train_datasets = []
    test_datasets = []

    #print the size of both train and test data
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


    # Prepare the remaining datasets with labels 2 to 29
    for label in range(2, 30):
        train_datasets.append(prepare_single_dataset(data_train, label))
        test_datasets.append(prepare_single_dataset(data_test, label))

    # return all the datasets within a set
    return train_datasets, test_datasets,dataset_benchmark(train_datasets=train_datasets, test_datasets=test_datasets)



    # return dataset_benchmark(train_datasets=train_datasets, test_datasets=test_datasets)


######

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
        self.conv1D_1 = nn.Conv1d(in_channels=1, out_channels=32, padding='same', kernel_size=3, )
        self.maxPool1D_1 = nn.MaxPool1d(4)
        self.conv1D_2 = nn.Conv1d(in_channels=32, out_channels=32, padding='same', kernel_size=3)
        self.maxPool1D_2 = nn.MaxPool1d(4)
        self.conv1D_3 = nn.Conv1d(in_channels=32, out_channels=16, padding='same', kernel_size=3)
        self.maxPool1D_3 = nn.MaxPool1d(4)
        self.conv1D_4 = nn.Conv1d(in_channels=16, out_channels=16, padding='same', kernel_size=3)
        self.maxPool1D_4 = nn.MaxPool1d(4)

        self.initial_out_features = initial_out_features


        self.fc1 = nn.Linear(256, 300)
        self.fc2 = nn.Linear(300, 128)


        self.classifier = nn.Linear(128, initial_out_features)


        # self.classifier = torch.nn.Linear(in_features, initial_out_features)
        au_init = torch.zeros(initial_out_features, dtype=torch.bool)
        self.register_buffer("active_units", au_init)

    # @torch.no_grad()
    # def adaptation(self, experience: CLExperience):
    #     """If `dataset` contains unseen classes the classifier is expanded.
    #
    #     :param experience: data from the current experience.
    #     :return:
    #     """
    #     in_features = self.classifier.in_features
    #     old_nclasses = self.classifier.out_features
    #     curr_classes = experience.classes_in_this_experience
    #     new_nclasses = max(self.classifier.out_features, max(curr_classes) + 1)
    #
    #     # update active_units mask
    #     if self.masking:
    #         if old_nclasses != new_nclasses:  # expand active_units mask
    #             old_act_units = self.active_units
    #             self.active_units = torch.zeros(new_nclasses, dtype=torch.bool)
    #             self.active_units[: old_act_units.shape[0]] = old_act_units
    #         # update with new active classes
    #         if self.training:
    #             self.active_units[curr_classes] = 1
    #
    #     # update classifier weights
    #     if old_nclasses == new_nclasses:
    #         return
    #     old_w, old_b = self.classifier.weight, self.classifier.bias
    #     self.classifier = torch.nn.Linear(in_features, new_nclasses)
    #     self.classifier.weight[:old_nclasses] = old_w
    #     self.classifier.bias[:old_nclasses] = old_b
    @torch.no_grad()
    def adaptation(self, experience: CLExperience):
        in_features = self.classifier.in_features
        old_nclasses = self.classifier.out_features
        curr_classes = experience.classes_in_this_experience
        new_nclasses = max(self.classifier.out_features, max(curr_classes) + 1)

        # update active_units mask
        if self.masking:
            if old_nclasses != new_nclasses:  # expand active_units mask
                old_act_units = self.active_units.clone()
                self.active_units = torch.zeros(new_nclasses, dtype=torch.bool)
                self.active_units[: old_act_units.shape[0]] = old_act_units
            # update with new active classes
            if self.training:
                self.active_units[curr_classes] = 1

        # update classifier weights
        if old_nclasses != new_nclasses:
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
        # x = self.fc3(x)
        x = self.classifier(x)
        out = torch.log_softmax(x, dim=1)

        if self.masking:
            masked_out = out.clone()  # Make a copy of the output tensor
            masked_out[..., torch.logical_not(self.active_units)] = self.mask_value
            return masked_out
        else:
            return out


model4 = IncrementalClassifierD1(in_features=4096, masking=True)
# %%

train_datasets, test_datasets,benchmark = prep_benchmark(train_loc='./DATA/TRAIN_DATA.csv', test_loc='./DATA/TEST_DATA.csv')


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
    train_mb_size=500, train_epochs=10, eval_mb_size=100,
    ewc_lambda=0.4,
    evaluator=eval_plugin,
    plugins=[ReplayPlugin(mem_size=1000, storage_policy=ReservoirSamplingBuffer(max_size=1000)),
             ]
)

# TRAINING LOOP
print('Starting experiment...')
results = []
model_incs = []
classes_exp = []
import yaml

all_results = {}
for experience in benchmark.train_stream:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    classes_exp.append(experience.classes_in_this_experience)
    # train returns a dictionary which contains all the metric values
    res = cl_strategy.train(experience)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    # test also returns a dictionary which contains all the metric values
    results.append(cl_strategy.eval(benchmark.test_stream))
    model_incs.append(torch.sum(model4.active_units).item())
    print("Model's classifier layer output shape: ", torch.sum(model4.active_units).item())

performance_profiler = {}

performance_profiler['Active Units'] = []
performance_profiler['Evaluation Accuracy'] = []
performance_profiler['Training Accuracy'] = []
for i in range(0, benchmark.n_experiences - 1):
    formatted_i = f"{i:03d}"
    performance_profiler['Evaluation Accuracy'].append({'Exp_'+formatted_i:    cl_strategy.evaluator.get_last_metrics()[f'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{formatted_i}']
 })
    training_accuracies= cl_strategy.evaluator.get_all_metrics()[f'Top1_Acc_Epoch/train_phase/train_stream/Task{formatted_i}'][1]
    print(training_accuracies.count())
    performance_profiler['Evaluation Accuracy'].append({'Exp_' + formatted_i: sum(training_accuracies)/len(training_accuracies)})

    performance_profiler['Active Units'].append({'Exp_'+formatted_i:  model_incs[i]})
print (performance_profiler)

with open("evaluation_results.yaml", "w") as yaml_file:
    yaml.dump(performance_profiler, yaml_file)

    # Write a todo block list
    # 1. Make the first experience include 2 classes
    # 2. Save training results from all metrics at each stage and save it for comparison
        # 2.1. What can be achived is to sample 10 samples from each previous datasets every 5 expereinces to a total of 6 tests
        # 2.2. The results can be saved in a nested dictionary with the following structure
            # expereince 5/10/15/20/25/30
                # Accuracies (0 or exprience -1) -> expereince: Accuracy
                # Average
    # 3. BWT  = Accuracy at experience 5/10/15/20/25/30 can be calculated from a nother cumulative dataset and the bwt transfer would would be representing the cumulative approach
    # 4. Try a different implementation of the model