import torch
from torch import nn
from torchmetrics import Accuracy, F1Score, Precision, AUROC
import pytorch_lightning as pl
import torchsummary

from torchmetrics import Accuracy, F1Score, Precision, AUROC, ConfusionMatrix
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pandas as pd
from avalanche.benchmarks import benchmark_with_validation_stream, nc_benchmark
from torch.utils.data import Dataset
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from avalanche.benchmarks.classic import SplitMNIST
from avalanche.models import MTSimpleMLP, IncrementalClassifier, MultiHeadClassifier, PNN
from avalanche.training.supervised import EWC
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, loss_metrics, timing_metrics, \
    cpu_usage_metrics, disk_usage_metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin
#%%




class TorchDataset(Dataset):

    def __init__(self,filePath):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Read CSV
        data = pd.read_csv(filePath)

        self.X = data.iloc[:,:-1].values
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
                                 , shuffle=True, seed=1234, task_labels=True,n_experiences=31,one_dataset_per_exp=True,



                                                         ))



  # per_exp_classes={0:0,
  #                                                                         1:1,
  #                                                                         2:2,
  #                                                                         3:3,
  #                                                                         4:4,
  #                                                                         5:5,
  #                                                                         6:6,
  #                                                                         7:7,
  #                                                                         8:8,
  #                                                                         9:9,
  #                                                                         10:10,
  #                                                                         11:11,
  #                                                                         12:12,
  #                                                                         13:13,
  #                                                                         14:14,
  #                                                                         15:15,
  #                                                                         16:16,
  #                                                                         17:17,
  #                                                                         18:18,
  #                                                                         19:19,
  #                                                                         20:20,
  #                                                                         21:21,
  #                                                                         22:22,
  #                                                                         23:23,
  #                                                                         24:24,
  #                                                                         25:25,
  #                                                                         26:26,
  #                                                                         27:27,
  #                                                                         28:28,
  #                                                                         29:29,
  #                                                                         30:30,}



# model4 = IncrementalClassifier(in_features=4096, initial_out_features=2,masking=False)
model4 = PNN(in_features=4096,)



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

# CREATE THE STRATEGY INSTANCE (EWC)
# model2 = avl.models.PNN(in_features=4096,hidden_features_per_column=1000,num_layers=7,)
# model3 =  MTSimpleCNN()


# TODO: 4096 is too much i might need to pass thought the cnn first and then pass it to the pnn
cl_strategy = EWC(
    model4, torch.optim.Adam(model4.parameters(), lr=0.001,),
    CrossEntropyLoss(), train_mb_size=500, train_epochs=2, eval_mb_size=100,
    evaluator=eval_plugin,ewc_lambda=0.4)

# TRAINING LOOP
print('Starting experiment...')
results = []
model_incs = []
classes_exp= []
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