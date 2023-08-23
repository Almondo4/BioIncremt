import yaml
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, timing_metrics, forgetting_metrics, \
    cpu_usage_metrics, disk_usage_metrics, EpochCPUUsage, EpochTime
from avalanche.logging import TensorboardLogger, TextLogger, InteractiveLogger
from avalanche.training import JointTraining, ReservoirSamplingBuffer
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy

from Incremental1DCNN import Incremental1DCNNClassifier
import DataPrep
import torch


model = Incremental1DCNNClassifier(in_features=4096, masking=True)
# %%

train_datasets, test_datasets, benchmark, cumulative_benchmark = DataPrep.prep_incremental_benchmark \
    (train_loc='../DATA/TRAIN_DATA.csv', test_loc='../DATA/TEST_DATA.csv')

tb_logger = TensorboardLogger()

# log to text file
text_logger = TextLogger(open('log.txt', 'a'))

# print to stdout
interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    # timing_metrics(epoch=True, epoch_running=True),
    forgetting_metrics(experience=True, stream=True),
    cpu_usage_metrics(experience=True,),
    EpochCPUUsage(),
    EpochTime(),
    disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loggers=[interactive_logger, text_logger, tb_logger]
)
# %%
cl_strategy = JointTraining(
    model=model,
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    criterion=CrossEntropyLoss(),
    train_mb_size=500, train_epochs=200, eval_mb_size=100,
    evaluator=eval_plugin,

)

# TRAINING LOOP
print('Starting experiment...')
results = []
model_incs = []
classes_exp = []
cumulative_set = 0


#Training for joint training
try:
    print('benchmark_Joint.train_stream: ', benchmark.train_stream.__len__())
    res = cl_strategy.train(benchmark.train_stream, )
except Exception as e:
    print(e)
# Testing for each particular class
eval_res = cl_strategy.eval(benchmark.test_stream)
testing_accuracy = []
for i in range(30):
    testing_accuracy.append(eval_res[f'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{i:03d}'])
print('Testing accuracy for each class: ', testing_accuracy)
#testing on total set
with torch.no_grad():
    y_pred_logits = cl_strategy.model(torch.tensor(cumulative_benchmark[-1].X, dtype=torch.float32))
    y_pred_labels = torch.argmax(y_pred_logits, dim=1)
unique_classes = torch.unique(cumulative_benchmark[-1].targets)
accuracy = Accuracy(task='multiclass', num_classes=31)
acc = accuracy(y_pred_labels, cumulative_benchmark[-1].targets)
print('Accuracy on cumulative_experience: ', acc)
accuracy.reset()
#reset accuracy
accuracy.reset()


# Cumulative Testing
cumulative_test_results = {}
for set in cumulative_benchmark:
    with torch.no_grad():
        y_pred_logits = cl_strategy.model(torch.tensor(set.X, dtype=torch.float32))
        y_pred_labels = torch.argmax(y_pred_logits, dim=1)

    unique_classes = torch.unique(set.targets)
    num_cumul_cls = len(unique_classes.tolist())
    accuracy = Accuracy(task='multiclass', num_classes=31)
    acc = accuracy(y_pred_labels, set.targets)
    print('Accuracy on cumulative_experience: ', acc)
    #reset accuracy
    accuracy.reset()
    # ad to dict cumulative_final_results
    cumulative_test_results['Cumulative Set ' + str(cumulative_benchmark.index(set))] = acc.item()

# Save the results in a dictionary and save it in a yaml file
all_results = {'Average Accuracy': acc,
               'Comprehensive accuracy': testing_accuracy,
               'Cumulative Test Results': cumulative_test_results,
               }
with open('Joint_200_results.yaml', 'w') as file:
    yaml.dump(all_results, file)