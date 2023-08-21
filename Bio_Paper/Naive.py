from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, timing_metrics, forgetting_metrics, \
    cpu_usage_metrics, disk_usage_metrics
from avalanche.logging import TensorboardLogger, TextLogger, InteractiveLogger
from avalanche.training import EWC, ReservoirSamplingBuffer
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin
from torch.nn import CrossEntropyLoss

from Incremental1DCNN import Incremental1DCNNClassifier
import DataPrep
import torch
import yaml

model4 = Incremental1DCNNClassifier(in_features=4096, masking=True)
# %%

train_datasets, test_datasets, benchmark = DataPrep.prep_incremental_benchmark(train_loc='./DATA/TRAIN_DATA.csv',
                                                                               test_loc='./DATA/TEST_DATA.csv')

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
    performance_profiler['Evaluation Accuracy'].append({'Exp_' + formatted_i: cl_strategy.evaluator.get_last_metrics()[
        f'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{formatted_i}']
                                                        })
    training_accuracies = \
    cl_strategy.evaluator.get_all_metrics()[f'Top1_Acc_Epoch/train_phase/train_stream/Task{formatted_i}'][1]
    print(training_accuracies.count())
    performance_profiler['Evaluation Accuracy'].append(
        {'Exp_' + formatted_i: sum(training_accuracies) / len(training_accuracies)})

    performance_profiler['Active Units'].append({'Exp_' + formatted_i: model_incs[i]})
print(performance_profiler)

with open("evaluation_results.yaml", "w") as yaml_file:
    yaml.dump(performance_profiler, yaml_file)
