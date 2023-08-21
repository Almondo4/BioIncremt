from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, timing_metrics, forgetting_metrics, \
    cpu_usage_metrics, disk_usage_metrics
from avalanche.logging import TensorboardLogger, TextLogger, InteractiveLogger
from avalanche.training import EWC, ReservoirSamplingBuffer
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy

from Incremental1DCNN import Incremental1DCNNClassifier
import DataPrep
import torch
import yaml

model4 = Incremental1DCNNClassifier(in_features=4096, masking=True)
# %%

train_datasets, test_datasets, benchmark, cumulative_benchmark = DataPrep.prep_incremental_benchmark \
    (train_loc='../DATA/TRAIN_DATA.csv', test_loc='../DATA/TEST_DATA.csv')

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
    plugins=[ReplayPlugin(mem_size=1000, storage_policy=ReservoirSamplingBuffer(max_size=1000)),
             ]
)

# TRAINING LOOP
print('Starting experiment...')
results = []
model_incs = []
classes_exp = []
cumulative_set = 0
cumulative_set_results = []
all_results = {}


for exp_id, experience in enumerate(benchmark.train_stream):
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    classes_exp.append(experience.classes_in_this_experience)
    # train returns a dictionary which contains all the metric values
    res = cl_strategy.train(experience)
    print('Training completed')

    # TODO Evaluate the current training accuracy on test set
    print('Computing accuracy on test set')
    results.append(cl_strategy.eval(benchmark.test_stream))

    model_incs.append(torch.sum(model4.active_units).item())
    print("Model's classifier layer output shape: ", torch.sum(model4.active_units).item())

    # Cumulative results
    ## if the expriec % 5 == 0 then evaluate on the cumulative benchmark and save the results
    if (exp_id ) % 5 == 0: #TODO: starting fom 0 and also 2 sets are combined -- was 1
        print(f'Computing accuracy on the cumulative test set {cumulative_set}:')
        cumulative_set += 1

        with torch.no_grad():
            y_pred_logits = cl_strategy.model(torch.tensor(cumulative_benchmark[cumulative_set].X, dtype=torch.float32))
            y_pred_labels = torch.argmax(y_pred_logits, dim=1)

        unique_classes = torch.unique(cumulative_benchmark[cumulative_set].targets)
        num_cumul_cls = len(unique_classes.tolist())
        accuracy = Accuracy(task='multiclass', num_classes=num_cumul_cls)
        acc = accuracy(y_pred_labels, cumulative_benchmark[cumulative_set].targets)
        print(f'Accuracy on cumulative_experience_{exp_id}: ', acc)
        cumulative_set_results.append({'Exp_' + str(exp_id)+f'({torch.sum(model4.active_units).item()} acs)': acc.item()})
        #reset accuracy
        accuracy.reset()

print('Computing accuracy on test set')
results.append(cl_strategy.eval(benchmark.test_stream))

performance_profiler = {}
performance_profiler['Active Units'] = []
performance_profiler['Evaluation Accuracy'] = []
performance_profiler['Training Accuracy'] = []

# 1. Evaluation Accuracy after last experience
for i in range(0, benchmark.n_experiences - 1):
    formatted_i = f"{i:03d}"
    performance_profiler['Evaluation Accuracy'].append({'Exp_' + formatted_i: cl_strategy.evaluator.get_last_metrics()[
        f'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{formatted_i}']
                                                        })
    training_accuracies = \
    cl_strategy.evaluator.get_all_metrics()[f'Top1_Acc_Epoch/train_phase/train_stream/Task{formatted_i}'][1]
    # print(training_accuracies.count())
    performance_profiler['Evaluation Accuracy'].append(
        {'Exp_' + formatted_i: sum(training_accuracies) / len(training_accuracies)})

    performance_profiler['Active Units'].append({'Exp_' + formatted_i: model_incs[i]})
print(performance_profiler)

with open("evaluation_results.yaml", "w") as yaml_file:
    yaml.dump(performance_profiler, yaml_file)

#print classes in first dataset in cumulative benchmark
classes = torch.unique(cumulative_benchmark[0].targets)
#get classes as list and transform it to string where each key is the class and the frequency is the value
num_cumul_cls = cumulative_benchmark[0].targets.tolist()
num_cumul_cls = [str(i) for i in num_cumul_cls]
num_cumul_cls = {i: num_cumul_cls.count(i) for i in num_cumul_cls}
print('Classes in first dataset in cumulative benchmark: ', num_cumul_cls)

# print('Classes in first dataset in cumulative benchmark: ', cumulative_benchmark[0].classes_in_this_dataset)