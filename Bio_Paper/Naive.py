
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics,  forgetting_metrics, \
    cpu_usage_metrics, disk_usage_metrics, EpochCPUUsage, EpochTime
from avalanche.logging import TensorboardLogger, TextLogger, InteractiveLogger
from avalanche.training import Naive
from avalanche.training.plugins import EvaluationPlugin
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy

from Incremental1DCNN import Incremental1DCNNClassifier
import IncrementalDataPrep
import torch


model4 = Incremental1DCNNClassifier(in_features=4096, masking=True)
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
cl_strategy = Naive(
    model=model4,
    optimizer=torch.optim.Adam(model4.parameters(), lr=1e-3),
    criterion=CrossEntropyLoss(),
    train_mb_size=500, train_epochs=20, eval_mb_size=100,
    evaluator=eval_plugin,

)

# TRAINING LOOP
print('Starting experiment...')
results = []
model_incs = []
classes_exp = []
cumulative_set = 0
cumulative_training_results = []
all_results = {}

cumulative_test_expriences = [4,9,14,19,24,29]

for exp_id, experience in enumerate(benchmark.train_stream):
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    classes_exp.append(experience.classes_in_this_experience)
    # train returns a dictionary which contains all the metric values
    res = cl_strategy.train(experience)
    print('Training completed')

    # Evaluate the current training accuracy on relative test streams (for comprehensive training results)
    print('Computing accuracy on test set')
    results.append(cl_strategy.eval(benchmark.test_stream[:experience.current_experience + 1]))
    model_incs.append(torch.sum(model4.active_units).item())
    print("Model's classifier layer output shape: ", torch.sum(model4.active_units).item())

    # Cumulative training results
    if exp_id  in cumulative_test_expriences:
        print(f'Computing accuracy on the cumulative test set {cumulative_set}:')
        with torch.no_grad():
            y_pred_logits = cl_strategy.model(torch.tensor(cumulative_benchmark[cumulative_set].X, dtype=torch.float32))
            y_pred_labels = torch.argmax(y_pred_logits, dim=1)

        unique_classes = torch.unique(cumulative_benchmark[cumulative_set].targets)
        num_cumul_cls = len(unique_classes.tolist())
        accuracy = Accuracy(task='multiclass', num_classes=num_cumul_cls)
        acc = accuracy(y_pred_labels, cumulative_benchmark[cumulative_set].targets)
        print(f'Accuracy on cumulative_experience_{exp_id}: ', acc)
        cumulative_training_results.append({'Exp_' + str(exp_id) + f'({torch.sum(model4.active_units).item()} acs)': acc.item()})
        #reset accuracy
        accuracy.reset()
        cumulative_set += 1




#for eahc incremental set in benchmark print the classes in it with exp id
for i, exp in enumerate(benchmark.train_stream):
    print(f'Classes in experience {i}: ', exp.classes_in_this_experience)

# for each culumlative dataset in cumulative benchmark claculate the accuracy on it using strategy.model and put the resutls into a dict with key as the cumulative set and value as the accuracy
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



# from the results list create a dict with key as the experience and valu as a sub dict where each key is an experience id extracted from Top1_acc_Exp and the value is  the value of that element. when done from expriences also append the key and value of 'Top1_Acc_Stream/eval_phase/test_stream/Task000' as Stream Accuracy . the following is a sample of the results item in the results list:

comprehensive_training_results = {}
for idx, result in enumerate(results):
    exp_results = {}  # Initialize the dictionary for this iteration
    stream_acc_key = 'Top1_Acc_Stream/eval_phase/test_stream/Task000'
    for key, value in result.items():
        if 'Top1_Acc_Exp' in key:
            exp_id = key.split('/')[-1]
            if f'Training Exprience({idx})' not in exp_results:
                exp_results[f'Training Exprience({idx})'] = {}  # Initialize the sub-dictionary if not present
            exp_results[f'Training Exprience({idx})'][exp_id] = value
    # Append the stream accuracy to the experience results
    if f'Training Exprience({idx})' not in exp_results:
        exp_results[f'Training Exprience({idx})'] = {}  # Initialize the sub-dictionary if not present
    exp_results[f'Training Exprience({idx})'][stream_acc_key] = result.get(stream_acc_key, None)
    print(idx)
    try:
        exp_results[f'Training Exprience({idx})'][f'Active units'] = model_incs[idx]
    except Exception:
        pass
    # Store the results in the final_training_results dictionary
    comprehensive_training_results[idx] = exp_results
print(comprehensive_training_results)


# training evaluation accuracy just after training (a trend between increasing number of active units and accuracy)
training_Results= {}
for result in comprehensive_training_results:
    print(result)
    training_Results[result] = {}
    try:
        training_Results[result]['Training Experience'] = {}
        training_Results[result]['Training Experience'][f'Accuracy'] = comprehensive_training_results[result][f'Training Exprience({result})'][f'Exp{result:03d}']
        training_Results[result]['Training Experience'][f'Active units'] = model_incs[result]
    except Exception:
        training_Results[result]['Training Experience'] = comprehensive_training_results[result][f'Training Exprience({result})']['Top1_Acc_Stream/eval_phase/test_stream/Task000']

print(training_Results)


# Stream Results
print('Stream Results')

stream_results = {}
for result in comprehensive_training_results:
    stream_results[f'Experience {result:03d}'] = comprehensive_training_results[result][f'Training Exprience({result})']['Top1_Acc_Stream/eval_phase/test_stream/Task000']
print(stream_results)


# testing resutls of final model
print('Testing final model')
print('Computing accuracy on test sets')
final_model_accuracy_results = comprehensive_training_results[29]['Training Exprience(29)']

#the following is the results i have collected, i need to combine them in a single yaml file
    # stream_results
    # training_Results
    # cumulative_training_results
    # comprehensive_training_results
    # final_model_accuracy_results
    # cumulative_test_results

all_results = {}
all_results['Stream Results'] = stream_results
all_results['Training Results'] = training_Results
all_results['Comprehensive Training Results'] = comprehensive_training_results
all_results['Final Model Accuracy Results'] = final_model_accuracy_results
all_results['Cumulative Test Results'] = cumulative_test_results
all_results['Cumulative Training Results'] = cumulative_training_results
# save as yaml file
import yaml
with open('Naive_results.yaml', 'w') as file:
    yaml.dump(all_results, file)