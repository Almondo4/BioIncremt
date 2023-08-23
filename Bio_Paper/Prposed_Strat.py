import pandas as pd
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, timing_metrics, forgetting_metrics, \
    cpu_usage_metrics, disk_usage_metrics, EpochCPUUsage, EpochTime
from avalanche.logging import TensorboardLogger, TextLogger, InteractiveLogger
from avalanche.training import EWC, ReservoirSamplingBuffer
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin
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
cumulative_training_results = []
all_results = {}

cumulative_test_Experiences = [4,9,14,19,24,29]

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
    if exp_id  in cumulative_test_Experiences:
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



# from the results list create a dict with key as the experience and valu as a sub dict where each key is an experience id extracted from Top1_acc_Exp and the value is  the value of that element. when done from Experiences also append the key and value of 'Top1_Acc_Stream/eval_phase/test_stream/Task000' as Stream Accuracy . the following is a sample of the results item in the results list:

comprehensive_training_results = {}
for idx, result in enumerate(results):
    exp_results = {}  # Initialize the dictionary for this iteration
    stream_acc_key = 'Top1_Acc_Stream/eval_phase/test_stream/Task000'
    for key, value in result.items():
        if 'Top1_Acc_Exp' in key:
            exp_id = key.split('/')[-1]
            if f'Training Experience({idx})' not in exp_results:
                exp_results[f'Training Experience({idx})'] = {}  # Initialize the sub-dictionary if not present
            exp_results[f'Training Experience({idx})'][exp_id] = value
    # Append the stream accuracy to the experience results
    if f'Training Experience({idx})' not in exp_results:
        exp_results[f'Training Experience({idx})'] = {}  # Initialize the sub-dictionary if not present
    exp_results[f'Training Experience({idx})'][stream_acc_key] = result.get(stream_acc_key, None)
    print(idx)
    try:
        exp_results[f'Training Experience({idx})'][f'Active units'] = model_incs[idx]
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
        training_Results[result]['Training Experience'][f'Accuracy'] = comprehensive_training_results[result][f'Training Experience({result})'][f'Exp{result:03d}']
        training_Results[result]['Training Experience'][f'Active units'] = model_incs[result]
    except Exception:
        training_Results[result]['Training Experience'] = comprehensive_training_results[result][f'Training Experience({result})']['Top1_Acc_Stream/eval_phase/test_stream/Task000']

print(training_Results)


# Stream Results
print('Stream Results')

stream_results = {}
for result in comprehensive_training_results:
    stream_results[f'Experience {result:03d}'] = comprehensive_training_results[result][f'Training Experience({result})']['Top1_Acc_Stream/eval_phase/test_stream/Task000']
print(stream_results)


# testing resutls of final model
print('Testing final model')
print('Computing accuracy on test sets')
final_model_accuracy_results = comprehensive_training_results[29]['Training Experience(29)']

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
with open('naive.yaml', 'w') as file:
    yaml.dump(all_results, file)

# graphs to make :
# 1- Stream Accuracy Over Experiences (using stream results) : to show the trend of traning accuracy over experiences
# 2- Cumulative Test Accuracy (using cumulative test results): to show the final models forgetting/ remembering performance
# 3- Active Units vs. Training Accuracy (using traning results): to show the trend of accuracy over active units and show that tmodel is actually incremental and efficient learning and adapting
# 4- Heatmap for Comprehensive Training Results (using comprehesive traning results): to show the trend of accuracy over experiences and active units
# 5- cumulative traning results (using cumulative traning results): to show the trend of BWT over the training experiences



# Create a new Excel workbook
combined_excel_writer = pd.ExcelWriter('Naive_combined_results.xlsx', engine='xlsxwriter')


## Stream Results
# Create a DataFrame from the dictionary
df = pd.DataFrame(list(all_results['Stream Results'].items()), columns=['Experience', 'Result'])
# Create a Pandas Excel writer using XlsxWriter as the engine.
excel_writer = pd.ExcelWriter('Starting_Stream_Results.xlsx', engine='xlsxwriter')
# Convert the DataFrame to an XlsxWriter Excel object.
df.to_excel(combined_excel_writer, sheet_name='Stream Results', index=False)


## Training Results
# Create a list of dictionaries for DataFrame construction
training_results_list = [
    {'Index': idx, 'Accuracy': data['Training Experience']['Accuracy'], 'Active Units': data['Training Experience']['Active units']}
    for idx, data in training_Results.items()
]
# Create a DataFrame from the list of dictionaries
df = pd.DataFrame(training_results_list)

# Create a Pandas Excel writer using XlsxWriter as the engine.
excel_writer = pd.ExcelWriter('Training_Results.xlsx', engine='xlsxwriter')

# Convert the DataFrame to an XlsxWriter Excel object.
df.to_excel(combined_excel_writer, sheet_name='Training Results', index=False)


## Comprehensive Training Results
# Create a list of dictionaries for DataFrame construction
comprehensive_results_list = []
for idx, data in comprehensive_training_results.items():
    entry_dict = {'Index': idx}
    entry_dict.update(data[f'Training Experience({idx})'])
    comprehensive_results_list.append(entry_dict)

# Create a DataFrame from the list of dictionaries
df = pd.DataFrame(comprehensive_results_list)

# Create a Pandas Excel writer using XlsxWriter as the engine.
excel_writer = pd.ExcelWriter('Comprehensive_Results.xlsx', engine='xlsxwriter')

# Convert the DataFrame to an XlsxWriter Excel object.
df.to_excel(combined_excel_writer, sheet_name='Comprehensive Training Results', index=False)


## Final Model Accuracy Results
# Create a DataFrame from the dictionary
df = pd.DataFrame(final_model_accuracy_results.items(), columns=['Experience', 'Accuracy'])

# Create an Excel writer
excel_writer = pd.ExcelWriter('final_model_results.xlsx', engine='xlsxwriter')

# Write the DataFrame to an Excel sheet
df.to_excel(combined_excel_writer, sheet_name='Final Model Accuracy Results', index=False)


## Cumulative Test Results
# Create a DataFrame from the dictionary
cumulative_df = pd.DataFrame(cumulative_test_results.items(), columns=['Cumulative Set', 'Accuracy'])

# Create an Excel writer
cumulative_excel_writer = pd.ExcelWriter('cumulative_test_results.xlsx', engine='xlsxwriter')

# Write the DataFrame to an Excel sheet
cumulative_df.to_excel(combined_excel_writer, sheet_name='Cumulative Test Results', index=False)


## Cumulative Training Results
import re
# Extract experiment number and active units from dictionary keys
pattern = r'Exp_(\d+)\((\d+) acs\)'

# Create lists to store extracted information
exp_numbers = []
active_units = []
accuracies = []

# Extract information from the dictionary keys
for result in cumulative_training_results:
    for key in result.keys():
        match = re.match(pattern, key)
        if match:
            exp_number = int(match.group(1))
            acs_number = int(match.group(2))
            exp_numbers.append(exp_number)
            active_units.append(acs_number)
            accuracies.append(result[key])

# Create a DataFrame from the extracted information
cumulative_training_df = pd.DataFrame({
    'Exp Number': exp_numbers,
    'Active Units': active_units,
    'Accuracy': accuracies
})

# Create an Excel writer
cumulative_training_excel_writer = pd.ExcelWriter('cumulative_training_results.xlsx', engine='xlsxwriter')

# Write the DataFrame to an Excel sheet
# Convert the DataFrame to an XlsxWriter Excel object for the combined file.
cumulative_training_df.to_excel(combined_excel_writer, sheet_name='Cumulative Training Results', index=False)

# Save the combined Excel file.
combined_excel_writer.save()

print("Combined Excel file created successfully.")


