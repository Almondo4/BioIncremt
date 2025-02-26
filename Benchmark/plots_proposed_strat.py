import yaml
import pandas as pd
import matplotlib.pyplot as plt

# Load data from YAML file
with open('proposed_start.yaml', 'r') as file:
    all_results = yaml.safe_load(file)

# Extract necessary data
stream_results = all_results['Stream Results']
cumulative_test_results = all_results['Cumulative Test Results']
training_results = all_results['Training Results']
comprehensive_training_results = all_results['Comprehensive Training Results']
cumulative_training_results = all_results['Cumulative Training Results']

# 1- Stream Accuracy Over Experiences
df_stream_results = pd.DataFrame.from_dict(stream_results, orient='index', columns=['Stream Accuracy'])
plt.figure(figsize=(10, 6))
df_stream_results.plot(kind='line', marker='o')
plt.title('Stream Accuracy Over Experiences')
plt.xlabel('Experience')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

# 2- Cumulative Test Accuracy
df_cumulative_test_results = pd.DataFrame.from_dict(cumulative_test_results, orient='index', columns=['Cumulative Test Accuracy'])
plt.figure(figsize=(10, 6))
df_cumulative_test_results.plot(kind='bar', legend=False)
plt.title('Cumulative Test Accuracy')
plt.xlabel('Cumulative Set')
plt.ylabel('Accuracy')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# 3- Active Units vs. Training Accuracy
active_units = []
training_accuracy = []

for exp_id, exp_data in training_results.items():
    active_units.append(exp_data[exp_id][f'Training Experience']['Active units'])
    training_accuracy.append(exp_data[exp_id]['Training Experience']['Accuracy'])

plt.figure(figsize=(10, 6))
plt.scatter(active_units, training_accuracy, color='b', marker='o')
plt.title('Active Units vs. Training Accuracy')
plt.xlabel('Active Units')
plt.ylabel('Training Accuracy')
plt.grid()
plt.tight_layout()
plt.show()

# 4- Heatmap for Comprehensive Training Results
comprehensive_data = []

for exp_id, exp_data in comprehensive_training_results.items():
    exp_results = exp_data[f'Training Experience ({exp_id})']
    comprehensive_data.append(list(exp_results.values()))

comprehensive_df = pd.DataFrame(comprehensive_data, columns=list(exp_results.keys()))
plt.figure(figsize=(12, 8))
plt.imshow(comprehensive_df, cmap='YlGnBu', aspect='auto', origin='lower')
plt.colorbar(label='Accuracy')
plt.title('Comprehensive Training Results')
plt.xlabel('Experiences')
plt.ylabel('Training Experiments')
plt.xticks(range(len(comprehensive_df.columns)), comprehensive_df.columns, rotation=45)
plt.tight_layout()
plt.show()

# 5- Cumulative Training Results
df_cumulative_training_results = pd.DataFrame.from_dict(cumulative_training_results, orient='index', columns=['Cumulative Training BWT'])
plt.figure(figsize=(10, 6))
df_cumulative_training_results.plot(kind='line', marker='o')
plt.title('Cumulative Training Backward Transfer')
plt.xlabel('Experience')
plt.ylabel('BWT')
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()
