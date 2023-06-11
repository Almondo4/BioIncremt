################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 01-12-2020                                                             #
# Author(s): Andrea Cossu                                                      #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
This example trains a Multi-head model on Split MNIST with Elastich Weight
Consolidation. Each experience has a different task label, which is used at test
time to select the appropriate head.
"""

import argparse
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from avalanche.benchmarks.classic import SplitMNIST
from avalanche.models import MTSimpleMLP, IncrementalClassifier
from avalanche.training.supervised import EWC
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin



    # Config
device = 'gpu' if torch.cuda.is_available() else 'cpu'
    # model
# model = IncrementalClassifier(in_features=28, initial_out_features=2,masking=False)
model = MTSimpleMLP()

    # CL Benchmark Creation
benchmark = SplitMNIST(n_experiences=5, return_task_id=True)
train_stream = benchmark.train_stream
test_stream = benchmark.test_stream

    # Prepare for training & testing
optimizer = Adam(model.parameters(), lr=0.01)
criterion = CrossEntropyLoss()

    # choose some metrics and evaluation method
interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=False, epoch=True, experience=True, stream=True
        ),
        forgetting_metrics(experience=True),
        loggers=[interactive_logger],
    )

    # Choose a CL strategy
strategy = EWC(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=128,
        train_epochs=3,
        eval_mb_size=128,
        device=device,
        evaluator=eval_plugin,
        ewc_lambda=0.4,
    )

    # train and test loop

results = []
model_incs = []
classes_exp = []

for train_task in train_stream:

        print("Start of experience: ", train_task.current_experience)
        print("Current Classes: ", train_task.classes_in_this_experience)
        targets = train_task.dataset.targets
        print(set(targets))
        model_incs.append(model.classifier)
        classes_exp.append(train_task.classes_in_this_experience)

        strategy.train(train_task)
        strategy.eval(test_stream)



