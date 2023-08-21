from avalanche.benchmarks import CLExperience
from avalanche.models import DynamicModule
import torch
from torch import nn


class Incremental1DCNNClassifier(DynamicModule):
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
