import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import datetime


class ModelBase(torch.nn.Module):
    def __init__(self, project):
        super().__init__()

        self._project_name = project

    def forward(self, x):
        pass

    def save(self):
        model_folder = 'model'
        project_folder = os.path.join(model_folder, self._project_name)

        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        if not os.path.exists(project_folder):
            os.mkdir(project_folder)

        file_name = self._project_name + '_' + \
            datetime.datetime.now().strftime('%H-%M-%S %d-%m-%Y') + '.pth'

        path = os.path.join(project_folder, file_name)
        torch.save(self.state_dict(), path)


class LinearFlatten(ModelBase):
    def __init__(self, project, input_size, hidden_size1, hidden_size2):
        super().__init__(project)

        self.nn = nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(input_size, hidden_size1),
                                torch.nn.ReLU(),
                                torch.nn.Linear(hidden_size1, hidden_size2),
                                torch.nn.ReLU(),
                                torch.nn.Linear(hidden_size2, 3))

    def forward(self, x):
        return self.nn(x)


class NatureCnn(ModelBase):
    def __init__(self, project, frame, final_layer=512):
        super().__init__(project)

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            n_flatten = self.cnn(frame).shape[1]

        self.nn = nn.Sequential(self.cnn, nn.Linear(
            n_flatten, final_layer), nn.ReLU(), nn.Linear(final_layer, 3))

    def forward(self, x):
        return self.nn(x)


class SingleLinear(ModelBase):
    def __init__(self, project, imput_size, hidden_size, n_actions):
        super().__init__(project)

        self.linear1 = torch.nn.Linear(imput_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = F.relu(self.linear1(x))

        return self.linear2(x)


class DoubleLinear(ModelBase):
    def __init__(self, project, input_size, hidden_size1, hidden_size2, n_actions):
        super().__init__(project)

        self.linear1 = torch.nn.Linear(input_size, hidden_size1)
        self.linear2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = torch.nn.Linear(hidden_size2, n_actions)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        return self.linear3(x)


class Conv1(ModelBase):
    def __init__(self, project):
        super().__init__(project)

        self.project = project

        self.l1 = nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=3, stride=3),
            torch.nn.BatchNorm2d(8),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.ReLU()
        )

        self.l2 = nn.Sequential(
            torch.nn.Conv2d(8, 16, kernel_size=3, stride=3),
            torch.nn.BatchNorm2d(16),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.ReLU()
        )

        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(8 * 5 * 6, 256)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x):
        x = self.l1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
