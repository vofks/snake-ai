import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
import torch.optim as optim
import torch.nn.functional as F
import os
import datetime
import numpy as np


class Linear1(nn.Module):
    def __init__(self, project, hidden_size):
        super().__init__()

        self.project = project

        self.linear1 = nn.Linear(13, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 3)

    def forward(self, x):
        x = F.relu(self.linear1(x))

        return self.linear2(x)

    def save(self):
        model_folder = 'model'
        project_folder = os.path.join(model_folder, self.project)

        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        if not os.path.exists(project_folder):
            os.mkdir(project_folder)

        file_name = self.project + '_' + \
            datetime.datetime.now().strftime('%H-%M-%S %d-%m-%Y') + '.pth'

        path = os.path.join(project_folder, file_name)
        torch.save(self.state_dict(), path)


class Linear2(nn.Module):
    def __init__(self, project, hidden_size1, hidden_size2):
        super().__init__()

        self.project = project

        self.linear1 = nn.Linear(13, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, 3)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        return self.linear3(x)

    def save(self):
        model_folder = 'model'
        project_folder = os.path.join(model_folder, self.project)

        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        if not os.path.exists(project_folder):
            os.mkdir(project_folder)

        file_name = self.project + '_' + \
            datetime.datetime.now().strftime('%H-%M-%S %d-%m-%Y') + '.pth'

        path = os.path.join(project_folder, file_name)
        torch.save(self.state_dict(), path)


class Conv1(nn.Module):
    def __init__(self, project):
        super().__init__()

        self.project = project

        self.l1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(864, 256)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x):
        x = self.l1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def save(self):
        model_folder = 'model'
        project_folder = os.path.join(model_folder, self.project)

        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        if not os.path.exists(project_folder):
            os.mkdir(project_folder)

        file_name = self.project + '_' + \
            datetime.datetime.now().strftime('%H-%M-%S %d-%m-%Y') + '.pth'

        path = os.path.join(project_folder, file_name)
        torch.save(self.state_dict(), path)


class QTrainer:
    def __init__(self, model, lr, gamma, device):
        self.device = device
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, new_state, done):
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)
        state = torch.tensor(
            np.array(state), dtype=torch.float, device=self.device)
        new_state = torch.tensor(
            np.array(new_state), dtype=torch.float, device=self.device)

        if len(state.shape) == 3:
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            new_state = torch.unsqueeze(new_state, 0)
            done = (done,)

        prediction = self.model(state)
        target = prediction.clone()

        for i in range(len(done)):
            q_new = reward[i]
            if not done[i]:
                q_new += self.gamma * \
                    torch.max(self.model(new_state[i].unsqueeze(0)))
            target[i][torch.argmax(action[i]).item()] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()

        self.optimizer.step()
