import torch
from transition import Transition


class Trainer:
    def __init__(self, model, target_model, device, criterion=torch.nn.MSELoss(), learning_rate=1e-3, gamma=0.99):
        self._device = device
        self._model = model
        self._target_model = target_model
        self._learning_rate = learning_rate
        self._gamma = gamma
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=learning_rate)
        self._criterion = criterion

    def step(self, batch):
        batch = Transition(*zip(*batch))

        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        next_states = torch.cat(batch.next_state)
        dones = torch.cat(batch.done)

        q_values = self._model(states)
        prediction = q_values.gather(1, actions)

        next_q_values = self._target_model(next_states).max(1, keepdim=True)[0]

        ''' 
            Better way to deal with final states
            Thanks to https://www.youtube.com/watch?v=NP8pXZdU-5U
        '''
        target = rewards + self._gamma * next_q_values * (1 - dones)

        self._optimizer.zero_grad()
        loss = self._criterion(prediction, target)
        loss.backward()

        self._optimizer.step()
