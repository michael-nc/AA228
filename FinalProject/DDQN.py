import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from NN import Network

class DDQNAgent():

    def __init__(self, input_channel, input_size, action_size, lr=0.00025, gamma=0.99, batch_size=32, experience_size=10000, target_replace_count=100, epsilon=1.0, epsilon_final=0.1, epsilon_decrease=2e-6, device="cpu"):

        self.input_channel = input_channel
        self.input_size = input_size
        self.action_size = action_size
        self.lr = lr
        self.model_name = "DDQN"

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_replace_count = target_replace_count
        self.epsilon = epsilon
        self.epsilon_final = epsilon_final
        self.epsilon_decrease = epsilon_decrease
        self.learn_counter = 0
        self.device = device

        self.experience_counter = 0
        self.experience_size = experience_size
        self.experience_replay = deque(maxlen=self.experience_size)

        self.DDQN = Network(self.input_channel, self.input_size, self.action_size).to(self.device)
        self.target_DDQN = Network(self.input_channel, self.input_size, self.action_size).to(self.device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.DDQN.parameters(), lr=self.lr)
        print("Initializing {self.model_name}!")

    def get_action(self, state):
        probability = np.random.random()

        if probability < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            state = torch.tensor(state).to(self.device).unsqueeze_(0)
            action = torch.argmax(self.DDQN(state)).item()

            return action

    def get_action_eval(self, state):
        state = torch.tensor(state).to(self.devcie)
        action = torch.argmax(self.DDQN(state)).item()

        return action

    def calculate_error(self, actions, states, next_states, rewards, dones):

        q_current = self.DDQN(states)[np.arange(self.batch_size), actions]

        q_current_argmax = self.target_DDQN.forward_max(next_states)
        q_next = self.target_DDQN(next_states)
        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next[np.arange(self.batch_size), q_current_argmax]

        return q_target, q_current

    def learn(self):
        
        if self.experience_counter > self.experience_size:

            # replace target network if replace count has been reached
            if self.learn_counter % self.target_replace_count == 0:
                self.target_DDQN.load_state_dict(self.DDQN.state_dict())

            actions, states, next_states, rewards, dones = self.sample_from_replay()
            q_target, q_current = self.calculate_error(actions, states, next_states, rewards, dones)
            self.back_propagation(q_target, q_current)
            self.decrease_epsilon()

    def store_experience(self, action, state, next_state, reward, done):

        experience = [action, state, next_state, reward, done]
        self.experience_replay.append(experience)
        self.experience_counter += 1

    def sample_from_replay(self):
        experiences = random.sample(self.experience_replay, self.batch_size)

        actions = torch.tensor([experience[0] for experience in experiences], dtype=torch.long).to(self.device)

        states = np.array([experience[1] for experience in experiences])
        states = torch.tensor(states, dtype=torch.float).to(self.device)

        next_states = np.array([experience[2] for experience in experiences])
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)

        rewards = torch.tensor([experience[3] for experience in experiences], dtype=torch.float).to(self.device)
        dones = torch.tensor([experience[4] for experience in experiences], dtype=torch.bool).to(self.device)

        return actions, states, next_states, rewards, dones

    def decrease_epsilon(self):
        if self.epsilon > self.epsilon_final:
            self.epsilon -= self.epsilon_decrease
        else:
            self.epsilon = self.epsilon_final

    def back_propagation(self, q_target, q_current):
        
        loss = self.loss_fn(q_target, q_current)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learn_counter += 1

    def save_model(self, base="."):

        torch.save(self.DDQN.state_dict(), f"{base}/model/{self.model_name}.pth")
        print(f"Save PyTorch {self.model_name} Model")


    def load_model(self, path):
        self.DDQN.load_state_dict(torch.load(path))
        print(f"Load PyTorch {self.model_name} Model")