import numpy as np
from game import GameMDP
from collections import defaultdict, deque
import random
from tqdm import tqdm
from time import sleep

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExperienceReplayDataset(object):
    def __init__(self, max_length=4000):
        self.samples = deque(maxlen=max_length)

    def append(self, x):
        self.samples.append(x)

    def extend(self, x):
        self.samples.extend(x)

    def __len__(self):
        return len(self.samples)

    def batch(self, batch_size=64):
        samples = None
        if len(self.samples) >= batch_size:
            samples = random.sample(self.samples, batch_size)
            samples = zip(*samples)
        return samples


class NeuralNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[]):
        super(NeuralNet, self).__init__()
        layer_dims = [input_dim] + hidden_dims + [output_dim]
        self.num_layers = len(layer_dims) - 1
        self.layers = nn.ModuleList([])
        for i in range(self.num_layers):
            self.layers.extend([
                nn.Linear(layer_dims[i], layer_dims[i + 1]),
                nn.ReLU()
            ])

    def forward(self, x):
        # print(x)
        for layer in self.layers:
            x = layer(x)
        # x = self.layers(x)
        return x


class QLearningFuncApproxNN(object):

    def __init__(self, game=None, args=None):
        self.game = game
        self.args = args
        # self.Q_values = defaultdict(float)
        self.actions = [0, 1]
        self.experience_replay_dataset = None
        if self.args['experience_replay']:
            self.experience_replay_dataset = ExperienceReplayDataset(max_length=self.args['experience_replay_length'])
        self.model = NeuralNet(input_dim=self.game.get_state_legngth(), 
                               output_dim=len(self.actions), 
                               hidden_dims=self.args['experience_replay_hidden_dims'])
        # print(self.model)
        self.criterion = torch.nn.MSELoss()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args['lr'])

    def train(self):
        with tqdm(total=self.args['train_epochs'], desc='QLearning Train Progress Bar') as pbar:
            scores = []
            epsilon = self.args['epsilon']
            min_epsilon = 0.01
            epsilon_decay = self.args['epsilon_decay']

            for i in range(self.args['train_epochs']):
                pbar.update(1)

                samples, score = self.run_simulation(epsilon=epsilon)
                scores.append(score)

                if self.args['experience_replay']:
                    self.experience_replay_dataset.extend(samples)
                    batch = self.experience_replay_dataset.batch(self.args['batch_size'])
                else:
                    batch = zip(*samples)
                
                if batch:
                    self.update(batch, discount_factor=self.args['discount_factor'])

                # if self.args['order'] == 'backward':
                #     samples = reversed(samples)
                # for (s, a, r, s_n) in samples:
                #     self.update(s, a, r, s_n, lr=self.args['lr'], discount_factor=self.args['discount_factor'])

                # tried reward-based epsilon decay from https://aakash94.github.io/Reward-Based-Epsilon-Decay/,
                # but didn't perform that well.
                # will keep exponential epsilon decay each training epoch (epsilon *= 0.999) - very slow decay
                epsilon *= epsilon_decay
                epsilon = max(epsilon, min_epsilon)

    def evaluate(self, epochs=1000):
        with tqdm(total=epochs, desc='QLearning Evaluate Progress Bar') as pbar:
            scores = []
            for i in range(epochs):
                pbar.update(1)
                samples, score = self.run_simulation(epsilon=0)
                scores.append(score)
            return scores

    def run_simulation(self, epsilon=0):
        samples = []
        self.game.reset()
        game_over = False
        score = 0
        s = self.game.get_state()
        s_n = None

        while not game_over:
            a, _ = self.pick_action(s, epsilon=epsilon)
            r, s_n, game_over, point = self.game.step(a)
            s_t = torch.FloatTensor(s)
            a_t = torch.LongTensor([a])
            r_t = torch.FloatTensor([r])
            s_n_t = torch.FloatTensor(s_n)
            status_t = torch.FloatTensor([0]) if game_over else torch.FloatTensor([1])
            samples.append((s_t, a_t, r_t, s_n_t, status_t))
            s = s_n
            score += point
        return samples, score

    # def update(self, s, a, r, s_n, lr, discount_factor):
    #     a_n, q_value_n = self.policy(s_n)
    #     self.Q_values[(s, a)] = self.Q_values[(s, a)] + lr * (r + discount_factor * q_value_n - self.Q_values[(s, a)])
    
    def update(self, batch, discount_factor):
        s, a, r, s_n, status = batch
        s, a, r, s_n, status = torch.stack(s), torch.cat(a), torch.cat(r), torch.stack(s_n), torch.cat(status)
        s, a, r, s_n, status = s.to(self.device), a.to(self.device), r.to(self.device), s_n.to(self.device), status.to(self.device)
        q_curr_values = self.model(s)
        q_curr = torch.gather(q_curr_values, 1, a.unsqueeze(0)).squeeze()
        q_next_values = self.model(s_n)
        q_next_values_max, _ = torch.max(q_next_values, 1)
        q_star_est = r + discount_factor * status * q_next_values_max
        loss = self.criterion(q_curr, q_star_est)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 

    def pick_action(self, s, epsilon=0):
        if random.random() < epsilon:
            return np.random.choice(self.actions), -1
        return self.policy(s)

    def policy(self, s):
        q_values = [self.get_Q_value(s, a) for a in self.actions]
        argmax_q_value = np.argmax(q_values)
        return self.actions[argmax_q_value], q_values[argmax_q_value]
    
    def get_Q_value(self, s, a):
        with torch.no_grad():
            s_t = torch.FloatTensor(s).unsqueeze(0)
            q_values = self.model(s_t).squeeze().numpy()
            q = q_values[a]
            return q
