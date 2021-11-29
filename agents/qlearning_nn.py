import numpy as np
from game import GameMDP
from collections import defaultdict
import random
from tqdm import tqdm
from time import sleep

class QLearning_NN(object):

    def __init__(self, game=None, args=None):
        self.game = game
        self.args = args
        self.Q_values = {}
        self.actions = [0, 1]

    def train(self):
        with tqdm(total=self.args['train_epochs'], desc='QLearning Train Progress Bar') as pbar:
            scores = []
            for i in range(self.args['train_epochs']):
                pbar.update(1)
                samples, score = self.run_simulation(epsilon=self.args['epsilon'])
                scores.append(score)

                if self.args['order'] == 'backward':
                    samples = reversed(samples)
                for (s, a, r, s_n) in samples:
                    self.update(s, a, r, s_n, lr=self.args['lr'], discount_factor=self.args['discount_factor'])

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
            samples.append((s, a, r, s_n))
            s = s_n
            score += point
        return samples, score

    def update(self, s, a, r, s_n, lr, discount_factor):
        a_n, q_value_n = self.policy(s_n)
        approx_s = s
        approx_s, valid_s = self.approximate_s(s)
        if s not in self.Q_values.keys():
            self.Q_values[s] = {}
        if valid_s:
            self.Q_values[s][a] = self.Q_values[approx_s][a] + lr * (r + discount_factor * q_value_n - self.Q_values[approx_s][a])
        else:
            self.Q_values[s][a] = 0 + lr * (r + discount_factor * q_value_n - 0)

    def pick_action(self, s, epsilon=0):
        if random.random() < epsilon:
            return np.random.choice(self.actions), -1
        return self.policy(s)

    def policy(self, s):
        approx_s = s
        approx_s, valid_s = self.approximate_s(s)
        if valid_s:
            q_values = [self.Q_values[approx_s][a] for a in self.actions]
            argmax_q_value = np.argmax(q_values)
            return self.actions[argmax_q_value], q_values[argmax_q_value]
        else:
            return np.random.choice(self.actions), 0

    def approximate_s(self, s):
        if s in self.Q_values and all(a in self.Q_values[s].keys() for a in self.actions):
            return s, 1
        approx_s = s
        min_d = float('inf')
        for s1 in self.Q_values:
            if all(a in self.Q_values[s1].keys() for a in self.actions):
                d = dist(s,s1)
                if d < min_d:
                    approx_s = s1
                    min_d = d
        if approx_s == s:
            return approx_s, 0
        else:
            return approx_s, 1

def dist(s, s1):
    d = 0.0
    for _, (a,b) in enumerate(zip(s,s1)):
        d += (a - b)**2
    return d

