import numpy as np
from game import GameMDP
from collections import defaultdict
import random

class QLearning(object):

    def __init__(self, game=None, args=None):
        self.game = game
        self.args = args
        self.Q_values = defaultdict(float)
        self.actions = [0, 1]

    def train(self):
        scores = []
        for i in range(self.args['train_epochs']):
            samples, score = self.run_simulation(epsilon=self.args['epsilon'])
            scores.append(score)

            if self.args['order'] == 'backward':
                samples = reversed(samples)
            for (s, a, r, s_n) in samples:
                self.update(s, a, r, s_n, lr=self.args['lr'], discount_factor=self.args['discount_factor'])

    def evaluate(self, epochs=1000):
        scores = []
        for i in range(epochs):
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
        self.Q_values[(s, a)] = self.Q_values[(s, a)] + lr * (r + discount_factor * q_value_n - self.Q_values[(s, a)])

    def pick_action(self, s, epsilon=0):
        if random.random() < epsilon:
            return np.random.choice(self.actions), -1
        return self.policy(s)

    def policy(self, s):
        q_values = [self.Q_values[(s, a)] for a in self.actions]
        argmax_q_value = np.argmax(q_values)
        return self.actions[argmax_q_value], q_values[argmax_q_value]
