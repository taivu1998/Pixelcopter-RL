import numpy as np
from game import GameMDP
from collections import defaultdict
import random
from tqdm import tqdm
from time import sleep
import pickle

np.random.seed(2) # can delete this after testing hyper parameters

class QLearning(object):

    def __init__(self, game=None, args=None):
        self.game = game
        self.args = args
        self.Q_values = defaultdict(float)
        self.actions = [0, 1]
        self.batch_evals = defaultdict(tuple)

    def print_parameters(self):
        print()
        parameters = [str(args)+': ' + str(self.args[str(args)]) for args in self.args]
        for param in parameters:
            print(param)
        print()

    def save_batch_evals(self, batch):
        scores = self.evaluate(epochs=100)
        batch_min = np.amin(scores)
        batch_max = np.amax(scores)
        batch_mean = np.mean(scores)
        batch_std = np.std(scores)
        results = (batch_mean, batch_std, batch_min, batch_max)
        self.batch_evals[batch] = results

        # print("Batch Min: {}".format(np.amin(scores)))
        # print("Batch Max: {}".format(np.amax(scores)))
        # print("Batch Mean: {}".format(np.mean(scores)))
        # print("Batch Std: {}".format(np.std(scores)))

    def write_evals_to_file(self):
        with open('qlearning_evals.pkl', 'wb') as f:
            pickle.dump(self.batch_evals, f)

    def train(self):
        with tqdm(total=self.args['train_epochs'], desc='QLearning Train Progress Bar') as pbar:
            epsilon = self.args['epsilon']
            min_epsilon = 0.001
            epsilon_decay = self.args['epsilon_decay']

            scores = []

            for i in range(self.args['train_epochs']):
                pbar.update(1)
                # performs 100 evaluations every 1000 epochs
                if i > 0 and i % 1000 == 0:
                    self.save_batch_evals(i)
                    # print("this is self.batch_evals[i]", self.batch_evals[i])

                samples, score = self.run_simulation(epsilon=epsilon)
                scores.append(score)

                if self.args['order'] == 'backward':
                    samples = reversed(samples)
                for (s, a, r, s_n) in samples:
                    self.update(s, a, r, s_n, lr=self.args['lr'], discount_factor=self.args['discount_factor'])

                # tried reward-based epsilon decay from https://aakash94.github.io/Reward-Based-Epsilon-Decay/,
                # but didn't perform that well.
                # will keep exponential epsilon decay each training epoch (epsilon *= 0.999) - very slow decay
                epsilon *= epsilon_decay
                epsilon = max(epsilon, min_epsilon)

        # saves batch evaluations (mean, std, min, max) to pickle file
        self.write_evals_to_file()

    def evaluate(self, epochs=1000):
        with tqdm(total=epochs, desc='QLearning Evaluate Progress Bar') as pbar:
            scores = []
            for i in range(epochs):
                pbar.update(1)
                samples, score = self.run_simulation(epsilon=0)
                scores.append(score)
            self.print_parameters()
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
