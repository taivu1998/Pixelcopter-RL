import numpy as np
from game import GameMDP
from collections import defaultdict
import random
from tqdm import tqdm
from time import sleep
import pickle


np.random.seed(2) # can delete this after testing hyper parameters

class SarsaLambda(object):

    def __init__(self, game=None, args=None):
        self.game = game
        self.args = args
        self.Q_values = defaultdict(float)
        # N is the trace (tracks state-action pairs)
        self.N = defaultdict(float)
        self.actions = [0, 1]
        # most recent experience tuple (s,a,r) is used in updateSarsaLambda
        self.recent_tuple = None
        # trace decay rate (common to use between 0 and 1)
        self.trace_decay = self.args['trace_decay']
        self.discount_factor = self.args['discount_factor']
        self.lr = self.args['lr']
        self.batch_evals = defaultdict(float)


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
        with open('sarsaLambda_evals.pkl', 'wb') as f:
            pickle.dump(self.batch_evals, f)

    def train(self):
        scores = []
        epsilon = self.args['epsilon']
        min_epsilon = 0.001
        epsilon_decay = self.args['epsilon_decay']

        with tqdm(total=self.args['train_epochs'], desc= 'sarsaLambda Train Progress Bar') as pbar:
            for i in range(self.args['train_epochs']):
                pbar.update(1)
                # performs 100 evaluations every 1000 epochs
                if i > 0 and i % 1000 == 0:
                    self.save_batch_evals(i)

                samples, score = self.run_simulation(epsilon=epsilon)
                scores.append(score)

                if self.args['order'] == 'backward':
                    samples = reversed(samples)
                for (s, a, r, s_n) in samples:
                    self.updateSarsaLambda(s, a, r, s_n, lr=self.lr,
                                           discount_factor=self.discount_factor,
                                           trace_decay=self.trace_decay)
                epsilon *= epsilon_decay
                epsilon = max(epsilon, min_epsilon)

        # saves batch evaluations (mean, std, min, max) to pickle file
        self.write_evals_to_file()

    def evaluate(self, epochs=1000):
        scores = []
        with tqdm(total=epochs, desc='sarsaLambda Evaluate Progress Bar') as pbar:
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

    def updateSarsaLambda(self, s, a, r, s_n, lr, discount_factor, trace_decay):
        if self.recent_tuple is not None:
            # update N(state, action) count
            self.N[(self.recent_tuple[0], self.recent_tuple[1])] += 1
            # delta is decay rate
            delta = self.recent_tuple[2] + (discount_factor * self.Q_values[(s, a)]) - \
                    self.Q_values[(self.recent_tuple[0], self.recent_tuple[1])]
            for state_action in self.Q_values.keys():
                    # print('this is a', a)
                    self.Q_values[state_action] += (lr * delta * self.N[state_action])
                    self.N[state_action] *= (discount_factor * trace_decay)
        else:
            # sets all keys to 0
            self.N.fromkeys(self.N, 0.0)
        # update most recent tuple
        self.recent_tuple = (s, a, r)

    def pick_action(self, s, epsilon=0):
        if random.random() < epsilon:
            return np.random.choice(self.actions), -1
        return self.policy(s)

    def policy(self, s):
        q_values = [self.Q_values[(s, a)] for a in self.actions]
        argmax_q_value = np.argmax(q_values)
        return self.actions[argmax_q_value], q_values[argmax_q_value]
