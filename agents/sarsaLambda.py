import numpy as np
from game import GameMDP
from collections import defaultdict
import random
from tqdm import tqdm
from time import sleep

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
        self.trace_decay = None

    def train(self):
        scores = []

        with tqdm(total=self.args['train_epochs'], desc= 'sarsaLambda Train Progress Bar') as pbar:
            for i in range(self.args['train_epochs']):
                pbar.update(1)
                samples, score = self.run_simulation(epsilon=self.args['epsilon'])
                scores.append(score)

                if self.args['order'] == 'backward':
                    samples = reversed(samples)
                for (s, a, r, s_n) in samples:
                    self.updateSarsaLambda(s, a, r, s_n, lr=self.args['lr'],
                                           discount_factor=self.args['discount_factor'],
                                           trace_decay=self.args['trace_decay'])

    def evaluate(self, epochs=1000):
        scores = []
        with tqdm(total=epochs, desc='sarsaLambda Evaluate Progress Bar') as pbar:
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
