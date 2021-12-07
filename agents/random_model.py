import numpy as np
from game import GameMDP
from collections import defaultdict
import pickle


np.random.seed(2) # can delete this after testing hyper parameters

class Random(object):

    def __init__(self, game=None):
        self.game = game
        self.batch_evals = defaultdict(tuple)

    def write_evals_to_file(self):
        with open('random_model_evals.pkl', 'wb') as f:
            pickle.dump(self.batch_evals, f)

    def train(self):
        return

    def evaluate(self, epochs=1000):
        scores = []

        for i in range(epochs):
            score = 0
            game_over = False
            while not game_over:
                action = np.random.choice([0, 1])
                reward, next_state, game_over, point = self.game.step(action)
                score += point
            scores.append(score)
            self.game.reset()
            # performs 100 evaluations every 1000 epochs
            if i > 0 and i % 1000 == 0:
                batch_min = np.amin(scores)
                batch_max = np.amax(scores)
                batch_mean = np.mean(scores)
                batch_std = np.std(scores)
                results = (batch_mean, batch_std, batch_min, batch_max)
                self.batch_evals[i] = results

        # saves batch evaluations (mean, std, min, max) to pickle file
        self.write_evals_to_file()

        return scores