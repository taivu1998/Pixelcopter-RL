import numpy as np
from game import GameMDP


class Random(object):

    def __init__(self, game=None):
        self.game = game

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

        return scores