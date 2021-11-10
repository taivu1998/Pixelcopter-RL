import numpy as np
from game import GameMDP


class Random(object):

    def __init__(self):
        return

    def train(self):
        return

    def evaluate(self, epochs=1000):
        game = GameMDP()
        scores = []

        for i in range(epochs):
            score = 0
            game_over = False
            while not game_over:
                action = np.random.choice([0, 1])
                reward, next_state, game_over, point = game.step(action)
                score += point
            scores.append(score)
            game.reset()

        return scores