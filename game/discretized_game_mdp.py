from ple import PLE
from ple.games.pixelcopter import Pixelcopter
import gym
import numpy as np
from .game_mdp import GameMDP


class DiscretizedGameMDP(GameMDP):
    
    def __init__(self, grid_size=5):
        super().__init__()
        self.grid_size = grid_size

    def get_state(self):
        state = self.game_env.getGameState()
        state = tuple([self.discretize(state[feature]) for feature in self.features])
        return state
    
    def discretize(self, value):
        return int(self.grid_size * np.floor(value / self.grid_size))