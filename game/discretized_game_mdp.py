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
        #print('this is state BEFORE discrete', state)
        state = tuple([self.discretize(state[feature]) for feature in self.features])
        # ############### converts decimal for player_vec to int #######################
        # states = []
        # for i in range(len(self.features)):
        #     feature = self.features[i]
        #     #print('this is feature', feature)
        #     if feature == "player_vel":
        #         #print('this is player_vel', feature)
        #         states.append(int(state[feature] * 10))
        #     else:
        #         states.append(self.discretize(state[feature]))
        # return tuple(states)
        # ###############################################################################
        #print('this is state AFTER discrete', states)
        return state

    def discretize(self, value):
        return int(self.grid_size * np.floor(value / self.grid_size))