from ple import PLE
from ple.games.pixelcopter import Pixelcopter
import gym
import numpy as np

class GameMDP(object):
    
    def __init__(self):
        self.game_env = PLE(Pixelcopter(), fps=30, display_screen=False)
        self.game_env.init()
        self.actions = self.game_env.getActionSet()
        self.action_set = range(len(self.actions))

    def get_state(self):
        state = self.game_env.getGameState()
        return state

    def step(self, action):
        reward = self.game_env.act(self.actions[action])
        game_over = self.game_env.game_over()
        next_state = self.get_state()
        point = 1 if reward >= 1 else 0
        return reward, next_state, game_over, point

    def reset(self):
        self.game_env.reset_game()