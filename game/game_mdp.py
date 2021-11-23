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
        self.features = sorted([
            'player_y',
            'player_vel',
            'player_dist_to_ceil',
            'player_dist_to_floor',
            'next_gate_dist_to_player',
            'next_gate_block_top',
            'next_gate_block_bottom'
        ])

    def get_state(self):
        state = self.game_env.getGameState()
        state = tuple([state[feature] for feature in self.features])
        return state

    def step(self, action):
        raw_reward = self.game_env.act(self.actions[action])
        game_over = self.game_env.game_over()
        next_state = self.get_state()
        point = 1 if raw_reward >= 1 else 0
        if game_over:
            reward = -2000
        elif raw_reward >= 1:
            reward = 3
        else:
            reward = 1
        return reward, next_state, game_over, point

    def reset(self):
        self.game_env.reset_game()