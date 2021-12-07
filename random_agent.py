from ple import PLE
from ple.games.pixelcopter import Pixelcopter
# from ple import PLE
import gym
# from gym.wrappers import Monitor
# import gym_ple
# from gym_ple import PLEEnv
import numpy as np
from tqdm import tqdm
from time import sleep


class GameMDP(gym.Env):
    ''' Game environment for SARSA and Q-Learning. '''
    
    def __init__(self):
        '''
        Initializes the environment.
        
        Args:
            env (PLEEnv): A Pygame environment.
            rounding (int): The level of discretization.
        '''
        # super().__init__(gym.make('PixelCopter-v0'))
        # # super().__init__(gym.make('FlappyBird-v0'))
        # self.env = Monitor(self.env, directory = "./out", force = True)
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
        return reward, next_state, game_over

    def reset(self):
        self.game_env.reset_game()


game = GameMDP()

scores = []
with tqdm(total=10000, desc='Random Evaluation Progress Bar') as pbar:
    for i in range(10000):
        pbar.update(1)
        score = 0
        game_over = False
        while not game_over:
            action = np.random.choice([0, 1])
            reward, next_state, game_over = game.step(action)
            if reward >= 1:
                score += 1
        scores.append(score)
        game.reset()

print('min', np.amin(scores), 'max', np.amax(scores), 'mean', np.mean(scores), 'std', np.std(scores)),


