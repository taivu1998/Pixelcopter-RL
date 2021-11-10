from ple import PLE
from ple.games.pixelcopter import Pixelcopter
# from ple import PLE
import gym
# from gym.wrappers import Monitor
# import gym_ple
# from gym_ple import PLEEnv
import numpy as np


# game = Pixelcopter()
# p = PLE(game, fps=30, display_screen=True)
# p.init()


# class GameMDP(gym.Wrapper):
#     ''' Game environment for SARSA and Q-Learning. '''
    
#     def __init__(self):
#         '''
#         Initializes the environment.
        
#         Args:
#             env (PLEEnv): A Pygame environment.
#             rounding (int): The level of discretization.
#         '''
#         # super().__init__(gym.make('PixelCopter-v0'))
#         # # super().__init__(gym.make('FlappyBird-v0'))
#         # self.env = Monitor(self.env, directory = "./out", force = True)



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
for i in range(10000):
    score = 0
    game_over = False
    while not game_over:
        action = np.random.choice([0, 1])
        reward, next_state, game_over = game.step(action)
        # print(reward)
        if reward >= 1:
            score += 1
    scores.append(score)
    game.reset()

print(np.amin(scores), np.amax(scores), np.mean(scores), np.std(scores)),


