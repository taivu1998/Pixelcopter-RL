import numpy as np
import scipy.signal  # for calculating discounted rewards fast
from scipy.special import expit # expit is optimized logistic function
from game import GameMDP
from collections import defaultdict
import random
from tqdm import tqdm
from time import sleep

np.random.seed(2)

class PolicyGradient(object):

    def __init__(self, game=None, args=None):
        self.game = game
        self.args = args
        self.actions = [0, 1]
        # initialize 7 parameters to between -1 and 1
        self.parameters = np.random.uniform(low=-1, high=1, size=len(self.game.get_state()))
        print('this is self.parameters', self.parameters)
        self.lr = self.args['lr']
        self.discount_factor = self.args['discount_factor']
        self.total_observations = None
        self.total_actions = len(self.actions) # actions are 0 or 1
        self.train_epochs = self.args['train_epochs']

    def discounted_rewards(self, rewards):
        # discounted cumulative rewards
        # https://stackoverflow.com/questions/47970683/vectorize-a-numpy-discount-calculation - shows how to use
        # signal.lfilter()
        discounted_rewards = scipy.signal.lfilter([1], [1, float(-self.discount_factor)], rewards[::-1], axis=0)[::-1]

        # Normalizing discounted rewards avoids outlier variability in rewards
        # TODO: research if normalizing reduces exploration
        normalized_discounted_rewards = self.normalized_discounted_rewards(discounted_rewards)
        return normalized_discounted_rewards

    def normalized_discounted_rewards(self, discounted_rewards):
        mean = np.mean(discounted_rewards)
        std = np.clip(np.std(discounted_rewards), a_min=1e-10, a_max=None)  # clip to prevent divide by zero
        return (discounted_rewards - mean) / std

    def probabilities(self, x):
        """
        returns probabilities of actions 0 or 1 (down or up)
        """
        # expit scipy optimized is logistic function
        prob_of_action_0 = expit(np.dot(x, self.parameters))
        # prob. of action 1 is inverse of prob. of action 0
        prob_of_action_1 = 1 - prob_of_action_0
        return np.array([prob_of_action_0, prob_of_action_1])

    def gradient_log_probabilities(self, x):
        # gradient log probabilities
        logistic = expit(np.dot(x, self.parameters))
        gradient_log_prob_action_0 = x - (x * logistic)
        gradient_log_prob_action_1 = - (x * logistic)
        return gradient_log_prob_action_0, gradient_log_prob_action_1


    def train(self):
        with tqdm(total=self.train_epochs, desc='PolicyGradient Train Progress Bar') as pbar:
            scores = []
            for i in range(self.train_epochs):
                pbar.update(1)

                # run one simulation
                score, rewards, states, actions, probabilities = self.run_simulation()

                # track epoch scores
                scores.append(score)

                # if 'backward', reverse order
                # if self.args['order'] == 'backward':
                #     rewards = reversed(rewards)
                #     states = reversed(states)
                #     actions = reversed(actions)

                # update policy
                self.update(rewards, states, actions)

    def evaluate(self, epochs=1000):
        with tqdm(total=epochs, desc='PolicyGradient Evaluate Progress Bar') as pbar:
            scores = []
            for i in range(epochs):
                pbar.update(1)
                score, rewards, states, actions, probabilities = self.run_simulation()
                scores.append(score)
            return scores

    def run_simulation(self):
        self.game.reset()
        game_over = False
        score = 0
        # states=observations: stored in 7-tuple state-space by default (may change w/ discretization)
        state = self.game.get_state()

        # TODO: try passing states to neural network and output probabilities of actions
        # (up or down) (will sum to one)

        states = [] # states/observations
        actions = []
        rewards = []
        probabilities = []

        # trajectory stores tuples of (action, state, log_probability, reward, next_state, game_over)
        trajectory = []
        next_state = None

        while not game_over:
            states.append(state)

            action, prob = self.pick_action(state)
            reward, next_state, game_over, point = self.game.step(action)

            state = next_state
            score += point
            rewards.append(reward)
            actions.append(action)
            probabilities.append(prob)

        return score, np.array(rewards), np.array(states), np.array(actions), np.array(probabilities)

    def update(self, rewards, states, actions):
        # for every state, gets gradients for each action
        gradient_log_probabilities = np.array([self.gradient_log_probabilities(state)[action] for
                                               state, action in zip(states, actions)])

        # temporally adjusted discounted rewards
        discounted_rewards = self.discounted_rewards(rewards)

        # gradient ascend parameters
        self.parameters = self.parameters + (self.lr * (gradient_log_probabilities.T.dot(discounted_rewards)))
        # decay step factor each update (pg 244 Aglos for Decision Making)
        self.lr *= 0.999 # doesn't work well. TODO: Check Mykel's optimiziation book for more details on this

    def pick_action(self, x):
        # pick action in accordance with probabilities
        probabilities = self.probabilities(x)
        # get action from probabilities
        action = np.random.choice([0, 1], p=probabilities)
        return action, probabilities[action]




