import torch as torch
import numpy as np
from game import GameMDP
from collections import defaultdict
import random
from tqdm import tqdm
from time import sleep
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Softmax(torch.nn.Module):
    """
    Neural network that uses pytorch Softmax to choose an action
    TODO: try other policies (Adam?)
    """
    def __init__(self, total_states, hidden_layers, total_actions):
        super(Softmax, self).__init__()
        self.linear1 = torch.nn.Linear(total_states, hidden_layers)
        self.linear2 = torch.nn.Linear(hidden_layers, total_actions)

        # TODO: I think I need to store log_probs and rewards before feed forward,
        # but not sure. Need to figure out
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, inputs):
        inputs = torch.nn.functional.relu(self.linear1(inputs))
        action_scores = self.linear2(inputs)
        return torch.nn.functional.softmax(action_scores, dim=-1) #TODO: check dimensions

class PolicyGradient(object):
    """
    Policy Gradient training and evaluation
    TODO: try policies other than softmax
    """
    def __init__(self, game=None, args=None):
        self.game = game
        self.args = args
        self.actions = [0, 1]
        self.lr = self.args['lr']
        self.discount_factor = self.args['discount_factor']
        self.total_actions = 2 # actions are 0 or 1 (down or up)
        self.train_epochs = self.args['train_epochs']
        self.hidden_layers = self.args['hidden_layers']
        # store the number of states/observations
        self.total_states = len(self.game.get_state())
        self.policy = Softmax(self.total_states, self.hidden_layers, self.total_actions)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def sample_action(self, state):
        # policy = Softmax(self.total_states, self.hidden_layers, self.total_actions)
        # policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        # get state
        state = torch.from_numpy(state).float().unsqueeze(0)
        # TODO: object not callable error
        probabilities = self.policy(state)
        m = torch.distributions.Categorical(probabilities)
        action = m.sample()
        log_probability = m.log_prob(action)
        return action.item(), log_probability
        #############################

        # probabilities = self.policy_optimizer(state)
        # m = torch.distributions.Categorical(probabilities)
        # action = m.sample()
        # log_probability = m.log_prob(action)
        # # return action.item(), log_probability

    def discount_rewards(self, rewards):
        # temporally adjusted discounted rewards
        discounted_rewards = []
        # accumulated rewards initialized to zero
        cumulative_rewards = 0
        # start w/ rewards at end of trajectory
        for reward in rewards[::-1]:
            cumulative_rewards = reward + (cumulative_rewards * self.discount_factor)
            discounted_rewards.insert(0, cumulative_rewards)
        # normalize the discounted rewards to avoid outlier variability in rewards
        # TODO: try w/o normalization
        discounted_rewards = torch.tensor((discounted_rewards - np.mean(discounted_rewards))
                                          / np.std(discounted_rewards) + float(1e-9))
        return discounted_rewards

    def train(self):
        with tqdm(total=self.train_epochs, desc='PolicyGradient Train Progress Bar') as pbar:
            scores = []
            epoch_rewards = []
            for i in range(self.train_epochs):
                pbar.update(1)

                # run one simulation
                # trajectory is a list of tuples in the from
                # [(action, state, log_probability(a_t|s_t), reward)_1, ... _t]
                score, trajectory = self.run_simulation()
                scores.append(score)

                # update policy
                if self.args['order'] == 'backward':
                    trajectory = reversed(trajectory)
                self.update(trajectory)

                # states = torch.tensor(states)
                # actions = torch.tensor(actions)
                # future_rewards = self.discount_rewards(rewards)
                # rewards = torch.tensor(rewards)
                # future_rewards = torch.tensor(future_rewards)

                # track simulation total_rewards
                # epoch_rewards.append(total_rewards)


    def evaluate(self, epochs=1000):
        with tqdm(total=epochs, desc='PolicyGradient Evaluate Progress Bar') as pbar:
            scores = []
            for i in range(epochs):
                pbar.update(1)
                score, trajectory = self.run_simulation()
                scores.append(score)
            return scores

    def run_simulation(self):
        samples = []
        # restarts game
        self.game.reset()
        game_over = False
        # track score per epoch
        score = 0
        # tracks reward per epoch
        epoch_reward = 0
        # observations/states stored in 7-tuple state-space by default (may change w/ discretization)
        state = self.game.get_state() # observation is the state
        # want to pass observations/states into neural network and output log probabilities of actions
        # (up or down) (will sum to one)

        # trajectory stores tuples of (action, state, log_probability, reward, next_state, game_over)
        trajectory = []

        next_state = None
        while not game_over:
            # pick action based on log_probability
            action, log_probability = self.sample_action(np.array(state))
            # take action and store reward, next state, and point
            reward, next_state, game_over, point = self.game.step(action)
            # store trajectory step in trajectories
            trajectory.append((action, state, log_probability, reward, next_state, game_over))

            state = next_state
            score += point
            epoch_reward += reward

        # When game over, returns trajectory list and score for updating
        # and evaluating the model
        return score, trajectory

    def update(self, trajectory):
        """
        trajectory is format [(action, state, log_probability, reward, next_state, game_over),...]
        """

        # store all trajectory log probabilities
        log_probs = [tupl[2] for tupl in trajectory]
        # store all trajectory rewards
        rewards = [tupl[3] for tupl in trajectory]

        # store discounted rewards
        returns = self.discount_rewards(rewards)

        loss = []

        for log_prob, r in zip(log_probs, returns):
            loss.append(-(log_prob * r))

        loss = torch.stack(loss).sum()
        # zero gradients before backward pass
        self.policy_optimizer.zero_grad()
        # get gradient of loss w.r.t learnable model parameters
        # TODO: figure out if graph retain is necessary
        loss.backward(retain_graph=True)
        # loss.backward()
        self.policy_optimizer.step()
        # delete log_probs and rewards
        del log_probs
        del rewards
        #return loss

