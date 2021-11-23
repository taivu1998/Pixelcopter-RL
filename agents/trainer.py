from .random_model import Random
from .qlearning import QLearning
from game import GameMDP, DiscretizedGameMDP

MODEL_TYPES = {
    'random': Random,
    'qlearning': QLearning,
}

MDP_TYPES = {
    'normal': GameMDP,
    'discretized': DiscretizedGameMDP,
}
class Trainer(object):

    def __init__(self, args=None):
        self.args = args
        if 'grid_size' in self.args and self.args['grid_size'] > 0:
            game = MDP_TYPES[self.args['mdp_type']](grid_size=self.args['grid_size'])
        else:
            game = MDP_TYPES[self.args['mdp_type']]()
        self.model = MODEL_TYPES[self.args['model_type']](game=game, args=self.args)

    def train(self):
        self.model.train()

    def evaluate(self):
        scores = self.model.evaluate(epochs=self.args['eval_epochs'])
        return scores

    def save_scores(self):
        return

    def save_q_values(self):
        return