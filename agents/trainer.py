from .random_model import Random
from .qlearning import QLearning

model_names = {
    'random': Random,
    'qlearning': QLearning,
}
class QLearning(object):

    def __init__(self, args=None):
        self.args = args

    def train(self):
        return

    def evaluate(self):
        return