
import numpy as np
from agents import QLearning_NN
from game import DiscretizedGameMDP
from agents import Trainer


def main():
    args = {
        'mdp_type': 'discretized',
        'model_type': 'qlearning_nn',
        'grid_size': 10,
        'lr': 0.02,
        'discount_factor': 0.92,
        'order': 'backward',
        'train_epochs': 10001,
        'eval_epochs': 100,
        'epsilon': 0.1,
        'epsilon_decay': 0.995,
    }
    trainer = Trainer(args=args)
    trainer.train()
    scores = trainer.evaluate()
    print()
    print("Min: {}".format(np.amin(scores)))
    print("Max: {}".format(np.amax(scores)))
    print("Mean: {}".format(np.mean(scores)))
    print("Std: {}".format(np.std(scores)))


if __name__ == '__main__':
    main()