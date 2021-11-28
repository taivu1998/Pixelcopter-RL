
import numpy as np
# from agents import QLearning_KNN
from game import DiscretizedGameMDP
from agents import Trainer


def main():
    args = {
        'mdp_type': 'discretized',
        'model_type': 'qlearning_nn',
        'grid_size': 14,
        'lr': 0.05,
        'discount_factor': 0.97,
        'order': 'backward',
        'train_epochs': 10000,
        'eval_epochs': 10000,
        'epsilon': 0.2,
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