import numpy as np
from agents import sarsaLambda
from game import DiscretizedGameMDP
from agents import Trainer


def main():
    args = {
        'mdp_type': 'discretized',
        'model_type': 'sarsaLambda',
        'grid_size': 17,
        'lr': 0.05, # alpha
        'discount_factor': 0.94, # gamma
        'order': 'backward',
        'train_epochs': 10000,
        'eval_epochs': 10000,
        'epsilon': 1,
        'epsilon_decay': 0.99,
        # Generally, trace decay 0.9 works well but try [0, 0.5, 0.8, 0.9, 0.95, 1.0]
        'trace_decay': 0.4 # trace decay rate (common to use between 0 and 1)

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