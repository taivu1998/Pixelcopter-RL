import numpy as np
from agents import sarsa
from game import DiscretizedGameMDP
from agents import Trainer


def main():
    args = {
        'mdp_type': 'discretized',
        'model_type': 'sarsa',
        'grid_size': 8,
        'lr': 0.04, # alpha
        'discount_factor': 0.95, # gamma
        'order': 'forward',
        'train_epochs': 20001,
        'eval_epochs': 100,
        'epsilon': 0.15, # try starting high and decreasing epsilon moderately
        'epsilon_decay': 0.995
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