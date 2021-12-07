import numpy as np
from agents import QLearning
from game import DiscretizedGameMDP
from agents import Trainer


def main():
    args = {
        'mdp_type': 'discretized',
        'model_type': 'qlearning',
        'grid_size': 8,
        'lr': 0.025, #* 0.02
        'discount_factor': 0.9,
        'order': 'backward',
        'train_epochs': 20001,
        'eval_epochs': 100,
        'epsilon': 0.2, # try starting high and decreasing epsilon moderately
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