
import numpy as np
from agents import PolicyGradient
from game import DiscretizedGameMDP
from agents import Trainer

def main():
    args = {
        'mdp_type': 'discretized',
        'model_type': 'policyGradient',
        'grid_size': 14,
        # lr: 0.01 better than ?
        'lr': 0.01, # alpha
        # discount factor: 989 better 98 better than 99
        'discount_factor': 0.989, # gamma
        'order': 'forward',
        'train_epochs': 4000,
        'eval_epochs': 2000,
        'hidden_layers': 10, # TODO: try gradient policy w/ neural network
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