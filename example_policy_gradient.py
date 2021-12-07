
import numpy as np
from agents import PolicyGradient
from game import DiscretizedGameMDP
from agents import Trainer

def main():
    args = {
        'mdp_type': 'discretized',
        'model_type': 'policyGradient',
        'grid_size': 12,
        # lr: much higher than 0.0001 seems to cause overshooting
        'lr': 0.0001, # alpha
        'min_lr': 0.00001,
        'lr_decay': 0.9999,
        # discount factor: 0.95 to 0.98 seem to be ideal
        'discount_factor': 0.96,
        'order': 'forward',
        'train_epochs': 20001,
        'eval_epochs': 100,
        'hidden_layers': 20,  # TODO: try gradient policy w/ neural network
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