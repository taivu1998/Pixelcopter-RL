import numpy as np
from agents import QLearning
from game import DiscretizedGameMDP
from agents import Trainer

def main():
    args = {
        'mdp_type': 'discretized',
        'model_type': 'qlearning_funcapprox_nn',
        'grid_size': 10,
        'lr': 0.0005,
        'discount_factor': 0.92,
        'order': 'backward',
        'train_epochs': 20001,
        'eval_epochs': 100,
        'epsilon': 0.3,
        'epsilon_decay': 0.995,
        'experience_replay': True,
        'experience_replay_length': 50,
        'experience_replay_hidden_dims': [16, 32],
        'batch_size': 64,
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