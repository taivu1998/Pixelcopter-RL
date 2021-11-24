import numpy as np
from agents import sarsa
from game import DiscretizedGameMDP
from agents import Trainer

# def main():
#     args = {
#         'lr': 0.9,
#         'discount_factor': 0.9,
#         'order': 'backward',
#         'train_epochs': 10000,
#         'eval_epochs': 10000,
#         'epsilon': 0.2,
#     }
#     agent = QLearning(game=DiscretizedGameMDP(grid_size=5), args=args)
#     agent.train()
#     scores = agent.evaluate(epochs=10000)
#     print()
#     print("Min: {}".format(np.amin(scores)))
#     print("Max: {}".format(np.amax(scores)))
#     print("Mean: {}".format(np.mean(scores)))
#     print("Std: {}".format(np.std(scores)))


def main():
    args = {
        'mdp_type': 'discretized',
        'model_type': 'sarsa',
        'grid_size': 7,
        'lr': 0.02, # alpha
        'discount_factor': 0.9, # gamma
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