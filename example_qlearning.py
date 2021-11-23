import numpy as np
from agents import QLearning
from game import DiscretizedGameMDP


def main():
    args = {
        'lr': 0.9,
        'discount_factor': 0.9,
        'order': 'backward',
        'train_epochs': 10000,
        'eval_epochs': 10000,
        'epsilon': 0.2,
    }
    agent = QLearning(game=DiscretizedGameMDP(grid_size=5), args=args)
    agent.train()
    scores = agent.evaluate(epochs=10000)
    print()
    print("Min: {}".format(np.amin(scores)))
    print("Max: {}".format(np.amax(scores)))
    print("Mean: {}".format(np.mean(scores)))
    print("Std: {}".format(np.std(scores)))


if __name__ == '__main__':
    main()