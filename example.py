import numpy as np
from agents import Random


def main():
    agent = Random()
    agent.train()
    scores = agent.evaluate(epochs=10000)
    print("Min: {}".format(np.amin(scores)))
    print("Max: {}".format(np.amax(scores)))
    print("Mean: {}".format(np.mean(scores)))
    print("Std: {}".format(np.std(scores)))


if __name__ == '__main__':
    main()