import numpy as np
from agents import Random
from game import GameMDP
import pickle
import matplotlib.pyplot as plt


def main():
    ###### Loads all evaluation results from pickel files to dictionaries ###########################################
    with open('qlearning_evals.pkl', 'rb') as f:
        # eval_dict = {'0-train-pochs': tuple(mean, std, min, max), '1000-train-epochs': tuple(mean, std, min, max),
        # '2000-train-epochs': tuple(mean, std, min, max), etc.}
        qlearning_dict = pickle.load(f)
    # with open('qlearning_evals_forward.pkl', 'rb') as f:
    #     qlearning_forward_dict = pickle.load(f)
    # with open('qlearning_evals_backward.pkl', 'rb') as f:
    #     qlearning_backward_dict = pickle.load(f)

    with open('sarsa_evals.pkl', 'rb') as f:
        sarsa_dict = pickle.load(f)
    # with open('sarsa_evals_forward.pkl', 'rb') as f:
    #     sarsa_forward_dict = pickle.load(f)
    # with open('sarsa_evals_backward.pkl', 'rb') as f:
    #     sarsa_backward_dict = pickle.load(f)

    with open('qLambda_evals.pkl', 'rb') as f:
        qLambda_dict = pickle.load(f)
    # with open('qLambda_evals_forward.pkl', 'rb') as f:
    #     qLambda_forward_dict = pickle.load(f)
    # with open('qLambda_evals_backward.pkl', 'rb') as f:
    #     qLambda_backward_dict = pickle.load(f)

    with open('sarsaLambda_evals.pkl', 'rb') as f:
        sarsaLambda_dict = pickle.load(f)
    # with open('sarsaLambda_evals_forward.pkl', 'rb') as f:
    #     sarsaLambda_forward_dict = pickle.load(f)
    # with open('sarsaLambda_evals_backward.pkl', 'rb') as f:
    #     sarsaLambda_backward_dict = pickle.load(f)

    with open('policyGradient_evals.pkl', 'rb') as f:
        policyGradient_dict = pickle.load(f)
    with open('qlearning_nn_evals.pkl', 'rb') as f:
        qlearning_nn_dict = pickle.load(f)
    with open('qlearning_funcapprox_nn_evals.pkl', 'rb') as f:
        qlearning_funcapprox_nn_dict = pickle.load(f)


    # with open('sarsaLambda_evals_normal.pkl', 'rb') as f:
    #     sarsaLambda_normal_dict = pickle.load(f)
    # with open('sarsaLambda_evals_grid5.pkl', 'rb') as f:
    #     sarsaLambda_grid5_dict = pickle.load(f)
    # with open('sarsaLambda_evals_grid10.pkl', 'rb') as f:
    #     sarsaLambda_grid10_dict = pickle.load(f)
    # with open('sarsaLambda_evals_grid15.pkl', 'rb') as f:
    #     sarsaLambda_grid15_dict = pickle.load(f)
    # with open('sarsaLambda_evals_grid20.pkl', 'rb') as f:
    #     sarsaLambda_grid20_dict = pickle.load(f)
    ##################################################################################################################
    ######### Q-Learning and Sarsa Plot #################
    plt.title('Q-Learning & Sarsa Evaluation per Thousand Training Epochs')
    plt.ylabel('Mean Score (1 point per barrier passed)')
    plt.xlabel('Thousand Training Epochs')
    epochs = [int(epoch/1000) for epoch in qlearning_dict.keys()]
    plt.xticks(range(len(qlearning_dict)), epochs)
    plt.plot(range(len(qlearning_dict)), [eval_results[0] for eval_results in list(qlearning_dict.values())], 'r.-', label='Q-Learning')
    plt.plot(range(len(sarsa_dict)), [eval_results[0] for eval_results in list(sarsa_dict.values())], 'b.-', label='Sarsa')

    ticks = [i for i in range(0, len(qlearning_dict) + 1, 5)]
    plt.xticks(np.arange(0, len(qlearning_dict) + 1, 5), ticks)  # Set text labels.
    plt.legend()
    plt.savefig("Q-Learning and Sarsa Plot.png")
    plt.show()

    ######################################################################################################
    ##### Q-Learning Error Bar Plots ##########################################################################
    plt.title('Q-Learning Error Bars per Thousand Training Epochs')
    plt.ylabel('Mean Score Standard Deviation, Min & Max')
    plt.xlabel('Thousand Training Epochs')
    # get data from dicts and put in lists
    mins = np.array([eval_results[2] for eval_results in list(qlearning_dict.values())])
    maxes = np.array([eval_results[3] for eval_results in list(qlearning_dict.values())])
    means = np.array([eval_results[0] for eval_results in list(qlearning_dict.values())])
    std = np.array([eval_results[1] for eval_results in list(qlearning_dict.values())])

    # create errorbars and connect means with plot line:
    plt.errorbar(range(len(qlearning_dict)), means, std, mec='blue', mfc='blue', fmt='ok', lw=6, label='Standard Deviation')
    plt.plot(range(len(qlearning_dict)), means, 'b.-',
             label='Mean')
    plt.errorbar(range(len(qlearning_dict)), means, [means - mins, maxes - means],
                 fmt='.k', ecolor='green', lw=1, label='Minimum\Maximum')

    # sets legend and ticks
    ticks = [i for i in range(0, len(qlearning_dict) + 1, 5)]
    plt.xticks(np.arange(0, len(qlearning_dict) + 1, 5), ticks)  # Set text labels.
    plt.legend()
    plt.savefig("Q-Learning Error Bars.png")
    plt.show()
    ######################################################################################################
    # ######### Sarsa Bar Chart #################
    # plt.title('Sarsa Evaluation per Thousand Training Epochs')
    # plt.ylabel('Score (1 point per barrier passed)')
    # plt.xlabel('Thousand Training Epochs')
    # # plt.legend(['train', 'test'], loc='upper left')
    # epochs = [int(epoch / 1000) for epoch in sarsa_dict.keys()]
    # plt.bar(range(len(sarsa_dict)), [eval_results[0] for eval_results in list(sarsa_dict.values())],
    #         align='center')
    # plt.xticks(range(len(sarsa_dict)), epochs)
    # plt.savefig("sarsa.png")
    # plt.show()
    #######################################################################################################
    ##### Sarsa Error Bar Plots ##########################################################################
    plt.title('Sarsa Error Bars per Thousand Training Epochs')
    plt.ylabel('Mean Score Standard Deviation, Min & Max')
    plt.xlabel('Thousand Training Epochs')
    # get data from dicts and put in lists
    mins = np.array([eval_results[2] for eval_results in list(sarsa_dict.values())])
    maxes = np.array([eval_results[3] for eval_results in list(sarsa_dict.values())])
    means = np.array([eval_results[0] for eval_results in list(sarsa_dict.values())])
    std = np.array([eval_results[1] for eval_results in list(sarsa_dict.values())])

    # create errorbars and connect means with plot line:
    plt.errorbar(range(len(sarsa_dict)), means, std, mec='blue', mfc='blue', fmt='ok', lw=6,
                 label='Standard Deviation')
    plt.plot(range(len(sarsa_dict)), means, 'b.-',
             label='Mean')
    plt.errorbar(range(len(sarsa_dict)), means, [means - mins, maxes - means],
                 fmt='.k', ecolor='green', lw=1, label='Minimum\Maximum')

    # sets legend and ticks
    ticks = [i for i in range(0, len(sarsa_dict) + 1, 5)]
    plt.xticks(np.arange(0, len(sarsa_dict) + 1, 5), ticks)  # Set text labels.
    plt.legend()
    plt.savefig("Sarsa Error Bars.png")
    plt.show()
    ######################################################################################################
    # ######### Q-Lambda Bar Chart #################
    # plt.title('Q-Lambda Evaluation per Thousand Training Epochs')
    # plt.ylabel('Score (1 point per barrier passed)')
    # plt.xlabel('Thousand Training Epochs')
    # # plt.legend(['train', 'test'], loc='upper left')
    # epochs = [int(epoch / 1000) for epoch in qLambda_dict.keys()]
    # plt.bar(range(len(qLambda_dict)), [eval_results[0] for eval_results in list(qLambda_dict.values())],
    #         align='center')
    # plt.xticks(range(len(qLambda_dict)), epochs)
    # plt.savefig("qLambda.png")
    # plt.show()
    ######################################################################################################
    ##### Q-Lambda Error Bar Plots ##########################################################################
    plt.title('Q-Lambda Error Bars per Thousand Training Epochs')
    plt.ylabel('Mean Score Standard Deviation, Min & Max')
    plt.xlabel('Thousand Training Epochs')
    # get data from dicts and put in lists
    mins = np.array([eval_results[2] for eval_results in list(qLambda_dict.values())])
    maxes = np.array([eval_results[3] for eval_results in list(qLambda_dict.values())])
    means = np.array([eval_results[0] for eval_results in list(qLambda_dict.values())])
    std = np.array([eval_results[1] for eval_results in list(qLambda_dict.values())])

    # create errorbars and connect means with plot line:
    plt.errorbar(range(len(qLambda_dict)), means, std, mec='blue', mfc='blue', fmt='ok', lw=6,
                 label='Standard Deviation')
    plt.plot(range(len(qLambda_dict)), means, 'b.-',
             label='Mean')
    plt.errorbar(range(len(qLambda_dict)), means, [means - mins, maxes - means],
                 fmt='.k', ecolor='green', lw=1, label='Minimum\Maximum')

    # sets legend and ticks
    ticks = [i for i in range(1, len(qLambda_dict) + 1)]
    plt.xticks(np.arange(len(qLambda_dict)), ticks)  # Set tick labels.
    plt.legend()
    plt.savefig("Q-Lambda Error Bars.png")
    plt.show()
    ######################################################################################################
    # ######### Sarsa-Lambda Bar Chart #################
    # plt.title('Sarsa-Lambda Evaluation per Thousand Training Epochs')
    # plt.ylabel('Score (1 point per barrier passed)')
    # plt.xlabel('Thousand Training Epochs')
    # # plt.legend(['train', 'test'], loc='upper left')
    # epochs = [int(epoch / 1000) for epoch in sarsaLambda_dict.keys()]
    # plt.bar(range(len(sarsaLambda_dict)), [eval_results[0] for eval_results in list(sarsaLambda_dict.values())],
    #         align='center')
    # plt.xticks(range(len(sarsaLambda_dict)), epochs)
    # plt.savefig("sarsaLambda.png")
    # plt.show()
    ######################################################################################################
    ##### Sarsa-Lambda Error Bar Plots ##########################################################################
    plt.title('Sarsa-Lambda Error Bars per Thousand Training Epochs')
    plt.ylabel('Mean Score Standard Deviation, Min & Max')
    plt.xlabel('Thousand Training Epochs')
    # get data from dicts and put in lists
    mins = np.array([eval_results[2] for eval_results in list(sarsaLambda_dict.values())])
    maxes = np.array([eval_results[3] for eval_results in list(sarsaLambda_dict.values())])
    means = np.array([eval_results[0] for eval_results in list(sarsaLambda_dict.values())])
    std = np.array([eval_results[1] for eval_results in list(sarsaLambda_dict.values())])

    # create errorbars and connect means with plot line:
    plt.errorbar(range(len(sarsaLambda_dict)), means, std, mec='blue', mfc='blue', fmt='ok', lw=6,
                 label='Standard Deviation')
    plt.plot(range(len(sarsaLambda_dict)), means, 'b.-',
             label='Mean')
    plt.errorbar(range(len(sarsaLambda_dict)), means, [means - mins, maxes - means],
                 fmt='.k', ecolor='green', lw=1, label='Minimum\Maximum')

    # sets legend and ticks
    ticks = [i for i in range(1, len(sarsaLambda_dict) + 1)]
    plt.xticks(np.arange(len(sarsaLambda_dict)), ticks)  # Set tick labels.
    plt.legend()
    plt.savefig("Sarsa-Lambda Error Bars.png")
    plt.show()
    ######################################################################################################
    ######### Q-Lambda and Sarsa-Lambda Plot #########################################################################
    # plt.title('Q-Lambda & Sarsa-Lambda Evaluation per Thousand Training Epochs')
    # plt.ylabel('Mean Score (1 point per barrier passed)')
    # plt.xlabel('Thousand Training Epochs')
    # epochs = [int(epoch / 1000) for epoch in qLambda_dict.keys()]

    # plt.bar(range(len(qlearning_dict)), [eval_results[0] for eval_results in list(qlearning_dict.values())], align='center')
    # plt.xticks(range(len(qlearning_dict)), epochs)
    ######################################################################################################
    # ##### Error Bar Plots ##########################################################################
    # # get data from dicts and put in lists
    # mins = [eval_results[2] for eval_results in list(qLambda_dict.values())]
    # maxes = [eval_results[3] for eval_results in list(qLambda_dict.values())]
    # means = [eval_results[0] for eval_results in list(qLambda_dict.values())]
    # std = [eval_results[1] for eval_results in list(qLambda_dict.values())]
    # # create stacked errorbars:
    # plt.errorbar(np.arange(8), means, std, fmt='ok', lw=3)
    # plt.errorbar(np.arange(8), means, [means - mins, maxes - means],
    #              fmt='.k', ecolor='gray', lw=1)
    # plt.xlim(-1, 8)
    # ######################################################################################################
    ###### Plot Graph #############################################################################################
    # plt.plot(range(len(qLambda_dict)), [eval_results[0] for eval_results in list(qLambda_dict.values())], 'r.-',
    #          label='Q-Lambda')
    # plt.plot(range(len(sarsaLambda_dict)), [eval_results[0] for eval_results in list(sarsaLambda_dict.values())], 'b.-',
    #          label='Sarsa-Lambda')
    ######################################################################################################
    # plt.xticks(range(len(qLambda_dict)), epochs)
    # # plt.xticks(np.arange(0, 51, 5))
    # plt.legend()
    #
    # plt.savefig("q-lambda and sarsa-lambda.png")
    # plt.show()

    ######################################################################################################
    # ######### Policy Gradient Bar Chart #################
    # plt.title('Policy Gradient Evaluation per Thousand Training Epochs')
    # plt.ylabel('Score (1 point per barrier passed)')
    # plt.xlabel('Thousand Training Epochs')
    # # plt.legend(['train', 'test'], loc='upper left')
    # epochs = [int(epoch / 1000) for epoch in policyGradient_dict.keys()]
    # plt.bar(range(len(policyGradient_dict)), [eval_results[0] for eval_results in list(policyGradient_dict.values())],
    #         align='center')
    # plt.xticks(range(len(policyGradient_dict)), epochs)
    # plt.savefig("policy_gradient.png")
    # plt.show()
    ######################################################################################################
    ##### Policy Gradient Error Bar Plots ##########################################################################
    plt.title('Policy Gradient Error Bars per Thousand Training Epochs')
    plt.ylabel('Mean Score Standard Deviation, Min & Max')
    plt.xlabel('Thousand Training Epochs')
    # get data from dicts and put in lists
    mins = np.array([eval_results[2] for eval_results in list(policyGradient_dict.values())])
    maxes = np.array([eval_results[3] for eval_results in list(policyGradient_dict.values())])
    means = np.array([eval_results[0] for eval_results in list(policyGradient_dict.values())])
    std = np.array([eval_results[1] for eval_results in list(policyGradient_dict.values())])

    # create errorbars and connect means with plot line:
    plt.errorbar(range(len(policyGradient_dict)), means, std, mec='blue', mfc='blue', fmt='ok', lw=6,
                 label='Standard Deviation')
    plt.plot(range(len(policyGradient_dict)), means, 'b.-',
             label='Mean')
    plt.errorbar(range(len(policyGradient_dict)), means, [means - mins, maxes - means],
                 fmt='.k', ecolor='green', lw=1, label='Minimum\Maximum')

    # sets legend and ticks
    ticks = [i for i in range(0, len(policyGradient_dict) + 1, 5)]
    plt.xticks(np.arange(0, len(policyGradient_dict) + 1, 5), ticks)  # Set text labels.
    plt.legend()
    plt.savefig("Policy Gradient Error Bars.png")
    plt.show()
    ######################################################################################################
    # ######### Q-Learning Nearest Neighbors Bar Chart #################
    # plt.title('Q-Learning Nearest Neighbors Evaluation per Thousand Training Epochs')
    # plt.ylabel('Score (1 point per barrier passed)')
    # plt.xlabel('Thousand Training Epochs')
    # # plt.legend(['train', 'test'], loc='upper left')
    # epochs = [int(epoch / 1000) for epoch in qlearning_nn_dict.keys()]
    # plt.bar(range(len(qlearning_nn_dict)), [eval_results[0] for eval_results in list(qlearning_nn_dict.values())],
    #         align='center')
    # plt.xticks(range(len(qlearning_nn_dict)), epochs)
    # plt.savefig("qlearning_nn.png")
    # plt.show()
    ######################################################################################################
    ##### Q-Learning Nearest Neighbors Error Bar Plots ##########################################################################
    plt.title('Q-Learning Nearest Neighbors Error Bars per Thousand Training Epochs')
    plt.ylabel('Mean Score Standard Deviation, Min & Max')
    plt.xlabel('Thousand Training Epochs')
    # get data from dicts and put in lists
    mins = np.array([eval_results[2] for eval_results in list(qlearning_nn_dict.values())])
    maxes = np.array([eval_results[3] for eval_results in list(qlearning_nn_dict.values())])
    means = np.array([eval_results[0] for eval_results in list(qlearning_nn_dict.values())])
    std = np.array([eval_results[1] for eval_results in list(qlearning_nn_dict.values())])

    # create errorbars and connect means with plot line:
    plt.errorbar(range(len(qlearning_nn_dict)), means, std, mec='blue', mfc='blue', fmt='ok', lw=6,
                 label='Standard Deviation')
    plt.plot(range(len(qlearning_nn_dict)), means, 'b.-',
             label='Mean')
    plt.errorbar(range(len(qlearning_nn_dict)), means, [means - mins, maxes - means],
                 fmt='.k', ecolor='green', lw=1, label='Minimum\Maximum')

    # sets legend and ticks
    ticks = [i for i in range(0, len(qlearning_nn_dict) + 1, 5)]
    plt.xticks(np.arange(0, len(qlearning_nn_dict) + 1, 5), ticks)  # Set text labels.
    plt.legend()
    plt.savefig("Q-Learning Nearest Neighbors Error Bars.png")
    plt.show()
    ######################################################################################################
    # ######### Q-Learning Function Approximation Nearest Neighbors Bar Chart #################
    # plt.title('Q-Learning Function Approximation Nearest Neighbors Evaluation per Thousand Training Epochs')
    # plt.ylabel('Score (1 point per barrier passed)')
    # plt.xlabel('Thousand Training Epochs')
    # # plt.legend(['train', 'test'], loc='upper left')
    # epochs = [int(epoch / 1000) for epoch in qlearning_funcapprox_nn_dict.keys()]
    # plt.bar(range(len(qlearning_funcapprox_nn_dict)), [eval_results[0] for eval_results in list(qlearning_funcapprox_nn_dict.values())],
    #         align='center')
    # plt.xticks(range(len(qlearning_funcapprox_nn_dict)), epochs)
    # plt.savefig("qlearning_funcapprox_nn.png")
    # plt.show()
    ######################################################################################################
    ##### Q-Learning Function Approximation Error Bar Plots ##########################################################################
    plt.title('Q-Learning Function Approximation Error Bars per Thousand Training Epochs')
    plt.ylabel('Mean Score Standard Deviation, Min & Max')
    plt.xlabel('Thousand Training Epochs')
    # get data from dicts and put in lists
    mins = np.array([eval_results[2] for eval_results in list(qlearning_funcapprox_nn_dict.values())])
    maxes = np.array([eval_results[3] for eval_results in list(qlearning_funcapprox_nn_dict.values())])
    means = np.array([eval_results[0] for eval_results in list(qlearning_funcapprox_nn_dict.values())])
    std = np.array([eval_results[1] for eval_results in list(qlearning_funcapprox_nn_dict.values())])

    # create errorbars and connect means with plot line:
    plt.errorbar(range(len(qlearning_funcapprox_nn_dict)), means, std, mec='blue', mfc='blue', fmt='ok', lw=6,
                 label='Standard Deviation')
    plt.plot(range(len(qlearning_funcapprox_nn_dict)), means, 'b.-',
             label='Mean')
    plt.errorbar(range(len(qlearning_funcapprox_nn_dict)), means, [means - mins, maxes - means],
                 fmt='.k', ecolor='green', lw=1, label='Minimum\Maximum')

    # sets legend and ticks
    ticks = [i for i in range(1, len(qlearning_funcapprox_nn_dict) + 1)]
    plt.xticks(np.arange(len(qlearning_funcapprox_nn_dict)), ticks)  # Set tick labels.
    plt.legend()
    plt.savefig("Q-Learning Function Approximation Error Bars.png")
    plt.show()
    ######################################################################################################
    # ######### All Model's Mean #################
    plt.title('All Model Mean Scores per Thousand Training Epochs')
    plt.ylabel('Mean Score (1 point per barrier passed)')
    plt.xlabel('Thousand Training Epochs')

    plt.plot(range(len(qlearning_dict)), [eval_results[0] for eval_results in list(qlearning_dict.values())], 'b.-',
             label='Q-Learning')
    plt.plot(range(len(qLambda_dict)), [eval_results[0] for eval_results in list(qLambda_dict.values())], 'g.-',
             label='Q-Lambda')
    plt.plot(range(len(sarsa_dict)), [eval_results[0] for eval_results in list(sarsa_dict.values())], 'r.-',
             label='Sarsa')
    plt.plot(range(len(sarsaLambda_dict)), [eval_results[0] for eval_results in list(sarsaLambda_dict.values())], 'c.-',
             label='Sarsa-Lambda')
    plt.plot(range(len(qlearning_nn_dict)), [eval_results[0] for eval_results in list(qlearning_nn_dict.values())], 'm.-',
             label='Q-Learning Nearest Neighbors')
    plt.plot(range(len(qlearning_funcapprox_nn_dict)), [eval_results[0] for eval_results in list(qlearning_funcapprox_nn_dict.values())], 'k.-',
             label='Q-Learning Function Approximation')
    plt.plot(range(len(policyGradient_dict)), [eval_results[0] for eval_results in list(policyGradient_dict.values())], 'y.-',
             label='Policy Gradient')

    ticks = [i for i in range(1, len(qlearning_dict)+1)]
    plt.xticks(np.arange(len(qlearning_dict)), ticks)  # Set text labels.
    plt.legend()
    plt.savefig("all models mean scores.png")
    plt.show()

    ######################################################################################################
    # ######### Vanilla Compared to Lambda Traces #################
    plt.title('Q & Sarsa vs. Q-Lambda & Sarsa-Lambda Mean Scores per Thousand Training Epochs')
    plt.ylabel('Mean Score (1 point per barrier passed)')
    plt.xlabel('Thousand Training Epochs')
    # Q and Sarsa
    plt.plot(range(len(qlearning_dict)), [eval_results[0] for eval_results in list(qlearning_dict.values())], 'r.-',
             label='Q-Learning')
    plt.plot(range(len(sarsa_dict)), [eval_results[0] for eval_results in list(sarsa_dict.values())], 'm.-',
             label='Sarsa')
    # Q-Lambda and Sarsa-Lambda
    plt.plot(range(len(qLambda_dict)), [eval_results[0] for eval_results in list(qLambda_dict.values())], 'c.-',
             label='Q-Lambda')
    plt.plot(range(len(sarsaLambda_dict)), [eval_results[0] for eval_results in list(sarsaLambda_dict.values())], 'b.-',
             label='Sarsa-Lambda')

    ticks = [i for i in range(1, len(qlearning_dict) + 1)]
    plt.xticks(np.arange(len(qlearning_dict)), ticks)  # Set text labels.
    plt.legend()
    plt.savefig("Q & Sarsa vs. Q-Lambda & Sarsa-Lambda.png")
    plt.show()
    ######################################################################################################

    # ######### Q-Lambda and Sarsa Lambda Model's Mean #################
    # plt.title('Q-Lambda and Sarsa Lambda per Thousand Training Epochs')
    # plt.ylabel('Mean Score (1 point per barrier passed)')
    # plt.xlabel('Thousand Training Epochs')
    #
    # plt.plot(range(len(qLambda_dict)), [eval_results[0] for eval_results in list(qLambda_dict.values())], 'g.-',
    #          label='Q-Lambda')
    # plt.plot(range(len(sarsaLambda_dict)), [eval_results[0] for eval_results in list(sarsaLambda_dict.values())], 'b.-',
    #          label='Sarsa-Lambda')
    #
    # ticks = [i for i in range(1, len(qlearning_dict) + 1)]
    # plt.xticks(np.arange(len(qlearning_dict)), ticks)  # Set text labels.
    # plt.legend()
    # plt.savefig("Q-Lambda and Sarsa-Lambda.png")
    # plt.show()

    ######################################################################################################
    # ######### Forward/Backward Comparisons #################
    # ## Q-Learning and Sarsa Forward-Backward ###
    # plt.title('Q-Learning and Sarsa Forward-Backward Mean Scores per Thousand Training Epochs')
    # plt.ylabel('Mean Score (1 point per barrier passed)')
    # plt.xlabel('Thousand Training Epochs')
    # ## Q-Learning Forward-Backward ###
    # plt.plot(range(len(qlearning_forward_dict)), [eval_results[0] for eval_results in list(qlearning_forward_dict.values())], 'r.-',
    #          label='Q-Learning Forward')
    # plt.plot(range(len(qlearning_backward_dict)),
    #          [eval_results[0] for eval_results in list(qlearning_backward_dict.values())], 'm.-',
    #          label='Q-Learning Backward')
    # ### Sarsa Forward-Backward ###
    # plt.plot(range(len(sarsa_forward_dict)),
    #          [eval_results[0] for eval_results in list(sarsa_forward_dict.values())], 'g.-',
    #          label='Sarsa Forward')
    # plt.plot(range(len(sarsa_backward_dict)),
    #          [eval_results[0] for eval_results in list(sarsa_backward_dict.values())], 'b.-',
    #          label='Sarsa Backward')
    #
    # ticks = [i for i in range(1, len(qlearning_dict) + 1)]
    # plt.xticks(np.arange(len(qlearning_dict)), ticks)  # Set text labels.
    # plt.legend()
    # plt.savefig("Q-Learning & Sarsa: Forward-Backward.png")
    # plt.show()

    ######################################################################################################
    ## Q-Lambda Sarsa-Lambda Forward-Backward ###
    # plt.title('Q-Lambda and Sarsa-Lambda Forward-Backward Mean Scores per Thousand Training Epochs')
    # plt.ylabel('Mean Score (1 point per barrier passed)')
    # plt.xlabel('Thousand Training Epochs')
    # plt.plot(range(len(qLambda_forward_dict)),
    #          [eval_results[0] for eval_results in list(qLambda_forward_dict.values())], 'r.-',
    #          label='Q-Lambda Forward')
    # plt.plot(range(len(qLambda_backward_dict)),
    #          [eval_results[0] for eval_results in list(qLambda_backward_dict.values())], 'm.-',
    #          label='Q-Lambda Backward')
    # ### Sarsa-Lambda Forward-Backward ###
    # plt.plot(range(len(sarsaLambda_forward_dict)),
    #          [eval_results[0] for eval_results in list(sarsaLambda_forward_dict.values())], 'g.-',
    #          label='Sarsa-Lambda Forward')
    # plt.plot(range(len(sarsaLambda_backward_dict)),
    #          [eval_results[0] for eval_results in list(sarsaLambda_backward_dict.values())], 'b.-',
    #          label='Sarsa-Lambda Backward')
    #
    # ticks = [i for i in range(1, len(qlearning_dict) + 1)]
    # plt.xticks(np.arange(len(qlearning_dict)), ticks)  # Set text labels.
    # plt.legend()
    # plt.savefig("Q-Lambda & Sarsa-Lambda: Forward-Backward.png")
    # plt.show()
    ######################################################################################################
    ## Discretization Effect on Sarsa-Lambda ###
    # plt.title('Discretization on Sarsa-Lambda: Mean Scores per Thousand Training Epochs')
    # plt.ylabel('Mean Score (1 point per barrier passed)')
    # plt.xlabel('Thousand Training Epochs')
    # plt.plot(range(len(sarsaLambda_normal_dict)),
    #          [eval_results[0] for eval_results in list(sarsaLambda_normal_dict.values())], 'r.-',
    #          label='Sarsa-Lambda No Discretization')
    # plt.plot(range(len(sarsaLambda_grid5_dict)),
    #          [eval_results[0] for eval_results in list(sarsaLambda_grid5_dict.values())], 'm.-',
    #          label='Sarsa-Lambda Dirscretization-Factor 5')
    # plt.plot(range(len(sarsaLambda_grid10_dict)),
    #          [eval_results[0] for eval_results in list(sarsaLambda_grid10_dict.values())], 'g.-',
    #          label='Sarsa-Lambda Dirscretization-Factor 10')
    # plt.plot(range(len(sarsaLambda_grid15_dict)),
    #          [eval_results[0] for eval_results in list(sarsaLambda_grid15_dict.values())], 'b.-',
    #          label='Sarsa-Lambda Dirscretization-Factor 15')
    # plt.plot(range(len(sarsaLambda_grid20_dict)),
    #          [eval_results[0] for eval_results in list(sarsaLambda_grid20_dict.values())], 'c.-',
    #          label='Sarsa-Lambda Dirscretization-Factor 20')
    #
    # ticks = [i for i in range(1, len(sarsaLambda_grid20_dict) + 1)]
    # plt.xticks(np.arange(len(sarsaLambda_grid20_dict)), ticks)  # Set text labels.
    # plt.legend()
    # plt.savefig("Different Discretizations for Sarsa-Lambda.png")
    # plt.show()
    #######################################################################################################



if __name__ == '__main__':
    main()