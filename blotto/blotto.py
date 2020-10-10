import time
import os

from absl import app
from absl import flags
import numpy as np
import matplotlib

# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pickle
from scipy import stats

import generalized_psro
import optimization_oracle
import pyspiel
from open_spiel.python.algorithms import policy_aggregator
from open_spiel.python.algorithms import exploitability

time_str = time.strftime("%Y%m%d-%H%M%S")


def plot_error(data, label=''):
    avg = np.mean(np.array(data), axis=0)
    error = stats.sem(np.array(data))
    plt.plot(avg, label=label)
    plt.fill_between([i for i in range(avg.shape[0])], avg - error, avg + error, alpha=0.3)


def run_psro(params, experiment=None, phase=None):
    game_name = params['game_name']
    number_players = params['number_players']
    sims_per_entry = params['sims_per_entry']
    rectify_training = params['rectify_training']
    training_strategy_selector = params['training_strategy_selector']
    meta_strategy_method = params['meta_strategy_method']
    number_policies_sampled = params['number_policies_sampled']
    number_episodes_sampled = params['number_episodes_sampled']
    rnr_iterations = params['rnr_iterations']
    lambda_weight = params['lambda_weight']

    meta_games = []
    meta_probabilities = []
    exp1 = []
    exp2 = []


    if game_name== 'blotto':
        game = pyspiel.load_game_as_turn_based(game_name, {"players": pyspiel.GameParameter(number_players)})
    else:
        game = pyspiel.load_game(game_name)

    oracle = optimization_oracle.EvolutionaryStrategyOracle(
        number_policies_sampled=number_policies_sampled,
        number_episodes_sampled=number_episodes_sampled,
        nb_players=number_players)

    g_psro_solver = generalized_psro.GenPSROSolver(game,
                                                   oracle,
                                                   sims_per_entry=sims_per_entry,
                                                   rectify_training=rectify_training,
                                                   training_strategy_selector=training_strategy_selector,
                                                   meta_strategy_method=meta_strategy_method,
                                                   lambda_weight=lambda_weight
                                                   )


    for i in range(rnr_iterations):

        start = time.time()
        g_psro_solver.iteration()

        meta_games.append(g_psro_solver.get_meta_game)
        meta_probabilities.append(g_psro_solver.get_and_update_meta_strategies())

        aggregator = policy_aggregator.PolicyAggregator(game)
        aggr_policies = aggregator.aggregate(
            range(2), g_psro_solver.get_policies, g_psro_solver._meta_strategy_probabilities)

        exploitabilities, expl_per_player = exploitability.nash_conv(
            game, aggr_policies, return_only_nash_conv=False)

        exp1.append(expl_per_player[0])
        exp2.append(expl_per_player[1])


        print('Phase ' + str(phase) +'Exper. ' + str(experiment) +'Iter ' + str(i) +
              ' in ' + str(time.time() - start))


    return meta_games, meta_probabilities, exp1, exp2


def main(params, num_exps=2, path=None):

    # DPP vs PSRO
    params['lambda_weight'] = [0.85, 1.]
    if params['rectify']:
        params['rectify_training'] = [False, True]
    else:
        params['rectify_training'] = [False, False]
    meta_games_dpp = []
    meta_probabilities_dpp = []
    exp1_dpp = []
    exp2_dpp = []
    for i in range(num_exps):
        meta_games, meta_probabilities, exp1, exp2 = run_psro(params, experiment=i, phase=0)
        meta_games_dpp.append(meta_games)
        meta_probabilities_dpp.append(meta_probabilities)
        exp1_dpp.append(exp1)
        exp2_dpp.append(exp2)

        pickle.dump({
                     'meta_games': meta_games_dpp,
                     'meta_probabilities': meta_probabilities_dpp,
                     'exp1': exp1_dpp,
                     'exp2': exp2_dpp,
                     },
                    open(os.path.join(path, 'data_dpp.p'), 'wb'))

    # PSRO vs PSRO
    params['lambda_weight'] = [1., 1.]
    if params['rectify']:
        params['rectify_training'] = [True, True]
    else:
        params['rectify_training'] = [False, False]
    meta_games_orig = []
    meta_probabilities_orig = []
    exp1_orig = []
    exp2_orig = []
    for i in range(num_exps):
        meta_games, meta_probabilities, exp1, exp2 = run_psro(params, experiment=i, phase=1)
        meta_games_orig.append(meta_games)
        meta_probabilities_orig.append(meta_probabilities)
        exp1_orig.append(exp1)
        exp2_orig.append(exp2)

        pickle.dump({
                     'meta_games': meta_games_orig,
                     'meta_probabilities': meta_probabilities_orig,
                     'exp1': exp1_orig,
                     'exp2': exp2_orig,
                     },
                    open(os.path.join(path, 'data_orig.p'), 'wb'))

    plt.figure()
    plot_error(exp1_dpp+exp2_dpp, label='dpp VS rectify')
    plot_error(exp1_orig+exp2_orig, label='rectify VS rectify')
    plt.savefig(os.path.join(path, 'exp.pdf'))
    plt.figure()
    plot_error(exp1_dpp, label='Exp DPP p1')
    plot_error(exp1_orig, label='Exp rectify p1')
    plt.savefig(os.path.join(path, 'exp1.pdf'))
    plt.figure()
    plot_error(exp2_orig, label='Exp rectify p2')
    plot_error(exp2_dpp, label='Exp DPP p2')
    plt.savefig(os.path.join(path, 'exp2.pdf'))
    plt.figure()

    plt.show()

if __name__ == "__main__":
    params = {
        # Game
        'game_name':'blotto',
        'number_players': 2,

        # PSRO
        'sims_per_entry': 5,  # Number of simulations to run to estimate each element of the game outcome matrix.
        'rectify_training': False,  # A boolean, specifying whether to train only against opponents we beat (True).
        'training_strategy_selector': 'probabilistic',
        # How to select the strategies to start training from
        #      String value can be:
        #        - "probabilistic_deterministic": selects the first
        #          policies with highest selection probabilities.
        #        - "probabilistic": randomly selects policies with
        #           probabilities determined by the meta strategies.
        #        - "exhaustive": selects every policy of every player.
        #        - "rectified": only selects strategies that have nonzero chance of
        #          being selected.
        #        - "uniform": randomly selects kwargs["number_policies_selected"]
        #           policies with uniform probabilities.
        'meta_strategy_method': 'nash',
        # String or callable taking a GenPSROSolver object and
        # returning a list of meta strategies (One list entry per player).
        #   String value can be:
        #       - "uniform": Uniform distribution on policies.
        #       - "nash": Taking nash distribution. Only works for 2 player, 0-sum games.
        #       - "prd": Projected Replicator Dynamics
        'rnr_iterations': 150,

        # Oracle parameters
        'number_policies_sampled': 50, # 50  # Number of different opponent policies sampled during evaluation of policy.
        'number_episodes_sampled': 5, # Number of episodes sampled to estimate the return  of different opponent policies.
        'lambda_weight': [0.85, 1.],  # Player 1 does dpp player 2 does not
    }

    path = os.path.join('results_rect', params['game_name'] + '_versus' + time_str)
    if not os.path.exists(path):
        os.makedirs(path)

    num_exps = 10

    main(params, num_exps=num_exps, path=path)

