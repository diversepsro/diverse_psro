import numpy as np
import matplotlib.pyplot as plt
import time
import os
from scipy import stats
import pickle
import json
from numpy.random import RandomState
import argparse
import multiprocessing as mp


np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

np.random.seed(0)

parser = argparse.ArgumentParser(description='Random GAmes of Skill form DPP')
parser.add_argument('--dim', type=int, default=1000)
parser.add_argument('--nb_iters', type=int, default=200)

args = parser.parse_args()

LR = 0.5
TH = 0.03

expected_card = []
sizes = []

time_string = time.strftime("%Y%m%d-%H%M%S")
PATH_RESULTS = os.path.join('results', time_string + '_' + str(args.dim))
if not os.path.exists(PATH_RESULTS):
    os.makedirs(PATH_RESULTS)


# Search over the pure strategies to find the BR to a strategy
def get_br_to_strat(strat, payoffs=None, verbose=False):
    row_weighted_payouts = strat @ payoffs
    br = np.zeros_like(row_weighted_payouts)
    br[np.argmin(row_weighted_payouts)] = 1
    if verbose:
        print(row_weighted_payouts[np.argmin(row_weighted_payouts)], "exploitability")
    return br


# Fictituous play as a nash equilibrium solver
def fictitious_play(iters=2000, payoffs=None, verbose=False):
    dim = payoffs.shape[0]
    pop = np.random.uniform(0, 1, (1, dim))
    pop = pop / pop.sum(axis=1)[:, None]
    averages = pop
    exps = []
    for i in range(iters):
        average = np.average(pop, axis=0)
        br = get_br_to_strat(average, payoffs=payoffs)
        exp1 = average @ payoffs @ br.T
        exp2 = br @ payoffs @ average.T
        exps.append(exp2 - exp1)
        # if verbose:
        #     print(exp, "exploitability")
        averages = np.vstack((averages, average))
        pop = np.vstack((pop, br))
    return averages, exps


# Solve exploitability of a nash equilibrium over a fixed population
def get_exploitability(pop, payoffs, iters=1000):
    emp_game_matrix = pop @ payoffs @ pop.T
    averages, _ = fictitious_play(payoffs=emp_game_matrix, iters=iters)
    strat = averages[-1] @ pop  # Aggregate
    test_br = get_br_to_strat(strat, payoffs=payoffs)
    exp1 = strat @ payoffs @ test_br.T
    exp2 = test_br @ payoffs @ strat
    return exp2 - exp1


def joint_loss(pop, payoffs, meta_nash, k, lambda_weight, lr):
    dim = payoffs.shape[0]

    br = np.zeros((dim,))
    values = []
    cards = []
    for i in range(dim):
        br_tmp = np.zeros((dim, ))
        br_tmp[i] = 1.

        aggregated_enemy = meta_nash @ pop[:k]
        value = br_tmp @ payoffs @ aggregated_enemy.T

        pop_k = lr * br_tmp + (1 - lr) * pop[k]
        pop_tmp = np.vstack((pop[:k], pop_k))
        M = pop_tmp @ payoffs @ pop_tmp.T
        metanash_tmp, _ = fictitious_play(payoffs=M, iters=1000)
        L = np.diag(metanash_tmp[-1]) @ M @ M.T @ np.diag(metanash_tmp[-1])
        l_card = np.trace(np.eye(L.shape[0]) - np.linalg.inv(L + np.eye(L.shape[0])))

        cards.append(l_card)
        values.append(value)

    if np.random.randn() < lambda_weight:
        br[np.argmax(values)] = 1
    else:
        br[np.argmax(cards)] = 1

    return br


def psro_steps(iters=5, payoffs=None, verbose=False, seed=0,
                        num_learners=4, improvement_pct_threshold=.03, lr=.2, loss_func='dpp', full=False):
    dim = payoffs.shape[0]

    r = np.random.RandomState(seed)
    pop = r.uniform(0, 1, (1 + num_learners, dim))
    pop = pop / pop.sum(axis=1)[:, None]
    exp = get_exploitability(pop, payoffs, iters=1000)
    exps = [exp]

    M = pop @ payoffs @ pop.T
    L = M @ M.T
    l_card = np.trace(np.eye(L.shape[0]) - np.linalg.inv(L + np.eye(L.shape[0])))
    l_cards = [l_card]

    learner_performances = [[.1] for i in range(num_learners + 1)]
    for i in range(iters):
        # Define the weighting towards diversity as a function of the fixed population size, this is currently a hyperparameter
        lambda_weight = 0.85
        if i % 5 == 0:
            print('iteration: ', i, ' exp full: ', exps[-1])
            print('size of pop: ', pop.shape[0])

        for j in range(num_learners):
            # first learner (when j=num_learners-1) plays against normal meta Nash
            # second learner plays against meta Nash with first learner included, etc.
            k = pop.shape[0] - j - 1
            emp_game_matrix = pop[:k] @ payoffs @ pop[:k].T
            meta_nash, _ = fictitious_play(payoffs=emp_game_matrix, iters=1000)
            population_strategy = meta_nash[-1] @ pop[:k]  # aggregated enemy according to nash

            if loss_func == 'br':
                # standard PSRO
                br = get_br_to_strat(population_strategy, payoffs=payoffs)
            else:
                # Diverse PSRO
                br = joint_loss(pop, payoffs, meta_nash[-1], k, lambda_weight, lr)
                br_orig = get_br_to_strat(population_strategy, payoffs=payoffs)

            # Update the mixed strategy towards the pure strategy which is returned as the best response to the
            # nash equilibrium that is being trained against.
            pop[k] = lr * br + (1 - lr) * pop[k]
            performance = pop[k] @ payoffs @ population_strategy.T + 1  # make it positive for pct calculation
            learner_performances[k].append(performance)

            # if the first learner plateaus, add a new policy to the population
            if j == num_learners - 1 and performance / learner_performances[k][-2] - 1 < improvement_pct_threshold:
                learner = np.random.uniform(0, 1, (1, dim))
                learner = learner / learner.sum(axis=1)[:, None]
                pop = np.vstack((pop, learner))
                learner_performances.append([0.1])

        # calculate exploitability for meta Nash of whole population
        exp = get_exploitability(pop, payoffs, iters=1000)
        exps.append(exp)

        M = pop @ payoffs @ pop.T
        L = M @ M.T
        l_card = np.trace(np.eye(L.shape[0]) - np.linalg.inv(L + np.eye(L.shape[0])))
        l_cards.append(l_card)

    return pop, exps, l_cards


# Define the self-play algorithm
def self_play_steps(iters=10, payoffs=None, verbose=False, improvement_pct_threshold=.03, lr=.2, seed=0):
    dim = payoffs.shape[0]
    r = np.random.RandomState(seed)
    pop = r.uniform(0, 1, (2, dim))
    pop = pop / pop.sum(axis=1)[:, None]
    exp = get_exploitability(pop, payoffs, iters=1000)
    exps = [exp]
    performances = [.01]

    M = pop @ payoffs @ pop.T
    L = M@M.T
    l_card = np.trace(np.eye(L.shape[0]) - np.linalg.inv(L + np.eye(L.shape[0])))
    l_cards = [l_card]

    for i in range(iters):
        if i % 10 == 0:
            print('iteration: ', i, 'exploitability: ', exps[-1])
        br = get_br_to_strat(pop[-2], payoffs=payoffs)
        pop[-1] = lr * br + (1 - lr) * pop[-1]
        performance = pop[-1] @ payoffs @ pop[-2].T + 1
        performances.append(performance)
        if performance / performances[-2] - 1 < improvement_pct_threshold:
            learner = np.random.uniform(0, 1, (1, dim))
            learner = learner / learner.sum(axis=1)[:, None]
            pop = np.vstack((pop, learner))
        exp = get_exploitability(pop, payoffs, iters=1000)
        exps.append(exp)

        M = pop @ payoffs @ pop.T
        L = M @ M.T
        l_card = np.trace(np.eye(L.shape[0]) - np.linalg.inv(L + np.eye(L.shape[0])))
        l_cards.append(l_card)

    return pop, exps, l_cards


# Define the PSRO rectified nash algorithm
def psro_rectified_steps(iters=10, payoffs=None, verbose=False, eps=1e-2, seed=0,
                         num_start_strats=1, num_pseudo_learners=4, lr=0.3, threshold=0.001):
    dim = payoffs.shape[0]
    r = np.random.RandomState(seed)
    pop = r.uniform(0, 1, (num_start_strats, dim))
    pop = pop / pop.sum(axis=1)[:, None]
    exp = get_exploitability(pop, payoffs, iters=1000)
    exps = [exp]
    counter = 0

    M = pop @ payoffs @ pop.T
    L = M @ M.T
    l_card = np.trace(np.eye(L.shape[0]) - np.linalg.inv(L + np.eye(L.shape[0])))
    l_cards = [l_card]

    while counter < iters * num_pseudo_learners:
        if counter % (5 * num_pseudo_learners) == 0:
            print('iteration: ', int(counter / num_pseudo_learners), ' exp: ', exps[-1])
            print('size of population: ', pop.shape[0])

        new_pop = np.copy(pop)
        emp_game_matrix = pop @ payoffs @ pop.T
        averages, _ = fictitious_play(payoffs=emp_game_matrix, iters=iters)

        # go through all policies. If the policy has positive meta Nash mass,
        # find policies it wins against, and play against meta Nash weighted mixture of those policies
        for j in range(pop.shape[0]):
            if counter > iters * num_pseudo_learners:
                return pop, exps, l_cards
            # if positive mass, add a new learner to pop and update it with steps, submit if over thresh
            # keep track of counter
            if averages[-1][j] > eps:
                # create learner
                learner = np.random.uniform(0, 1, (1, dim))
                learner = learner / learner.sum(axis=1)[:, None]
                new_pop = np.vstack((new_pop, learner))
                idx = new_pop.shape[0] - 1

                current_performance = 0.02
                last_performance = 0.01
                while current_performance / last_performance - 1 > threshold:
                    counter += 1
                    mask = emp_game_matrix[j, :]
                    mask[mask >= 0] = 1
                    mask[mask < 0] = 0
                    weights = np.multiply(mask, averages[-1])
                    weights /= weights.sum()
                    strat = weights @ pop
                    br = get_br_to_strat(strat, payoffs=payoffs)
                    new_pop[idx] = lr * br + (1 - lr) * new_pop[idx]
                    last_performance = current_performance
                    current_performance = new_pop[idx] @ payoffs @ strat + 1

                    if counter % num_pseudo_learners == 0:
                        # count this as an 'iteration'

                        # exploitability
                        exp = get_exploitability(new_pop, payoffs, iters=1000)
                        exps.append(exp)

                        M = pop @ payoffs @ pop.T
                        L = M @ M.T
                        l_card = np.trace(np.eye(L.shape[0]) - np.linalg.inv(L + np.eye(L.shape[0])))
                        l_cards.append(l_card)

        pop = np.copy(new_pop)

    return pop, exps, l_cards


def run_experiment(param_seed):
    params, seed = param_seed
    iters = params['iters']
    num_threads = params['num_threads']
    dim = params['dim']
    lr = params['lr']
    thresh = params['thresh']

    psro = params['psro']
    pipeline_psro = params['pipeline_psro']
    dpp_psro = params['dpp_psro']
    rectified = params['rectified']
    self_play = params['self_play']

    psro_exps = []
    psro_cardinality = []
    pipeline_psro_exps = []
    pipeline_psro_cardinality = []
    dpp_psro_exps = []
    dpp_psro_cardinality = []
    rectified_exps = []
    rectified_cardinality = []
    self_play_exps = []
    self_play_cardinality = []

    print('Experiment: ', seed + 1)
    np.random.seed(seed)
    W = np.random.randn(dim, dim)
    S = np.random.randn(dim, 1)
    payoffs = 0.5 * (W - W.T) + S - S.T

    if psro:
        print('PSRO')
        pop, exps, cards = psro_steps(iters=iters, num_learners=1, seed=seed+1,
                                                              improvement_pct_threshold=thresh, lr=lr,
                                                              payoffs=payoffs, loss_func='br')
        psro_exps = exps
        psro_cardinality = cards
    if pipeline_psro:
        print('Pipeline PSRO')
        pop, exps, cards = psro_steps(iters=iters, num_learners=num_threads, seed=seed+1,
                                                              improvement_pct_threshold=thresh, lr=lr,
                                                              payoffs=payoffs, loss_func='br')
        pipeline_psro_exps = exps
        pipeline_psro_cardinality = cards
    if dpp_psro:
        print('Pipeline DPP')
        pop, exps, cards = psro_steps(iters=iters, num_learners=num_threads, seed=seed+1,
                                                              improvement_pct_threshold=thresh, lr=lr,
                                                              payoffs=payoffs, loss_func='dpp')
        dpp_psro_exps = exps
        dpp_psro_cardinality = cards
    if rectified:
        print('Rectified')
        pop, exps, cards = psro_rectified_steps(iters=iters, num_pseudo_learners=num_threads, payoffs=payoffs, seed=seed+1,
                                         lr=lr, threshold=thresh)
        rectified_exps = exps
        rectified_cardinality = cards
    if self_play:
        print('Self-play')
        pop, exps, cards = self_play_steps(iters=iters, payoffs=payoffs, improvement_pct_threshold=thresh, lr=lr, seed=seed+1)
        self_play_exps = exps
        self_play_cardinality = cards


    return {
        'psro_exps': psro_exps,
        'psro_cardinality': psro_cardinality,
        'pipeline_psro_exps': pipeline_psro_exps,
        'pipeline_psro_cardinality': pipeline_psro_cardinality,
        'dpp_psro_exps': dpp_psro_exps,
        'dpp_psro_cardinality': dpp_psro_cardinality,
        'rectified_exps': rectified_exps,
        'rectified_cardinality': rectified_cardinality,
        'self_play_exps': self_play_exps,
        'self_play_cardinality': self_play_cardinality,
    }


def run_experiments(num_experiments=1, iters=40, num_threads=20, dim=60, lr=0.6, thresh=0.001, logscale=True,
                    psro=False,
                    pipeline_psro=False,
                    rectified=False,
                    self_play=False,
                    dpp_psro=False,
                    ):

    params = {
        'num_experiments': num_experiments,
        'iters': iters,
        'num_threads': num_threads,
        'dim': dim,
        'lr': lr,
        'thresh': thresh,
        'psro': psro,
        'pipeline_psro': pipeline_psro,
        'dpp_psro': dpp_psro,
        'rectified': rectified,
        'self_play': self_play,
    }

    psro_exps = []
    psro_cardinality = []
    pipeline_psro_exps = []
    pipeline_psro_cardinality = []
    dpp_psro_exps = []
    dpp_psro_cardinality = []
    rectified_exps = []
    rectified_cardinality = []
    self_play_exps = []
    self_play_cardinality = []

    with open(os.path.join(PATH_RESULTS, 'params.json'), 'w', encoding='utf-8') as json_file:
        json.dump(params, json_file, indent=4)

    pool = mp.Pool()
    result = pool.map(run_experiment, [(params, i) for i in range(num_experiments)])

    for r in result:
        psro_exps.append(r['psro_exps'])
        psro_cardinality.append(r['psro_cardinality'])
        pipeline_psro_exps.append(r['pipeline_psro_exps'])
        pipeline_psro_cardinality.append(r['pipeline_psro_cardinality'])
        dpp_psro_exps.append(r['dpp_psro_exps'])
        dpp_psro_cardinality.append(r['dpp_psro_cardinality'])
        rectified_exps.append(r['rectified_exps'])
        rectified_cardinality.append(r['rectified_cardinality'])
        self_play_exps.append(r['self_play_exps'])
        self_play_cardinality.append(r['self_play_cardinality'])

    d = {
        'psro_exps': psro_exps,
        'psro_cardinality': psro_cardinality,
        'pipeline_psro_exps': pipeline_psro_exps,
        'pipeline_psro_cardinality': pipeline_psro_cardinality,
        'dpp_psro_exps': dpp_psro_exps,
        'dpp_psro_cardinality': dpp_psro_cardinality,
        'rectified_exps': rectified_exps,
        'rectified_cardinality': rectified_cardinality,
        'self_play_exps': self_play_exps,
        'self_play_cardinality': self_play_cardinality,
    }
    pickle.dump(d, open(os.path.join(PATH_RESULTS, 'data.p'), 'wb'))

    def plot_error(data, label=''):
        data_mean = np.mean(np.array(data), axis=0)
        error_bars = stats.sem(np.array(data))
        plt.plot(data_mean, label=label)
        plt.fill_between([i for i in range(data_mean.size)],
                         np.squeeze(data_mean - error_bars),
                         np.squeeze(data_mean + error_bars), alpha=alpha)

    alpha = .4
    for j in range(2):
        fig_handle = plt.figure()

        if psro:
            if j == 0:
                plot_error(psro_exps, label='PSRO')
            elif j == 1:
                plot_error(psro_cardinality, label='PSRO')
        if pipeline_psro:
            if j == 0:
                plot_error(pipeline_psro_exps, label='P-PSRO')
            elif j == 1:
                plot_error(pipeline_psro_cardinality, label='P-PSRO')
        if rectified:
            if j == 0:
                length = min([len(l) for l in rectified_exps])
                for i, l in enumerate(rectified_exps):
                    rectified_exps[i] = rectified_exps[i][:length]
                plot_error(rectified_exps, label='PSRO-rN')
            elif j == 1:
                length = min([len(l) for l in rectified_cardinality])
                for i, l in enumerate(rectified_cardinality):
                    rectified_cardinality[i] = rectified_cardinality[i][:length]
                plot_error(rectified_cardinality, label='PSRO-rN')
        if self_play:
            if j == 0:
                plot_error(self_play_exps, label='Self-play')
            elif j == 1:
                plot_error(self_play_cardinality, label='Self-play')
        if dpp_psro:
            if j == 0:
                plot_error(dpp_psro_exps, label='Ours')
            elif j == 1:
                plot_error(dpp_psro_cardinality, label='Ours')


        plt.legend(loc="upper left")
        plt.title('Dim {:d}'.format(args.dim))

        if logscale and (j==0):
            plt.yscale('log')

        plt.savefig(os.path.join(PATH_RESULTS, 'figure_'+ str(j) + '.pdf'))


if __name__ == "__main__":
    run_experiments(num_experiments=10, num_threads=2, iters=args.nb_iters, dim=args.dim, lr=.5, thresh=TH,
                    psro=True,
                    pipeline_psro=True,
                    rectified=True,
                    self_play=True,
                    dpp_psro=True,
                    )


