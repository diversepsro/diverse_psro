import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import os
from scipy import stats
import pickle
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
np.random.seed(0)
from numpy.random import RandomState
import scipy.linalg as la
import copy

dim = 30
payoffs = np.tril(np.random.uniform(-1, 1, (dim,dim)), -1)
payoffs = (payoffs - payoffs.T)

LR = 0.1
TRAIN_ITERS = 5

expected_card = []
sizes = []

time_string = time.strftime("%Y%m%d-%H%M%S")
PATH_RESULTS = os.path.join('results', time_string)
os.mkdir(PATH_RESULTS)

device = 'cpu'

FILE_TRAJ = {
    'rectified': 'rectified.p',
    'psro': 'psro.p',
    'p-psro': 'p_psro.p',
    'dpp': 'dpp.p',
             }


class MyGaussianPDF(nn.Module):
    def __init__(self, mu):
        super(MyGaussianPDF, self).__init__()
        self.mu = mu
        self.cov = 0.54*torch.eye(2)
        # self.c = (1./(2*np.pi))
        self.c = 1.

    def forward(self, x):
        return self.c*torch.exp(-0.5*torch.diagonal( (x-self.mu)@self.cov@(x-self.mu).t() ))

class GMMAgent(nn.Module):
    def __init__(self, mu):
        super(GMMAgent, self).__init__()
        self.gauss = MyGaussianPDF(mu).to(device)
        self.x = nn.Parameter(0.01*torch.randn(2, dtype=torch.float), requires_grad=False)

    def forward(self):
        return self.gauss(self.x)

class TorchPop:
    def __init__(self, num_learners, seed=0):
        torch.manual_seed(seed)
        self.pop_size = num_learners + 1

        mus = np.array([[2.8722, -0.025255],
                        [1.8105, 2.2298],
                        [1.8105, -2.2298],
                        [-0.61450, 2.8058],
                        [-0.61450, -2.8058],
                        [-2.5768, 1.2690],
                        [-2.5768, -1.2690]]
                       )
        mus = torch.from_numpy(mus).float().to(device)
        self.mus = mus

        self.game = torch.from_numpy(np.array([
                                               [0., 1., 1., 1, -1, -1, -1],
                                               [-1., 0., 1., 1., 1., -1., -1.],
                                               [-1., -1., 0., 1., 1., 1., -1],
                                               [-1., -1., -1., 0, 1., 1., 1.],
                                               [1., -1., -1., -1., 0., 1., 1.],
                                               [1., 1., -1., -1, -1, 0., 1.],
                                               [1., 1., 1., -1., -1., -1., 0.]
                                               ])).float().to(device)

        self.pop = [GMMAgent(mus) for _ in range(self.pop_size)]
        self.pop_hist = [[self.pop[i].x.detach().cpu().clone().numpy()] for i in range(self.pop_size)]

    def visualise_pop(self, br=None, ax=None, color=None):

        def multivariate_gaussian(pos, mu, Sigma):
            """Return the multivariate Gaussian distribution on array pos."""

            n = mu.shape[0]
            Sigma_det = np.linalg.det(Sigma)
            Sigma_inv = np.linalg.inv(Sigma)
            N = np.sqrt((2 * np.pi) ** n * Sigma_det)
            # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
            # way across all the input variables.
            fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)
            return np.exp(-fac / 2) / N

        agents = [agent.x.detach().cpu().numpy() for agent in self.pop]
        agents = list(zip(*agents))

        # Colors
        if color is None:
            colors = cm.rainbow(np.linspace(0, 1, len(agents[0])))
        else:
            colors = [color]*len(agents[0])

        # fig = plt.figure(figsize=(6, 6))
        ax.scatter(agents[0], agents[1], alpha=1., marker='.', color=colors, s=8*plt.rcParams['lines.markersize'] ** 2)
        if br is not None:
            ax.scatter(br[0], br[1], marker='.', c='k')
        for i, hist in enumerate(self.pop_hist):
            if hist:
                hist = list(zip(*hist))
                ax.plot(hist[0], hist[1], alpha=0.8, color=colors[i], linewidth=4)

        # ax = plt.gca()
        for i in range(7):
            ax.scatter(self.mus[i, 0].item(), self.mus[i, 1].item(), marker='x', c='k')
            for j in range(4):
                delta = 0.025
                x = np.arange(-4.5, 4.5, delta)
                y = np.arange(-4.5, 4.5, delta)
                X, Y = np.meshgrid(x, y)
                pos = np.empty(X.shape + (2,))
                pos[:, :, 0] = X
                pos[:, :, 1] = Y
                Z = multivariate_gaussian(pos, self.mus[i,:].numpy(), 0.54 * np.eye(2))
                levels = 10
                # levels = np.logspace(0.01, 1, 10, endpoint=True)
                CS = ax.contour(X, Y, Z, levels, colors='k', linewidths=0.5, alpha=0.2)
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
                # ax.clabel(CS, fontsize=9, inline=1)
                # circle = plt.Circle((0, 0), 0.2, color='r')
                # ax.add_artist(circle)
        ax.set_xlim([-4.5, 4.5])
        ax.set_ylim([-4.5, 4.5])


    def get_payoff(self, agent1, agent2):
        p = agent1()
        q = agent2()
        return p @ self.game @ q + 0.5*(p-q).sum()

    def get_payoff_aggregate(self, agent1, metanash, K):
        # Computes the payoff of agent1 against the aggregated first :K agents using metanash as weights
        agg_agent = metanash[0]*self.pop[0]()
        for k in range(1, K):
            agg_agent += metanash[k]*self.pop[k]()
        return agent1() @ self.game @ agg_agent + 0.5*(agent1()-agg_agent).sum()

    def get_payoff_aggregate_weights(self, agent1, weights, K):
        # Computes the payoff of agent1 against the aggregated first :K agents using metanash as weights
        agg_agent = weights[0]*self.pop[0]()
        for k in range(1, len(weights)):
            agg_agent += weights[k]*self.pop[k]()
        return agent1() @ self.game @ agg_agent + 0.5*(agent1()-agg_agent).sum()

    def get_br_to_strat(self, metanash, lr, nb_iters=20):
        br = GMMAgent(self.mus)
        br.x = nn.Parameter(0.1*torch.randn(2, dtype=torch.float), requires_grad=False)
        br.x.requires_grad = True
        optimiser = optim.Adam(br.parameters(), lr=lr)
        for _ in range(nb_iters*10):
            loss = -self.get_payoff_aggregate(br, metanash, self.pop_size,)
            # Optimise !
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        return br

    def get_metagame(self, k=None, numpy=False):
        if k==None:
            k = self.pop_size
        if numpy:
            with torch.no_grad():
                metagame = torch.zeros(k, k)
                for i in range(k):
                    for j in range(k):
                        metagame[i, j] = self.get_payoff(self.pop[i], self.pop[j])
                return metagame.detach().cpu().clone().numpy()
        else:
            metagame = torch.zeros(k, k)
            for i in range(k):
                for j in range(k):
                    metagame[i, j] = self.get_payoff(self.pop[i], self.pop[j])
            return metagame

    def add_new(self):
        with torch.no_grad():
            self.pop.append(GMMAgent(self.mus))
            self.pop_hist.append([self.pop[-1].x.detach().cpu().clone().numpy()])
            self.pop_size += 1

    def get_exploitability(self, metanash, lr, nb_iters=20):
        br = self.get_br_to_strat(metanash, lr, nb_iters=nb_iters)
        with torch.no_grad():
            exp = self.get_payoff_aggregate(br, metanash, self.pop_size).item()
        return exp


def gradient_loss_update(torch_pop, k, train_iters=10, lambda_weight=0.1, lr=0.1, dpp=True):

    # We compute metagame M and then L in a differentiable way
    # We compute expected payoff of agent k-1 against aggregated strat

    # Make strategy k trainable
    torch_pop.pop[k].x.requires_grad = True

    # Optimiser
    optimiser = optim.Adam(torch_pop.pop[k].parameters(), lr=lr)

    for iter in range(train_iters):

        # Get metagame and metastrat
        M = torch_pop.get_metagame(k=k+1)
        meta_nash = fictitious_play(payoffs=M.detach().cpu().clone().numpy()[:k, :k], iters=1000)[0][-1]

        # Compute cardinality of pop up until :k UNION training strategy. We use payoffs as features.
        if dpp:
            M =  f.normalize(M,dim=1,p=2) #  Normalise
            L = M @ M.t()  # Compute kernel
            L_card = torch.trace(torch.eye(L.shape[0]) - torch.inverse(L + torch.eye(L.shape[0])))  # Compute cardinality

            # Compute the expected return given that enemy plays agg_strat (using :k first strats)
            exp_payoff = torch_pop.get_payoff_aggregate(torch_pop.pop[k], meta_nash, k)

            # Loss
            loss = -(lambda_weight * exp_payoff + (1. - lambda_weight) * L_card)
        else:
            with torch.no_grad():
                M =  f.normalize(M,dim=1,p=2) #  Normalise
                L = M @ M.t()  # Compute kernel
                L_card = torch.trace(torch.eye(L.shape[0]) - torch.inverse(L + torch.eye(L.shape[0])))  # Compute cardinality

            # Compute the expected return given that enemy plays agg_strat (using :k first strats)
            exp_payoff = torch_pop.get_payoff_aggregate(torch_pop.pop[k], meta_nash, k)

            # Loss
            loss = -(lambda_weight * exp_payoff)

        # Optimise !
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    torch_pop.pop_hist[k].append(torch_pop.pop[k].x.detach().cpu().clone().numpy())

    # Make strategy k non-trainable
    torch_pop.pop[k].x.requires_grad = False
    return exp_payoff.item(), L_card.item()


def psro_gradient(iters=5, num_learners=4, lr=.2, train_iters=10, dpp=True, seed=0):

    # Generate population
    torch_pop = TorchPop(num_learners, seed=seed)

    # Compute initial exploitability and init stuff
    metagame = torch_pop.get_metagame(numpy=True)
    metanash = fictitious_play(payoffs=metagame, iters=1000)[0][-1]
    exp = torch_pop.get_exploitability(metanash, lr, nb_iters=train_iters)
    exps = [exp]
    L_card = 0.
    L_cards = []

    for i in range(iters):
        # Define the weighting towards diversity
        lambda_weight = 0. + (0.7 / (1 + np.exp(-0.25*(i-(25)))) )
        lambda_weight = 1. - lambda_weight
        for j in range(num_learners):
            # first learner (when j=num_learners-1) plays against normal meta Nash
            # second learner plays against meta Nash with first learner included, etc.
            k = torch_pop.pop_size - j - 1

            # Diverse PSRO
            exp_payoff, L_card = gradient_loss_update(torch_pop, k, train_iters=train_iters, lr=lr,
                                                      lambda_weight=lambda_weight, dpp=dpp)
            if j == num_learners - 1:
                torch_pop.add_new()


        metagame = torch_pop.get_metagame(numpy=True)
        metanash = fictitious_play(payoffs=metagame, iters=1000)[0][-1]
        exp = torch_pop.get_exploitability(metanash, lr, nb_iters=train_iters)
        exps.append(exp)
        L_cards.append(L_card)

        if i % 1 == 0:
            print('ITERATION: ', i, ' exp full: {:.4f}'.format(exps[-1]), 'L_CARD: {:.3f}'.format(L_cards[-1]),
                  'lw: {:.3f}'.format(lambda_weight))

    fig1, axs1 = plt.subplots(1, 1)
    torch_pop.visualise_pop(br=None, ax=axs1)

    if num_learners==1:
        fstr = 'psro'
    else:
        fstr = 'dppLoss_' if dpp else 'origLoss'
    plt.savefig(os.path.join(PATH_RESULTS, 'trajectories_' + fstr + '.pdf'))

    return torch_pop, exps, L_cards


def gradient_loss_update_rectified(torch_pop, k, weights, train_iters=10, lr=0.1):

    # Make strategy k trainable
    torch_pop.pop[k].x.requires_grad = True

    # Optimiser
    optimiser = optim.Adam(torch_pop.pop[k].parameters(), lr=lr)

    for iter in range(train_iters):

        # Get metagame and metastrat
        M = torch_pop.get_metagame(k=k+1)

        # Compute cardinality of pop up until :k UNION training strategy. We use payoffs as features.
        with torch.no_grad():
            M =  f.normalize(M,dim=1,p=2) #  Normalise
            L = M @ M.t()  # Compute kernel
            L_card = torch.trace(torch.eye(L.shape[0]) - torch.inverse(L + torch.eye(L.shape[0])))  # Compute cardinality

        # Compute the expected return given that enemy plays agg_strat (using :k first strats)
        exp_payoff = torch_pop.get_payoff_aggregate_weights(torch_pop.pop[k], weights, k)

        # Loss
        loss = -exp_payoff

        # Optimise !
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    torch_pop.pop_hist[k].append(torch_pop.pop[k].x.detach().cpu().clone().numpy())

    # Make strategy k non-trainable
    torch_pop.pop[k].x.requires_grad = False
    return exp_payoff.item(), L_card.item()


# Define the PSRO rectified nash algorithm
def psro_rectified_gradient(iters=10, eps=1e-2, seed=0, train_iters=10,
                         num_pseudo_learners=4, lr=0.3):
    # Generate population
    torch_pop = TorchPop(num_pseudo_learners, seed=seed)

    # Compute initial exploitability and init stuff
    metagame = torch_pop.get_metagame(numpy=True)
    metanash = fictitious_play(payoffs=metagame, iters=1000)[0][-1]
    exp = torch_pop.get_exploitability(metanash, lr, nb_iters=train_iters)
    exps = [exp]
    L_cards = []

    counter = 0
    while counter < iters * num_pseudo_learners:
        if counter % (5 * num_pseudo_learners) == 0:
            print('iteration: ', int(counter / num_pseudo_learners), ' exp: ', exps[-1])
            print('size of population: ', torch_pop.pop_size)

        new_pop = copy.deepcopy(torch_pop)
        emp_game_matrix = torch_pop.get_metagame(numpy=True)
        averages, _ = fictitious_play(payoffs=emp_game_matrix, iters=iters)

        # go through all policies. If the policy has positive meta Nash mass,
        # find policies it wins against, and play against meta Nash weighted mixture of those policies
        for j in range(torch_pop.pop_size):
            if counter > iters * num_pseudo_learners:
                fig1, axs1 = plt.subplots(1, 1)
                torch_pop.visualise_pop(br=None, ax=axs1)
                plt.savefig(os.path.join(PATH_RESULTS, 'trajectories_rectified.pdf'))
                return torch_pop, exps, L_cards
            # if positive mass, add a new learner to pop and update it with steps, submit if over thresh
            # keep track of counter
            if averages[-1][j] > eps:
                # create learner
                new_pop.add_new()
                idx = new_pop.pop_size - 1
                counter += 1
                print(counter)

                mask = emp_game_matrix[j, :]
                mask += 1e-5
                mask[mask >= 0] = 1
                mask[mask < 0] = 0
                weights = np.multiply(mask, averages[-1])
                weights /= weights.sum()

                exp_payoff, L_card = gradient_loss_update_rectified(new_pop, idx, weights,
                                                                    train_iters=train_iters, lr=lr)

                if counter % num_pseudo_learners == 0:
                    metagame = new_pop.get_metagame(numpy=True)
                    metanash = fictitious_play(payoffs=metagame, iters=1000)[0][-1]
                    exp = new_pop.get_exploitability(metanash, lr, nb_iters=train_iters)
                    exps.append(exp)
                    L_cards.append(L_card)
        torch_pop = copy.deepcopy(new_pop)


    fig1, axs1 = plt.subplots(1, 1)
    torch_pop.visualise_pop(br=None, ax=axs1)
    plt.savefig(os.path.join(PATH_RESULTS, 'trajectories_rectified.pdf'))


    return torch_pop, exps, L_cards


#Search over the pure strategies to find the BR to a strategy
def get_br_to_strat(strat, payoffs=payoffs, verbose=False):
    row_weighted_payouts = strat@payoffs
    br = np.zeros_like(row_weighted_payouts)
    br[np.argmin(row_weighted_payouts)] = 1
    if verbose:
        print(row_weighted_payouts[np.argmin(row_weighted_payouts)], "exploitability")
    return br


#Fictituous play as a nash equilibrium solver
def fictitious_play(iters=2000, payoffs=payoffs, verbose=False):
    dim = payoffs.shape[0]
    pop = np.random.uniform(0,1,(1,dim))
    pop = pop/pop.sum(axis=1)[:,None]
    averages = pop
    exps = []
    for i in range(iters):
        average = np.average(pop, axis=0)
        br = get_br_to_strat(average, payoffs=payoffs)
        exp1 = average@payoffs@br.T
        exp2 = br@payoffs@average.T
        exps.append(exp2-exp1)
        # if verbose:
        #     print(exp, "exploitability")
        averages = np.vstack((averages, average))
        pop = np.vstack((pop, br))
    return averages, exps


def run_experiments(num_experiments=1, num_threads=20, iters=40,
                    rectified=False, psro=False,
                    pipeline_psro=False, dpp_psro=False,
                    yscale='none', verbose=False, train_iters=10):

    rectified_exps = []
    rectified_cardinality = []

    psro_exps = []
    psro_cardinality = []

    pipeline_exps = []
    pipeline_cardinality = []

    dpp_exps = []
    dpp_cardinality = []

    for i in range(num_experiments):
        print('Experiment: ', i + 1)

        if rectified:
            print('Rectified')
            torch_pop, exps, L_cards = psro_rectified_gradient(iters=iters, seed=i, train_iters=train_iters,
                                    num_pseudo_learners=1, lr=LR)
            rectified_exps.append(exps)
            rectified_cardinality.append(L_cards)
            pickle.dump({'pop': torch_pop}, open(os.path.join(PATH_RESULTS, FILE_TRAJ['rectified'])+'.p', 'wb'))

        if dpp_psro:
            print('Grad DPP')
            torch_pop, exps, L_cards = psro_gradient(iters=iters, num_learners=num_threads, lr=LR, train_iters=train_iters, seed=i,
                                          dpp=True)
            dpp_exps.append(exps)
            dpp_cardinality.append(L_cards)
            pickle.dump({'pop': torch_pop}, open(os.path.join(PATH_RESULTS, FILE_TRAJ['dpp'])+'.p', 'wb'))

        if pipeline_psro:
            print('Grad no DPP')
            torch_pop, exps, L_cards = psro_gradient(iters=iters, num_learners=num_threads, lr=LR, train_iters=train_iters, seed=i,
                                          dpp=False)
            pipeline_exps.append(exps)
            pipeline_cardinality.append(L_cards)
            pickle.dump({'pop': torch_pop}, open(os.path.join(PATH_RESULTS, FILE_TRAJ['p-psro'])+'.p', 'wb'))

        if psro:
            print('PSRO no DPP')
            torch_pop, exps, L_cards = psro_gradient(iters=iters, num_learners=1, lr=LR, train_iters=train_iters, seed=i,
                                          dpp=False)
            psro_exps.append(exps)
            psro_cardinality.append(L_cards)
            pickle.dump({'pop': torch_pop}, open(os.path.join(PATH_RESULTS, FILE_TRAJ['psro'])+'.p', 'wb'))


        d = {
            'rectified_exps':rectified_exps,
            'rectified_cardinality':rectified_cardinality,
            'pipeline_exps':pipeline_exps,
            'pipeline_cardinality':pipeline_cardinality,
            'dpp_exps':dpp_exps,
            'dpp_cardinality':dpp_cardinality,
            'psro_exps':psro_exps,
            'psro_cardinality':psro_cardinality,
         }
        pickle.dump(d, open(os.path.join(PATH_RESULTS, 'checkpoint_'+str(i)), 'wb'))

    def plot_error(data, label=''):
        avg = np.mean(np.array(data), axis=0)
        error_bars = stats.sem(np.array(data))
        plt.plot(avg, label=label)
        plt.fill_between([i for i in range(avg.shape[0])],
                         avg - error_bars,
                         avg + error_bars, alpha=alpha)

    num_plots = 2

    alpha = .4
    for j in range(num_plots):
        fig_handle = plt.figure()
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

        if psro:
            if j == 0:
                plot_error(pipeline_exps, label='PSRO')
            elif j == 1:
                plot_error(pipeline_cardinality, label='PSRO')

        if pipeline_psro:
            if j == 0:
                plot_error(pipeline_exps, label='P-PSRO')
            elif j == 1:
                plot_error(pipeline_cardinality, label='P-PSRO')

        if dpp_psro:
            if j == 0:
                plot_error(dpp_exps, label='Ours (DPP Loss)')
            elif j == 1:
                plot_error(dpp_cardinality, label='Ours (DPP Loss)')

        plt.legend(loc="upper left")

        if yscale == 'both':
            if j == 0:
                plt.yscale('log')
        elif yscale == 'log':
            plt.yscale('log')


        plt.savefig(os.path.join(PATH_RESULTS, 'figure_'+ str(j) + '.pdf'))

def run_traj():

    titles = {
        'rectified': 'PSRO-rN',
        'dpp': 'DPP-PSRO',
        'p-psro': 'P-PSRO',
        'psro': 'PSRO',
    }
    pops = {}
    fig1, axs1 = plt.subplots(1, 4, figsize=(5 * 4, 5 * 1), dpi=200)
    axs1 = axs1.flatten()
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for i, key in enumerate(FILE_TRAJ.keys()):
        ax = axs1[i]
        d = pickle.load(open(os.path.join(PATH_RESULTS, FILE_TRAJ[key])+'.p', 'rb'))
        pops[FILE_TRAJ[key]] = d['pop']
        pops[FILE_TRAJ[key]].visualise_pop(ax=ax, color=colors[i])
        ax.set_title(titles[key])

    fig1.tight_layout()
    fig1.savefig(os.path.join(PATH_RESULTS, 'trajectories.pdf'))


if __name__ =="__main__":

    run_experiments(num_experiments=10, num_threads=4, iters=50,
                    pipeline_psro=True,
                    dpp_psro=True,
                    rectified=True,
                    psro=True,
                    yscale='none', train_iters=TRAIN_ITERS)
    run_traj()

    plt.show()




