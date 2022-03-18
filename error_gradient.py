import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import time

from planning import test
from agents import Greedy, Random, Offline, Gradient
from utils import generate_random_A

d = 4
m = 4
sigma = 1e-2
gamma = 1
n_samples = 4000
rho = 0.9

n_gradient, batch_size = 150, 200
test_batch_size = 100


if __name__ == '__main__':
    task_id = int(sys.argv[1])
    T = 20 + task_id * 4
    output = {
        'sigma': sigma,
        'gamma': gamma,
        'T': T,
        'n_gradient': n_gradient,
        'batch_size': batch_size
        }

    error = np.zeros((n_samples, n_gradient))
    time_values = np.zeros(n_samples)
    for sample_index in range(n_samples):

        A = rho*generate_random_A(d)
        B = np.eye(d, m)
        A_, B_ = torch.tensor(A, dtype=torch.float), torch.tensor(
            B, dtype=torch.float)
        # B = np.random.randn(d, m)

        def dynamics_step(x, u):
            # assert (np.linalg.norm(u) <= gamma *(1.1))
            noise = sigma * np.random.randn(d)
            return A@x + B@u + noise

        prior_estimate = np.zeros((d, d))
        A_star = A.copy()

        prior_moments = np.zeros((d, d, d))
        for j in range(d):
            #     # prior_moments[j] =  np.eye(d) + 0.01*np.diag(abs(np.random.randn(d)))
            prior_moments[j] = 1e-6*np.diag(abs(np.random.randn(d)))
        # prior_moments[0] =  0.01*np.diag(abs(np.random.randn(d)))

        agent = Gradient(
            dynamics_step,
            B,
            gamma,
            sigma,
            prior_estimate,
            prior_moments,
        )
        x0 = torch.zeros(1, d)
        schedule = [0, T//10, T]
        start_time = time.time()
        agent.identify(T, n_gradient, batch_size,
                       A_star=None, schedule=schedule)
        stop_time = time.time()
        duration = stop_time - start_time
        time_values[sample_index] = duration

        U_values = agent.U_values
        sample_error_values = np.zeros(n_gradient)
        for gradient_step in range(n_gradient):
            U = torch.tensor(U_values[gradient_step], dtype=torch.float)
            x0 = torch.zeros(test_batch_size, d)
            sample_batch_error = test(x0, A_, B_, U, T, sigma)
            sample_error_values[gradient_step] = sample_batch_error.mean()
        error[sample_index] = sample_error_values

    output['error'] = error
    output['time'] = time_values

    output_name = f'T-{T}_gradient-{n_gradient}_{n_samples}-samples_{task_id}'

    with open(f'{output_name}.pkl', 'wb') as f:
        pickle.dump(output, f)

# error_values[index, :] = sample_error_values
# mean_error = np.mean(error, axis=0)
# yerr = np.sqrt(2*np.var(error, axis=0)/n_samples)
# plt.errorbar(np.arange(n_gradient), mean_error, yerr=yerr, alpha=0.7)
# # plt.plot(error_values)
# plt.legend()
# plt.yscale('log')
# plt.show()
