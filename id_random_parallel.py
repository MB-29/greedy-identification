import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle 
import torch

from agents import Greedy, Random, Offline, Gradient
from utils import generate_random_A

d = 4
m = 4
sigma = 1e-2
gamma = 1
T = 100
n_samples = 500
rho = 0.9

n_gradient, batch_size = 120, 100

agent_name = 'noise'
agent_name = 'offline'
agent_name = 'sequential'
# agent_name = 'gradient'
agent_types = {'noise': Random, 'offline':Offline, 'gradient': Gradient, 'sequential': Greedy}
agent_ = agent_types[agent_name]


if __name__ == '__main__':
    task_id = int(sys.argv[1])

    output = {'sigma':sigma, 'gamma':gamma, 'T':T}

    residuals = np.zeros((n_samples, T+1, d, d))
    for sample_index in range(n_samples):

        A = rho*generate_random_A(d)
        B = np.eye(d, m)
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

        agent = agent_(
                dynamics_step,
                B,
                gamma,
                sigma,
                prior_estimate,
                prior_moments,
            )
        if agent_name=='gradient':
            x0 = torch.zeros(1, d)
            schedule = [0, T//10, T]
            sample_estimation_values = agent.identify(T, n_gradient, batch_size, A_star=None, schedule=schedule)
            sample_residual_values = sample_estimation_values - A
            residuals[sample_index] = sample_residual_values
            output['residuals'] = residuals
            continue

        sample_estimation_values = agent.identify(T)
        # sample_estimation_values = agent.identify(T, A_star)
        sample_residual_values = sample_estimation_values - A
        residuals[sample_index] = sample_residual_values
        # sample_error_values = np.linalg.norm(sample_residual_values, axis=(1, 2))
        # error_values[sample_index, :] = sample_error_values

    # output['error_values'] = error_values
    output['residuals'] = residuals

    output_name = f'{agent_name}_T-{T}_{n_samples}-samples_{task_id}'

    # with open(f'{output_name}.pkl', 'wb') as f:
    #     pickle.dump(output, f)

error_values = np.linalg.norm(residuals, axis=(2, 3), ord='fro')
# error_values[index, :] = sample_error_values
mean_error = np.mean(error_values, axis=0)
print(f'mean error {mean_error[-1]:3e}')
yerr = np.sqrt(2*np.var(error_values, axis=0)/n_samples)
plt.errorbar(np.arange(T+1), mean_error, yerr=yerr, alpha=0.7)
# plt.plot(error_values)
plt.legend()
plt.yscale('log')
plt.show()
