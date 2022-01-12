import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle

from agent import Agent
from utils import generate_random_A

d = 4
sigma = 1
gamma = np.sqrt(1000)
T = 25
n_samples = 10




if __name__ == '__main__':
    task_id = int(sys.argv[1])

    output = {'sigma':sigma, 'gamma':gamma, 'T':T}

    residuals = np.zeros((n_samples, T, d, d))
    for sample_index in range(n_samples):

        A = generate_random_A(d)
        B = np.eye(d, d)
        def dynamics_step(x, u):
            assert (np.linalg.norm(u) <= gamma *(1.1))
            noise = sigma * np.random.randn(d)
            return A@x + B@u + noise



        prior_mean = np.zeros((d, d))
        prior_precision = np.zeros((d, d, d))
        for j in range(d):
            # prior_precision[j] =  np.diag(abs(np.random.randn(d)))
            prior_precision[j] =  np.eye(d) + 0.01*np.diag(abs(np.random.randn(d)))

        agent = Agent(
                dynamics_step,
                B,
                gamma,
                sigma,
                prior_mean,
                prior_precision,
                method='linear-BOD'
            )

        sample_estimation_values = np.array(agent.identify(T))
        sample_residual_values = sample_estimation_values - A
        residuals[sample_index] = sample_residual_values
        # sample_error_values = np.linalg.norm(sample_residual_values, axis=(1, 2))
        # error_values[sample_index, :] = sample_error_values

        # output['error_values'] = error_values
        output['residuals'] = residuals

        output_name = f'online_{n_samples}-samples_{task_id}'

        with open(f'{output_name}.pkl', 'wb') as f:
            pickle.dump(output, f)

