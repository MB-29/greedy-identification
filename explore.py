import numpy as np
import matplotlib.pyplot as plt

from agent import Agent

A = np.array([
    [0.9, 1, 0, 0],
    [0, 0.9, 1, 0],
    [0, 0, 0.9, 1],
    [0, 0, 0, 0.9]
])
d, _ = A.shape
B = np.eye(d, d)

prior_mean = np.zeros((d, d))
prior_precision = np.zeros((d, d, d))
for j in range(d):
    prior_precision[j] =  np.diag(abs(np.random.randn(d)))


sigma = 1
gamma = np.sqrt(1000)
T = 25000
n_samples = 10


def dynamics_step(x, u):
    assert (np.linalg.norm(u) <= gamma *(1.1))
    noise = sigma * np.random.randn()
    return A@x + B@u + noise

for control_method in ['linear-BOD', 'random']:
    error_values = np.zeros((n_samples, T))
    for sample_index in range(n_samples):
        agent = Agent(dynamics_step,
                B,
                gamma,
                sigma,
                prior_mean,
                prior_precision,
                control_method
            )

        sample_estimation_values = np.array(agent.identify(T))
        sample_residual_values = sample_estimation_values - A
        sample_error_values = np.linalg.norm(sample_residual_values, axis=(1, 2))
        error_values[sample_index, :] = sample_error_values
    mean_error = np.mean(error_values, axis=0)
    yerr = np.sqrt(2*np.var(error_values, axis=0)/n_samples)
    plt.errorbar(np.arange(T), mean_error, yerr=yerr, label=control_method, alpha=0.7)
plt.legend()
plt.yscale('log')
plt.show()
