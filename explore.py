import numpy as np
import matplotlib.pyplot as plt

from agent import Agent
from utils import generate_random_A

A = np.array([
    [0.9, 1, 0, 0],
    [0, 0.9, 1, 0],
    [0, 0, 0.9, 1],
    [0, 0, 0, 0.9]
])
A = np.array([
    [-0.2, -.171, .001, .179],
    [0, 0.9, 1, 0],
    [.0984, 0, -1.588, -.562],
    [-1.131, 0, 1, -.737]
])
# A = 0.9*generate_random_A(4)
d, _ = A.shape
B = np.eye(d, d)

prior_mean = np.zeros((d, d))
prior_precision = np.zeros((d, d, d))
for j in range(d):
    # prior_precision[j] =  np.diag(abs(np.random.randn(d)))
    prior_precision[j] =  np.eye(d) + 0.01*np.diag(abs(np.random.randn(d)))


sigma = 1
gamma = 100
T = 100


def dynamics_step(x, u):
    assert (np.linalg.norm(u) <= gamma *(1.1))
    noise = sigma * np.random.randn()
    return A@x + B@u + noise

agent = Agent(dynamics_step,
        B,
        gamma,
        sigma,
        prior_mean,
        prior_precision,
        # method='linear-BOD',
        method='random'
    )

sample_estimation_values = np.array(agent.identify(T))
sample_residual_values = sample_estimation_values - A
sample_error_values = np.linalg.norm(sample_residual_values, axis=(1, 2))
error_values = sample_error_values
# mean_error = np.mean(error_values, axis=0)
# yerr = np.sqrt(2*np.var(error_values, axis=0)/n_samples)
# plt.errorbar(np.arange(T), mean_error, yerr=yerr, label=control_method, alpha=0.7)
plt.plot(error_values)

# S_values = np.array(agent.S_values)[:, 0] / np.cumsum((np.arange(1,T)))
# # plt.plot(S_values[:, 0])
# plt.plot(S_values)
# plt.legend()
plt.yscale('log')
plt.show()


