# Greedy online identification of linear dynamic systems

Implementation of our greedy algorithm along with baselines.

## Example

```python

from agents import Random, Greedy, Gradient

def dynamics_step(x, u):
            noise = sigma * np.random.randn(d)
            return A_star@x + B@u + noise

agent_ = Greedy  # or Random, or Gradient

agent = agent_(
    dynamics_step,
    B,
    gamma,
    sigma,
    prior_estimate,
    prior_moments,
    )

sample_estimation_values = np.array(agent.identify(T))

```