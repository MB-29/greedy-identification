import numpy as np
from numpy.linalg import norm
from scipy.linalg import solve
import torch

from quadratic import *
from planning import Planning


class Agent:
    """ Models a learner or controller, playing inputs to identify a
        the dynamic system.
        
        This codes is adapted to the identification of matrix A.
    """

    def __init__(self, dynamics_step, B, gamma, sigma, A_0, M_0, x0=None):
        """

        :param dynamics_step: map of the dynamics, with unknown parameter 
        :type dynamics_step: funciton
        :param B: Control matrix.
        :type B: size d x m numpy array
        :param gamma: gamma**2 is the maximal power.
        :type gamma: float
        :param sigma: Size of the noise.
        :type sigma: float
        :param A_0: First guess of A.
        :type A_0: size d x d numpy array
        :param M_0: Per row initial moment matrices.
        :type M_0: size d x d x d numpy array
        """
        

        self.dynamics_step = dynamics_step
        self.B = B

        self.gamma = gamma
        self.sigma = sigma

        self.d = A_0.shape[0]
        _, self.m = B.shape

        self.A_t = A_0.copy()  # shape [d, d]
        self.M_t = M_0.copy()  # shape [d, d, d]


        x = x0 if x0 is not None else np.zeros(self.d)


        self.x = x


    def draw_random_control(self):
        u = np.random.randn(self.m)
        u *= self.gamma/norm(u)
        return u
    

    def identify(self, T, A_star=None):
        """System identification process

        :param T: Time horizon, i.e. total number of observations
        of the true system.
        :type T: int
        :param A_star: True dynamics for the oracle. If provided,
        this ground truth matrix will be used for planning.
        Note that the matrix used for the observation of the true system
        is contained in self.dynamics_step.
        Defaults to None
        :type A_star: size d x d numpy array, optional
        :return: Estimates of A_star over time
        :rtype: size T x d x d numpy array
        """

        self.A_star = A_star

        self.A_t_values = np.zeros((T+1, self.d, self.d))
        self.A_t_values[0] = self.A_t
        self.u_values = np.zeros((T, self.m))
        
        self.x_values = torch.zeros(T+1, self.d)
        for t in range(T):

            u_t = self.choose_control(t)

            x_t_ = self.dynamics_step(self.x, u_t)

            self.online_OLS(self.x, x_t_, u_t)
            self.x = x_t_
            self.x_values[t+1] = torch.tensor(x_t_)

            self.A_t_values[t+1] = self.A_t.copy()
            self.u_values[t] = u_t

            M = self.M_t.mean(axis=0)
            S, _ = np.linalg.eig(M)

        return self.A_t_values

    def online_OLS(self, x_t, x_t_, u_t):
        """Online least squares estimation of A.

        :param x_t: Current state.
        :type x_t: size d numpy array
        :param x_t_: Next state.
        :type x_t_: size d numpy array
        :param u_t: Played input.
        :type u_t: size m numpy array
        """

        y_t = x_t_ - self.B@u_t

        for row_index in range(self.d):
            prior_moments = self.M_t[row_index]
            prior_estimate = self.A_t[row_index]
            posterior_moments = prior_moments + x_t[:, None]@x_t[None, :]
            combination = prior_moments@prior_estimate + y_t[row_index]*x_t
            if np.allclose(posterior_moments, 0, atol=1e-4):
                continue

            posterior_estimate = solve(posterior_moments, combination)

            self.M_t[row_index] = posterior_moments
            self.A_t[row_index] = posterior_estimate


class Random(Agent):
    """This agent plays random inputs with maximal power.
    """

    def choose_control(self, t):
        return self.draw_random_control()


class Greedy(Agent):
    """This agent plays one-step-ahead-optimal designs.
    """

    def choose_control(self, t):
        M = self.M_t.mean(axis=0)
        if  np.allclose(M, 0):
            return self.draw_random_control()
        A = self.A_t_values[t] if self.A_star is None else self.A_star
        u = greedy_optimal_input(M, A, self.B, self.x, self.gamma)
        u *= self.gamma / norm(u)
        return u

class Gradient(Agent):
    """This agent optimizes the inputs over epochs by projected gradient descent.
    """

    def plan(self, A, T, Xt, n_gradient, batch_size):
        print(f'planning T = {T}')

        A, B = torch.tensor(A, dtype=torch.float), torch.tensor(
            self.B, dtype=torch.float)
        self.planning = Planning(A, B, T, self.gamma, self.sigma, Xt, 'A-optimality')
        self.planning.plan(n_gradient, batch_size)

        return self.planning.U.clone().detach().numpy()

    def identify(self, T, n_gradient, batch_size, A_star=None, schedule=None):
        self.schedule = schedule
        self.n_gradient = n_gradient
        self.batch_size = batch_size
        self.i = 0

        self.U_values = np.zeros((n_gradient, T, self.m))
        
        return super().identify(T, A_star)

    def choose_control(self, t):

        if t < self.schedule[1]:
            return self.draw_random_control()

        A = self.A_t_values[t] if self.A_star is None else self.A_star
        ti_ = self.schedule[self.i+1]
        if t == ti_:
            self.i += 1
            ti__ = self.schedule[self.i+1]
            Xt = self.x_values[:t]
            self.U = self.plan(A, ti__, Xt, self.n_gradient, self.batch_size)

        ti = self.schedule[self.i]

        for gradient_step in range(self.n_gradient):
            u_gradient = self.planning.U_values[gradient_step][t-ti].numpy()
            self.U_values[gradient_step, t] = u_gradient


        return self.U[t-ti]


    