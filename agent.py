import numpy as np
from numpy.linalg import norm
from scipy.linalg import solve
import torch

from quadratic import *
from planning import Planning


class Agent:

    def __init__(self, dynamics_step, B, gamma, sigma, A_0, M_0, x0=None):

        self.dynamics_step = dynamics_step
        self.B = B

        self.gamma = gamma
        self.sigma = sigma

        self.d = A_0.shape[0]
        _, self.m = B.shape

        self.A_t = A_0.copy()  # shape [d, d]
        self.M_t = M_0.copy()  # shape [d, d, d]

        u_0 = self.draw_random_control()

        x = x0 if x0 is not None else np.zeros(self.d)

        # x_1 = dynamics(x_0, u_0)
        # self.online_OLS(x_0, x_1, u_0)

        # self.x, self.u = x_1, u_0

        self.x = x

        # control_methods = {
        #     'OD': self.one_step_planning,
        #     'random': self.draw_random_control
        # }
        # self.choose_control = control_methods[method]

    def draw_random_control(self):
        u = np.random.randn(self.m)
        u *= self.gamma/norm(u)
        return u
    
    def plan(self, T):
        return

    def identify(self, T, A_star=None):

        self.A_star = A_star

        self.A_t_values = np.zeros((T+1, self.d, self.d))
        self.A_t_values[0] = self.A_t
        self.u_values = np.zeros((T, self.m))
        
        for t in range(T):
            
            u_t = self.choose_control(t)
            self.u = u_t

            x_t_ = self.dynamics_step(self.x, u_t)

            self.online_OLS(self.x, x_t_, u_t)
            self.x = x_t_

            self.A_t_values[t+1] = self.A_t.copy()
            self.u_values[t] = self.u

            M = self.M_t.mean(axis=0)
            S, _ = np.linalg.eig(M)
            # self.S_values.append(np.sort(S))

        return self.A_t_values

    def online_OLS(self, x_t, x_t_, u_t):

        y_t = x_t_ - self.B@u_t

        for row_index in range(self.d):
            prior_moments = self.M_t[row_index]
            prior_estimate = self.A_t[row_index]
            posterior_moments = prior_moments + x_t[:, None]@x_t[None, :]
            # print(f'covariates {covariates}')
            # print(f'moments {moments}')
            combination = prior_moments@prior_estimate + y_t[row_index]*x_t
            if np.allclose(posterior_moments, 0, atol=1e-4):
                continue
            # print(posterior_moments)

            posterior_estimate = solve(posterior_moments, combination)

            self.M_t[row_index] = posterior_moments
            self.A_t[row_index] = posterior_estimate


class Random(Agent):

    def choose_control(self, t):
        return self.draw_random_control()


class Sequential(Agent):

    def choose_control(self, t):
        M = self.M_t.mean(axis=0)
        if  np.allclose(M, 0):
            return self.draw_random_control()
        A = self.A_t_values[t] if self.A_star is None else self.A_star
        u = approximate_D_optimal(M, A, self.B, self.x, self.gamma)
        u *= self.gamma / norm(u)
        return u


class Offline(Agent):

    # def identify(self, T, A_star=None):
    #     self.A_star = A_star
    #     A = self.A_t if self.A_star is None else self.A_star
    #     self.plan(A, T)
    #     return super().identify(T, A_star)

    def plan(self, A, T, n_gradient, batch_size):

        x = torch.tensor(self.x, dtype=torch.float)
        A, B = torch.tensor(A, dtype=torch.float), torch.tensor(
            self.B, dtype=torch.float)
        self.planning = Planning(A, B, T, self.gamma, self.sigma, x, 'MSE')
        self.planning.plan(n_gradient, batch_size)

        self.U = self.planning.U.clone().detach().numpy()

    def choose_control(self, t, A_star=None):

        return self.U[t]
