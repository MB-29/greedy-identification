import numpy as np
from numpy.linalg import norm
from scipy.linalg import solve

from quadratic import approximate_D_optimal

class Agent:

    def __init__(self, dynamics, B, gamma, sigma, mean, precision, method):

        self.step = dynamics
        self.B = B

        self.gamma = gamma
        self.sigma = sigma

        self.d = mean.shape[0]
        _, self.m = B.shape

        self.mean = mean.copy() #shape [d, d]
        self.precision = precision.copy() #shape [d, d, d]
        

        u_0 = self.draw_random_control()

        x_0 = np.zeros(self.d)

        x_1 = dynamics(x_0, u_0)
        self.update_posterior(x_0, x_1, u_0)
        
        self.x, self.u = x_1, u_0

        self.estimations_values = [self.mean.copy()]
        self.u_values = [self.u]

        control_methods = {'linear-BOD':self.online_planning, 'random': self.draw_random_control}
        self.choose_control = control_methods[method]

    def online_planning(self):
        M = self.precision.mean(axis=0)
        u = approximate_D_optimal(M, self.mean, self.B, self.x, self.gamma)
        u *= self.gamma / norm(u)
        return u

    def draw_random_control(self):
        u = np.random.randn(self.m)
        u *= self.gamma/norm(u)
        return u

    
    def update_posterior(self, x_t, x_t_, u_t):

        y_t = x_t_ - self.B@u_t

        for row_index in range(self.d):
            prior_precision = self.precision[row_index]
            prior_mean = self.mean[row_index]
            posterior_precision = prior_precision + x_t[:, None]@x_t[None, :]
            # print(f'covariates {covariates}')
            # print(f'precision {precision}')
            combination = prior_precision@prior_mean + y_t[row_index]*x_t
            # print(f'combination {combination}')
            posterior_mean = solve(posterior_precision, combination)

            self.mean[row_index] = posterior_mean
            self.precision[row_index] = posterior_precision
        # print(f'updated mean {self.mean}')
       
    
    def identify(self, T):

        for t in range(1, T):

            # print(f'iteration {t}')

            u_t = self.choose_control()
            # print(f'u_t has norm {norm(u_t)}')
            self.u = u_t
            
            x_t_ = self.step(self.x, u_t)
            self.update_posterior(self.x, x_t_, u_t)
            self.x = x_t_


            self.estimations_values.append(self.mean.copy())
            self.u_values.append(self.u)
    
        return self.estimations_values
