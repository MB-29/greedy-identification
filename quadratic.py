import numpy as np
from scipy.optimize import newton, brentq



def approximate_D_optimal(M0, A, B, x, gamma):

    # M_inv = np.linalg.inv(M0 + A@x[:, None]@x[None, :]@A.T)
    M_inv = np.linalg.inv(M0)
    # print(f'x {x}')
    # print(f'A {A}')
    # print(f'M_inv {M_inv}')
    Q = - B.T @ M_inv @ B
    b = B.T @ M_inv @ A @ x
    # print(f'Q = {Q}')
    # print(f'b = {b}')

    return maximize_quadratic_ball(Q, b, gamma)
    
def approximate_E_optimal(M0, A, B, x, gamma):

    eigenvalues, eigenvectors = np.linalg.eig(M0)
    idx = eigenvalues.argsort() 
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    v = eigenvectors[:, 0]
    sign = np.sign((A@x)@v)
    sign = 1 if sign == 0 else sign
    u = sign * B.T@v
    u *= gamma/np.linalg.norm(u)
    # print(np.linalg.norm(u))

    return u

def approximate_A_optimal(M0, A, B, x, gamma):

    M_inv = np.linalg.inv(M0 + A@x[:, None]@x[None, :]@A.T)
    # print(f'x {x}')
    # print(f'A {A}')
    # print(f'M_inv {M_inv}')
    Q = - B.T @ M_inv**2 @ B
    b = B.T @ M_inv**2 @ A @ x
    # print(f'Q = {Q}')
    # print(f'b = {b}')

    return maximize_quadratic_ball(Q, b, gamma)

def maximize_quadratic_ball(Q, b, gamma):


    eigenvalues, eigenvectors = np.linalg.eig(Q)
    idx = eigenvalues.argsort() 
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # print(f'Q = {Q}, eigenvalues {eigenvalues}')
    if not b.any():
        return gamma*eigenvectors[:, 0]

    beta = eigenvectors.T @ b
    # print(f'b = {b}')
    # print(f'beta = {beta}')
    mu_l = -eigenvalues[0] + 0.9*(1/gamma)*abs(beta[0]) 
    mu_u = -eigenvalues[0] + (1/gamma)*(np.linalg.norm(b)) 
    # print(f'mu_l = {mu_l}, mu_0 = {mu0}, mu_u = {mu_u}')
    # mu0 = np.linalg.norm(b)/gamma -eigenvalues[0] 
    def func(mu):
        return (beta**2 / (eigenvalues+mu)**2).sum() - gamma**2
    # print(f'f_l = {func(mu_l)}, f0 = {func(mu0)}, f_u = {func(mu_u)}')
    mu = brentq(
        func,
        mu_l,
        mu_u,
        # fprime=fprime,
        # fprime2=fprime2
    ) 
    # print(f'f* = {func(mu)}')
    # print(mu + eigenvalues)

    c = beta / (eigenvalues + mu)
    
    u = eigenvectors @ c

    return u
