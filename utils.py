import numpy as np
import torch

def generate_random_A(d):
    M = np.random.randn(d, d)
    eigenvals = np.linalg.eigvals(M)
    rho = np.abs(eigenvals).max()
    return M / rho


def A_criterion(S, T):
    return T * (1/S**2).sum(dim=1).mean()


def D_criterion(S, T):
    return -torch.log(torch.prod(S, dim=1).mean())


def L_criterion(S, T):
    return -torch.sum(torch.log(S), dim=1).mean()


def E_criterion(S, T):
    return - S[:, -1].mean()


def T_criterion(S, T):
    return - (1/T) * (S**2).mean(dim=1).mean()


criteria = {
    'A': A_criterion,
    'D': D_criterion,
    'E': E_criterion,
    'L': L_criterion,
    'T': T_criterion
}
