import numpy as np

def generate_random_A(d):
    M = np.random.randn(d, d)
    eigenvals = np.linalg.eigvals(M)
    rho = np.abs(eigenvals).max()
    return M / rho