import numpy as np
import torch


def integration(x, A, B, U, T, sigma):
    batch_size, d = x.shape
    x_values = torch.zeros(batch_size, T+1, d)
    W = sigma*torch.randn(batch_size, T, d)
    for t in range(T):
        w = W[:, t, :]
        u =  U[t, :] 
        x = (A @ x.T).T + B@u + w
        x_values[:, t+1, :] = x
    return x_values, W

def estimate(x_values, B, U):
    X = x_values[:, :-1]
    Y = x_values[:, 1:, :] - U@(B.T)
    ols, _, _, _ = torch.linalg.lstsq(X, Y)
    return ols.permute(0, 2, 1)

def test(x, A, B, U, T, sigma):
    x_values, W = integration(x, A, B, U, T, sigma)
    estimate_values = estimate(x_values, B, U)
    residuals = estimate_values - A.unsqueeze(0)
    # X =  x_values[:, :-1]
    # residuals = X.pinverse()@W
    return torch.linalg.norm(residuals, dim=(1, 2))

def D_optimality(X, W):
    M = X.permute(0, 2, 1)@X
    return -torch.logdet(M).mean()
# def E_optimality(X, W):
#     M = X.permute(0, 2, 1)@X
#     S = -torch.linalg.svdvals(M)[:, -1].mean()
#     return -torch.logdet(X.permute(0, 2, 1)@X).mean()
def MSE_error(X, W):
    residual = X.pinverse()@W
    return torch.norm(residual, dim=(1,2)).mean()


class Planning:
    def __init__(self, A, B, T, gamma, sigma, x0, functional):
        super().__init__()
        self.T = T
        self.A = A
        self.B = B
        self.d, self.m = B.shape
        self.x = x0

        self.gamma = gamma
        self.sigma = sigma

        self.U = torch.randn(self.T, self.m, requires_grad=True)
        self.U_values = []  

        self.functional = {
            'D-optimality': D_optimality,   
            # 'E-optimality': E_optimality,
            'MSE': MSE_error
            }[functional]

    def forward(self, x):
        U = self.gamma * np.sqrt(self.T) * self.U / torch.norm(self.U)
        x_values, W = integration(x, self.A, self.B, U, self.T, self.sigma)
        return x_values, W, U

    def plan(self, n_steps, batch_size, learning_rate=0.1):
        optimizer = torch.optim.Adam([self.U], lr=learning_rate)
        loss_values = []
        for step_index in range(n_steps):
            x = self.x.unsqueeze(0).expand(batch_size, self.d)
            x_values, W, U = self.forward(x)
            X = x_values[:, :-1]

            loss = self.functional(X, W)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_values.append(loss.item())

            self.U.data = self.gamma * np.sqrt(self.T) * self.U / torch.norm(self.U)
            self.U_values.append(U.detach().clone())
            
        return self.U_values, loss_values
