import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, N=100, N_in=2, N_out=2, tau=1.0, noise_std=0.01, device="cpu"):
        super().__init__()
        self.N = N
        self.N_in = N_in
        self.N_out = N_out
        self.tau = tau
        self.dt = tau / 10.0
        self.noise_std = noise_std
        self.device = device

        # Parameters
        self.W_rec = nn.Parameter(torch.randn(N, N) / torch.sqrt(torch.tensor(N, dtype=torch.float)))
        self.W_in = nn.Parameter(torch.randn(N, N_in) * 0.1)
        self.W_out = nn.Parameter(torch.randn(N_out, N) * 0.1)
        self.b = nn.Parameter(torch.zeros(N))

    def forward(self, I, T=500):
        """
        Simulate RNN.
        I: external input, shape [T, N_in]
        Returns:
            U: unit activities [T, N]
            Y: readout [T, N_out]
        """
        # state variables
        x = torch.zeros(self.N, device=self.device)
        U = []
        Y = []

        for t in range(T):
            u = torch.tanh(x)

            # readout
            y = self.W_out @ u
            Y.append(y.unsqueeze(0))
            U.append(u.unsqueeze(0))

            # noise
            noise = torch.randn(self.N, device=self.device) * self.noise_std

            # external input at time t
            I_t = I[t] if I is not None else torch.zeros(self.N_in, device=self.device)

            # Euler update
            dx = (-x + self.W_rec @ u + self.W_in @ I_t + self.b + noise) * (self.dt / self.tau)
            x = x + dx

        return torch.cat(U, dim=0), torch.cat(Y, dim=0)

    def set_device(self, device):
        self.device = device
        self.to(device)