import torch
import torch.nn as nn
import torch.optim as optim

class NavigationLoss(nn.Module):
    def __init__(self, lambda_L2=1e-4, lambda_FR=1e-3):
        super().__init__()
        self.lambda_L2 = lambda_L2
        self.lambda_FR = lambda_FR

    def forward(self, Y_pred, Y_target, U, model, batch_lengths=None):
        """
        Compute the total loss according to equations (3)-(5).
        Args:
            Y_pred: [M, T, N_out]
            Y_target: [M, T, N_out]
            U: [M, T, N]  firing rates (tanh outputs)
            model: the ContinuousRNN instance (to access weights)
        """
        M, T, N_out = Y_pred.shape
        N, N_in = model.W_in.shape
        
        # --- (3) task loss ---
        if batch_lengths is not None:
            T_max = batch_lengths.max()
            mask = torch.arange(T_max)[None, :] < batch_lengths[:, None]  # [B, T_max]
            task_loss = ((Y_pred - Y_target)**2 * mask.unsqueeze(-1)).sum() / mask.sum()
        else:
            task_loss = torch.mean((Y_pred - Y_target)**2)

        # --- (4) weight regularization ---
        reg_in = torch.mean(model.W_in**2)
        reg_out = torch.mean(model.W_out**2)
        reg_L2 = reg_in + reg_out

        # --- (5) metabolic cost ---
        reg_FR = torch.mean(U**2)

        # --- total loss ---
        total_loss = task_loss + self.lambda_L2 * reg_L2 + self.lambda_FR * reg_FR
        return total_loss, task_loss, reg_L2, reg_FR
