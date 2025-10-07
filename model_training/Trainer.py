import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model_training.Loss import NavigationLoss
import tqdm


class RNNTrainer:
    def __init__(self, model, lr=1e-3, lambda_L2=1e-4, lambda_FR=1e-3, device="cpu"):
        self.model = model.to(device)
        self.device = device
        self.loss_fn = NavigationLoss(lambda_L2, lambda_FR)
        self.optimizer = optim.Adam(self.model.parameters(), amsgrad=True)
        self.loss_history = []

    def train_batch(self, X_batch, Y_batch, batch_lengths=None):
        """
        X_batch: [M, T, N_in]
        Y_batch: [M, T, N_out]
        """
        self.optimizer.zero_grad()

        M, T, N_in = X_batch.shape  
        Y_preds = []
        Us = []

        # Run each sequence separately (or you could vectorize this)
        for m in range(M):
            U, Y_pred = self.model(X_batch[m], T=T)
            Y_preds.append(Y_pred.unsqueeze(0))
            Us.append(U.unsqueeze(0))

        Y_preds = torch.cat(Y_preds, dim=0)
        Us = torch.cat(Us, dim=0)

        total_loss, task_loss, reg_L2, reg_FR = self.loss_fn(Y_preds, Y_batch, Us, self.model, batch_lengths=batch_lengths, device=self.device)
        total_loss.backward()
        self.optimizer.step()

        return {
            "total": total_loss.item(),
            "task": task_loss.item(),
            "reg_L2": reg_L2.item(),
            "reg_FR": reg_FR.item(),
        }

    def train(self, dataset: DataLoader, n_epochs=100, verbose=True):
        pbar = tqdm.tqdm(range(n_epochs)) if verbose else range(n_epochs)
        for epoch in pbar:
            total_loss = 0
            n_batches = 0
            for batch in dataset:
                X_batch, Y_batch = batch[0].to(self.device), batch[1].to(self.device)
                batch_lengths = batch[2].to(self.device) if len(batch) > 2 else None
                stats = self.train_batch(X_batch, Y_batch, batch_lengths=batch_lengths)
                total_loss += stats["total"]
                n_batches += 1
            if verbose:
                pbar.set_description(f"Epoch {epoch:3d} | Total Loss: {total_loss/n_batches:.6f}")
            self.loss_history.append(total_loss / n_batches)
        if verbose:
            pbar.close()