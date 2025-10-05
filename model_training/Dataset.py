import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.nn.utils.rnn import pad_sequence


class NavigationDataset(Dataset):
    """
    Dataset for RNN navigation task with multiple runs.
    Each item corresponds to a single trajectory (run).
    """

    def __init__(self, data: pd.DataFrame):
        """
        Args:
            data: pandas DataFrame with columns:
                - 'run_id': unique identifier for each run
                - 'direction': input angle at each time step
                - 'speed': input speed at each time step
                - 'x': target x position at each time step
                - 'y': target y position at each time step
        """

        data = data.sort_values(by=['run_id', 'step'])
        # group by run_id and create dicts
        self.data = {}
        for run_id, group in data.groupby('run_id'):
            X = group[['direction', 'speed']].to_numpy(dtype='float32')
            Y = group[['x', 'y']].to_numpy(dtype='float32')
            self.data[run_id] = {'X': torch.tensor(X), 'Y': torch.tensor(Y)}
        self.run_ids = data['run_id'].unique()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        return entry['X'], entry['Y'], idx

    def get_run(self, run_id):
        """Retrieve a single run by its ID."""
        if run_id in self.data:
            entry = self.data[run_id]
            return entry['X'], entry['Y']
        raise KeyError(f"Run ID {run_id} not found")

    def get_all_run_ids(self):
        return sorted(set(self.run_ids))

    def collate_padded_runs(self, batch):
        """
        Pads variable-length sequences in a batch to the max length.
        Returns padded tensors and lengths for masking.
        """
        Xs = [b[0] for b in batch]
        Ys = [b[1] for b in batch]
        run_ids = [b[2] for b in batch]

        # Pad to [B, T_max, N_in] and [B, T_max, N_out]
        X_padded = pad_sequence(Xs, batch_first=True)  # default pad with zeros
        Y_padded = pad_sequence(Ys, batch_first=True)

        lengths = torch.tensor([x.shape[0] for x in Xs])  # original lengths

        return X_padded, Y_padded, lengths, run_ids


    def get_dataloader(self, batch_size=16, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_padded_runs)