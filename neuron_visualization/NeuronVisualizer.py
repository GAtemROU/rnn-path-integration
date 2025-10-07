from model_training.Dataset import NavigationDataset
import matplotlib.pyplot as plt
import tqdm
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.metrics.pairwise import cosine_similarity


class NeuronVisualizer:
    def __init__(self, model):
        self.model = model
        self.neuron_activations = np.zeros(self.model.N)

    def retrieve_activations(self, data, use_predicted=False):
        self.neuron_activations = np.empty((self.model.N, data.shape[0], 3))
        dataset = NavigationDataset(data)
        loader = dataset.get_dataloader(batch_size=1, shuffle=False)
        pbar = tqdm.tqdm(loader)
        self.model.eval()
        row = 0
        for batch in pbar:
            X, Y, lengths, run_ids = batch
            X = X[0]
            Y = Y[0]
            length = lengths[0]
            activations, Y_pred = self.model.forward(X, length)
            Y_pred = Y_pred.detach().cpu().numpy()
            activations = activations.detach().cpu().numpy()
            for j in range(Y.shape[0]):
                for i in range(self.model.N):
                    if use_predicted:
                        x, y = Y_pred[j]
                    else:
                        x, y = Y[j]
                    try:
                        self.neuron_activations[i][row] = (x, y, activations[j][i])
                    except Exception:
                        print(i, row)
                row+=1
        pbar.close()
    
    def get_spatial_maps(self, bins=40, smooth_sigma=1.5):
        maps = np.empty((self.model.N, bins, bins))
        for i in range(self.model.N):
            activations = self.neuron_activations[i]
            x = np.array([a[0] for a in activations])
            y = np.array([a[1] for a in activations])
            u = np.array([a[2] for a in activations])

            heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=u)
            counts, _, _ = np.histogram2d(x, y, bins=bins)
            
            mean_activity = np.divide(heatmap, counts, out=np.zeros_like(heatmap), where=counts>0)
            mean_activity = gaussian_filter(mean_activity, sigma=smooth_sigma)
            for j in range(bins):
                for k in range(bins):
                    maps[i][j][k] = mean_activity[j][k]
        return maps
    
    def flatten_norm_maps(self, maps):
        flat = np.array([m.flatten() for m in maps])
        flat = (flat - flat.mean(axis=1, keepdims=True)) / (flat.std(axis=1, keepdims=True) + 1e-6)
        return flat

        
    def compute_cos_sim_on_maps(self, maps):
        flat = self.flatten_norm_maps(maps)
        return cosine_similarity(flat)

    def plot_hist(self, mean_activity):
        plt.imshow(mean_activity.T, origin='lower', cmap='viridis')
        plt.colorbar(label='Mean activity')
        plt.xlabel("x position")
        plt.ylabel("y position")
        plt.title("Spatial tuning map of one neuron")
        plt.show()

    def get_neuron_activations(self):
        return self.neuron_ativations