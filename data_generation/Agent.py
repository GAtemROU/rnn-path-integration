from data_generation.Environment import Environment
from numpy.random import default_rng
import math
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np



class Agent:

    def __init__(self, environment: Environment,
                 x0: float = 0.0,
                 phi0: float = 0.0,
                 pos0: tuple[float, float] = (0.0,0.0),
                 v_min = 0.02, 
                 v_max = 0.6, 
                 dt = 0.2,
                 theta_s = 0.2, 
                 mu_s = 0.0, 
                 sigma_s = 0.3,
                 kappa = 4.0,
                 speed_change_prob: float = 0.1,
                 seed: None | int = None):
        if seed is not None:
            self.rng = default_rng(seed)
        else:
            self.rng = default_rng()

        self.x0 = x0
        self.pos0 = pos0
        self.phi0 = phi0
        self.xs = x0
        self.phi = phi0
        self.pos = np.array(pos0, dtype=float)
        self.v_min = v_min
        self.v_max = v_max
        self.dt = dt
        self.theta_s = theta_s
        self.mu_s = mu_s
        self.sigma_s = sigma_s
        self.kappa = kappa
        self.environment = environment
        self.v = 0.0
        self.speed_change_prob = speed_change_prob
        self.path = [self.pos.copy()]
        self.data = pd.DataFrame([{'step': 0, 'direction' : 0, 'speed': 0, 'x': self.pos[0], 'y': self.pos[1], 'collision': False}])
        self.steps = 0
        self.resample_counter = 0

    def step(self, dt: float) -> np.ndarray:

        if self.rng.random() < self.speed_change_prob:
            noise = self.rng.normal() * self.sigma_s * np.sqrt(self.dt)
            self.xs += self.theta_s * (self.mu_s - self.xs) * self.dt + noise
        # Map to speed
        frac = self.sigmoid(self.xs)
        self.v = self.v_min + (self.v_max - self.v_min) * frac

        self.steps += 1

        collision = False
        while True:
            # Heading update: sample turn from von Mises (center 0)
            dphi = self.rng.vonmises(0.0, self.kappa)
            self.phi += dphi + 2 * math.pi
            self.phi %= 2 * math.pi
            dx = self.v * self.dt * np.cos(self.phi)
            dy = self.v * self.dt * np.sin(self.phi)

            if self.environment.is_inside(self.pos + np.array([dx, dy])):
                self.pos += np.array([dx, dy])
                break
            else:
                collision = True
                self.resample_counter += 1
        
        self.data = pd.concat([self.data, pd.DataFrame([[self.steps, self.phi, self.v, self.pos[0], self.pos[1], collision]], 
                                                       columns=['step', 'direction', 'speed', 'x', 'y', 'collision'])], ignore_index=True)

        self.path.append(self.pos.copy())
        return self.pos

    def sigmoid(self, x: float) -> float:
        return 1 / (1 + math.exp(-x))

    def get_path(self):
        return self.path

    def get_data(self) -> pd.DataFrame:
        return self.data
    
    def visualize_path(self):
        self.environment.visualize(details=False)
        points = self.data[['x', 'y']].to_numpy()
        steps = self.data['step'].to_numpy()
        plt.scatter(points[:, 0], points[:, 1], c=steps, cmap='viridis', s=20)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Path Visualization')
        plt.colorbar(label='Step')
        plt.show()

    def get_agent_params(self) -> dict[str, float]:
        return {
            'v_min': self.v_min,
            'v_max': self.v_max,
            'theta_s': self.theta_s,
            'mu_s': self.mu_s,
            'sigma_s': self.sigma_s,
            'kappa': self.kappa,
            'speed_change_prob': self.speed_change_prob
        }

    def reset(self):
        self.xs = self.x0
        self.phi = self.phi0
        self.pos = np.array(self.pos0, dtype=float)
        self.v = 0.0
        self.path = [self.pos.copy()]
        self.data = pd.DataFrame([{'step': 0, 'direction' : 0, 'speed': 0, 'x': self.pos[0], 'y': self.pos[1], 'collision': False}])
        self.steps = 0