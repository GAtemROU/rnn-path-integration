from Environment import Environment
import random
import math
from matplotlib import pyplot as plt
import pandas as pd

class Agent:

    def __init__(self, environment: Environment,
                 min_speed: float = 0,
                 max_speed: float = 1.5,
                 angle_change_prob: float = 0.7,
                 angle_change_std: float = math.pi / 4,
                 speed_change_prob: float = 0.1,
                 speed_change_std: float = 0.1,
                 seed: None | int = None):
        if seed is not None:
            random.seed(seed)
        self.environment = environment
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.x = 0.0
        self.y = 0.0
        self.speed = random.uniform(min_speed, max_speed)
        self.angle = random.uniform(0, 2 * math.pi)
        self.angle_change_prob = angle_change_prob
        self.angle_change_std = angle_change_std
        self.speed_change_prob = speed_change_prob
        self.speed_change_std = speed_change_std
        self.path = [(self.x, self.y)]
        self.data = pd.DataFrame()
        self.steps = 0


    def step(self, dt: float) -> tuple[float, float]:
        self.steps += 1
        # Update the position based on the current speed and angle
        x = self.x + self.speed * math.cos(self.angle) * dt
        y = self.y + self.speed * math.sin(self.angle) * dt

        # Check if the new position is inside the environment
        collision = False
        if not self.environment.is_inside((x, y)):
            collision = True
            # If not, reflect the angle
            self.angle = (self.angle + math.pi) % (2 * math.pi)

            # find the intersection point
            intersection = self.environment.find_intersection(self.x, self.y, x, y)
            if intersection is not None:
                self.x = intersection[0]
                self.y = intersection[1]
        else:
            self.x = x
            self.y = y

        self.data = pd.concat([self.data, pd.DataFrame([[self.steps, self.angle, self.speed, self.x, self.y, collision]], 
                                                       columns=['step', 'angle', 'speed', 'x', 'y', 'collision'])], ignore_index=True)

        # modify angle with angle_change_prob
        if random.random() < self.angle_change_prob:
            self.angle += random.gauss(0, self.angle_change_std)
            self.angle %= 2 * math.pi

        # modify speed with speed_change_prob
        if random.random() < self.speed_change_prob:
            self.speed += (self.min_speed + self.max_speed)/2 - self.speed + random.gauss(0, self.speed_change_std)
            self.speed = self.min_speed + (self.max_speed - self.min_speed) * (1 / (1 + math.exp(-self.speed)))

        self.path.append((self.x, self.y))
        return (self.x, self.y)

    def get_path(self):
        return self.path

    def get_data(self) -> pd.DataFrame:
        return self.data
    
    def visualize_path(self):
        self.environment.visualize(details=False)
        points = self.data[['x', 'y']].values
        steps = self.data['step'].values
        plt.scatter(points[:, 0], points[:, 1], c=steps, cmap='viridis', s=20)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Path Visualization')
        plt.colorbar(label='Step')
        plt.show()

    def get_agent_params(self) -> dict[str, float]:
        return {
            'min_speed': self.min_speed,
            'max_speed': self.max_speed,
            'speed_change_prob': self.speed_change_prob,
            'speed_change_std': self.speed_change_std,
            'angle_change_prob': self.angle_change_prob,
            'angle_change_std': self.angle_change_std,

        }

    def reset(self):
        self.x = 0.0
        self.y = 0.0
        self.speed = random.uniform(self.min_speed, self.max_speed)
        self.angle = random.uniform(0, 2 * math.pi)
        self.path = [(self.x, self.y)]
        self.data = pd.DataFrame([{'step': 0, 'angle' : self.angle, 'speed': self.speed, 'x': self.x, 'y': self.y, 'collision': False}],
            columns=['step', 'angle', 'speed', 'x', 'y', 'collision'])
        self.steps = 0