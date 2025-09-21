from Environment import Environment
import random
import math
from matplotlib import pyplot as plt
import pandas as pd

class Agent:

    def __init__(self, environment: Environment,
                 min_speed: float = 0,
                 max_speed: float = 1.5,
                 angle_change_prob: float = 0.1,
                 angle_change_std: float = math.pi / 8,
                 speed_change_prob: float = 0.1,
                 speed_change_std: float = 0.1,
                 seed: int = 42):
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
        self.data = pd.DataFrame(columns=['step', 'angle', 'speed', 'x', 'y', 'collision'])
        self.steps = 0
        random.seed(seed)

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

        self.data = pd.concat([self.data, pd.DataFrame([[self.steps, self.angle, self.speed, self.x, self.y, collision]], columns=['step', 'angle', 'speed', 'x', 'y', 'collision'])], ignore_index=True)

        # modify angle with angle_change_prob
        if random.random() < self.angle_change_prob:
            self.angle += random.gauss(0, self.angle_change_std)
            self.angle %= 2 * math.pi

        # modify speed with speed_change_prob
        if random.random() < self.speed_change_prob:
            self.speed += random.gauss(0, self.speed_change_std)
            self.speed = max(self.min_speed, min(self.max_speed, self.speed))

        self.path.append((self.x, self.y))
        return (self.x, self.y)

    def get_path(self):
        return self.path
    
    def get_data(self):
        return self.data
    
    def visualize_path(self):
        self.environment.visualize(details=False)
        plt.plot(self.data['x'], self.data['y'], 'ro-')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Path Visualization')
        plt.show()

