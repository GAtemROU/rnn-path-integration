from copy import deepcopy
from shapely.geometry import Polygon, Point
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math
from shapely.geometry import LineString
import random
import numpy as np

class Environment:

    def __init__(self, num_corners: int, 
                 min_radius: float,
                 max_radius: float,
                 seed: None | int = None):
        if seed is not None:
            random.seed(seed)
        assert num_corners > 2, "Number of corners must be greater than 2"
        self.num_corners = num_corners
        assert min_radius <= max_radius and min_radius > 0, "Invalid radius values"
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.positions = self.gen_positions()
        self.env = self.create_env()


    def create_env(self) -> Polygon:
        return Polygon(self.positions)

    def gen_positions(self) -> list[tuple[float, float]]:
        positions = []
        # divide the circle into num_corners segments
        for i in range(self.num_corners):
            angle = 2 * math.pi * i / self.num_corners
            radius = random.uniform(self.min_radius, self.max_radius)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            positions.append((x, y))
        return positions

    def visualize(self, details: bool = False):
        # plot current env
        x, y = self.env.exterior.xy
        plt.fill(x, y, alpha=0.5, fc='blue', ec='black')
        plt.xlim(-self.max_radius * 1.5, self.max_radius * 1.5)
        plt.ylim(-self.max_radius * 1.5, self.max_radius * 1.5)
        if details:
            min_circle = Circle((0, 0), self.min_radius, color='red', fill=False)
            max_circle = Circle((0, 0), self.max_radius, color='green', fill=False)
            plt.gca().add_artist(min_circle)
            plt.gca().add_artist(max_circle)
            for i in range(self.num_corners):
                angle = 2 * math.pi * i / self.num_corners
                plt.plot([0, self.max_radius * math.cos(angle)], [0, self.max_radius * math.sin(angle)], color='black', linestyle='--')
        plt.title("Environment Visualization")
        

    def get_positions(self) -> list[tuple[float, float]]:
        return self.positions
    
    def is_inside(self, point: tuple[float, float] | np.ndarray) -> bool:
        return self.env.contains(Point(point))
    
    def find_intersection(self, x_start, y_start, x_end, y_end):
        line = LineString([(x_start, y_start), (x_end, y_end)])
        intersection = self.env.intersection(line)

        if intersection.is_empty:
            return None
        elif intersection.geom_type == 'Point':
            return (intersection.coords[0][0], intersection.coords[0][1])
        elif intersection.geom_type == 'MultiPoint':
            raise ValueError("Multiple intersection points found")
        elif intersection.geom_type == 'LineString':
            return intersection.coords[0]
        else:
            return None
        

    def get_env_params(self) -> dict:
        return {
            'num_corners': self.num_corners,
            'min_radius': self.min_radius,
            'max_radius': self.max_radius,
        }