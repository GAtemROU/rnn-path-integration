import argparse
import os
import numpy as np
from Agent import Agent
from Environment import Environment
import random


def generate_data(args):
    environment = 
    agent = Agent(
        environment=environment,
        min_speed=args.min_speed,
        max_speed=args.max_speed,
        seed=args.seed
    )

    os.makedirs(args.output, exist_ok=True)
    all_data = []
    for i in range(args.episodes // args.episodes_per_env):
        env = Environment(
            num_corners=random.randint(args.min_num_corners, args.max_num_corners),
            min_radius=args.min_radius,
            max_radius=args.max_radius,
            seed=args.seed
        )
        for j in range(args.episodes_per_env):
            agent = Agent(
                environment=environment,
                min_speed=args.min_speed,
                max_speed=args.max_speed,
                seed=args.seed
            )
            for k in range(random.randint(args.min_steps, args.max_steps)):
                agent.step(dt=2)

            all_data.append(agent.get_data())

            

def main():
    parser = argparse.ArgumentParser(description="Generate data from agent and environment.")
    parser.add_argument("--min_radius", type=float, default=10.0, help="Minimum radius of the environment.")
    parser.add_argument("--max_radius", type=float, default=100.0, help="Maximum radius of the environment.")
    parser.add_argument("--min_num_corners", type=int, default=3, help="Minimum number of corners in the environment polygon.")
    parser.add_argument("--max_num_corners", type=int, default=6, help="Maximum number of corners in the environment polygon.")
    parser.add_argument("--min_steps", type=int, default=100, help="Minimum steps per episode.")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum steps per episode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--min_speed", type=float, default=0.0, help="Minimum speed of the agent.")
    parser.add_argument("--max_speed", type=float, default=1.0, help="Maximum speed of the agent.")
    parser.add_argument("--episodes_per_env", type=int, default=10, help="Number of episodes per environment.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to generate.")
    parser.add_argument("--output", type=str, default="output", help="Output directory for data.")
    args = parser.parse_args()

    # Import or define your agent and environment here
    random.seed(args.seed)
    generate_data(args)

if __name__ == "__main__":
    main()