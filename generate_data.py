import argparse
import os
from data_generation.DataGenerator import DataGenerator
import random


def generate_data(args):
    

    os.makedirs(args.output, exist_ok=True)
    data_gen = DataGenerator(n_agents=args.agents, 
                             n_environments=args.envs, 
                             n_runs=args.runs,
                             n_corners=args.n_corners,
                             min_steps=args.min_steps, 
                             max_steps=args.max_steps, 
                             seed=args.seed)

    data = data_gen.generate_data()
    # Save the data to the output directory
    data.to_csv(os.path.join(args.output, args.output_file + ".csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description="Generate data from agent and environment.")
    parser.add_argument("--agents", type=int, default=10, help="Number of agents to generate.")
    parser.add_argument("--envs", type=int, default=5, help="Number of environments to generate.")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per agent/environment pair.")
    parser.add_argument("--n_corners", type=int, default=4, help="Number of corners in the environment.")
    parser.add_argument("--min_steps", type=int, default=100, help="Minimum number of steps per run.")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum number of steps per run.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output", type=str, default="output", help="Output directory for data.")
    parser.add_argument("--output_file", type=str, default="generated_data", help="Output file name (without extension).")

    args = parser.parse_args()

    random.seed(args.seed)
    generate_data(args)

if __name__ == "__main__":
    main()