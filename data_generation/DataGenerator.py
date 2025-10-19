import random
import pandas as pd
from data_generation.Environment import Environment
from data_generation.Agent import Agent
import json
from tqdm import tqdm


class DataGenerator:

    def __init__(self, n_agents: int, n_environments: int, n_runs: int, n_corners: int, min_steps: int = 100, max_steps: int = 1000, seed: int = 42):
        random.seed(seed)
        self.n_agents = n_agents
        self.n_environments = n_environments
        self.n_runs = n_runs
        self.n_corners = n_corners
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.data = pd.DataFrame()


    def generate_agents(self):
        agents_params_ranges = [
            {'v_min': [0.02, 0.1], 
            'v_max': [0.6, 0.8], 
            'dt': [0.1, 0.1],
            'theta_s': [0.2, 0.4], 
            'mu_s': [0.0, 0.0], 
            'sigma_s': [0.3, 0.5],
            'kappa': [4.0, 10.0],
            'speed_change_prob': [0.1, 0.2],
            }
        ]
        agents = []
        for _ in range(self.n_agents):
            agent_params = {key: random.uniform(*value) for key, value in agents_params_ranges[0].items()}
            agents.append(agent_params)
        return agents

    def generate_environments(self):
        env_params_ranges = [
            {'num_corners': [self.n_corners, self.n_corners],
             'min_radius': [1, 1],
             'max_radius': [1, 1]}
        ]
        environments = []
        for _ in range(self.n_environments):
            env_params = {key: random.randint(*value) if isinstance(value, list) and len(value) == 2 else value for key, value in env_params_ranges[0].items()}
            environments.append(env_params)
        return environments

    def compress_data(self):
        self.data['step'] = self.data['step'].astype('Int16')
        self.data['agent_id'] = self.data['agent_id'].astype('Int16')
        self.data['environment_id'] = self.data['environment_id'].astype('Int16')
        self.data['run_id'] = self.data['run_id'].astype('Int16')
        self.data['collision'] = self.data['collision'].astype('boolean')
        self.data['direction'] = self.data['direction'].astype('float16')
        self.data['speed'] = self.data['speed'].astype('float16')
        self.data['x'] = self.data['x'].astype('float16')
        self.data['y'] = self.data['y'].astype('float16')

    def generate_data(self):
        agents = self.generate_agents()
        environments = self.generate_environments()
        total_iterations = self.n_environments * self.n_agents * self.n_runs
        pbar = tqdm(total=total_iterations, desc="Generating data")
        run_id = 0
        for env_id, env_params in enumerate(environments):
            env = Environment(**env_params)
            for agent_id, agent_params in enumerate(agents):
                agent = Agent(environment=env, **agent_params)
                for _ in range(self.n_runs):
                    for _ in range(random.randint(self.min_steps, self.max_steps)):
                        agent.step(1)
                    run_data = agent.get_data().copy()
                    run_data['agent_id'] = agent_id
                    run_data['agent_params'] = json.dumps(agent.get_agent_params())
                    run_data['environment_id'] = env_id
                    run_data['environment_params'] = json.dumps(env.get_env_params())
                    run_data['run_id'] = run_id
                    self.data = pd.concat([self.data, run_data], ignore_index=True)
                    run_id += 1
                    pbar.update(1)
                    agent.reset()
        pbar.close()
        self.compress_data()
        return self.data