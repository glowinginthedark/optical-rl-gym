"""
Hyperparameter search for - reinforcement learning using - the
stable-baselines3 proximal policy optimization (PPO) function.

https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
"""

import os
import pickle
import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv

from grey_wolf import GreyWolfOptimizer

from typing import List

import uuid
import time
from datetime import datetime


PROJECT_ROOT_DIR = os.environ.get('GWO_PROJECT_ROOT_DIR') # project root directory (not current dir)
CURRENT_DIR = f"{PROJECT_ROOT_DIR}/examples/stable_baselines/grey_wolf"
LOG_DIR = F"{CURRENT_DIR}/grey_wolf_deep_rmsa_ppo"

TOPOLOGY_NAME = "nsfnet_chen_eon"
K_PATHS = 5

USE_US_TOPOLOGY = True
use_us_topology = ''

if USE_US_TOPOLOGY:
    use_us_topology = '_nobel_us'

TOPOLOGY_FILE_PATH = f"{PROJECT_ROOT_DIR}/examples/topologies/{TOPOLOGY_NAME}_{K_PATHS}-paths{use_us_topology}.h5"


class VecExtractDictObs(VecEnvWrapper):
    """
    A vectorized wrapper for filtering a specific key from dictionary observations.
    Similar to Gym's FilterObservation wrapper:
        https://github.com/openai/gym/blob/master/gym/wrappers/filter_observation.py

    :param venv: The vectorized environment
    :param key: The key of the dictionary observation
    """

    def __init__(self, venv: VecEnv, key: str):
        self.key = key
        super().__init__(venv=venv, observation_space=venv.observation_space)

    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self.venv.step_async(actions)

    def step_wait(self) -> VecEnvStepReturn:
        obs, reward, done, info = self.venv.step_wait()
        return obs, reward, done, info


# callback from https://stable-baselines.readthedocs.io/en/master/guide/examples.html#using-callback-monitoring-training
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                 # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {} - ".format(self.num_timesteps), end="")
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))
                  # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                        self.model.save(self.save_path)
                #if self.verbose > 0:
                    #clear_output(wait=True)

        return True

# loading the topology binary file containing the graph and the k-shortest paths
# if you want to generate your own binary topology file, check examples/create_topology_rmsa.py
with open(TOPOLOGY_FILE_PATH, 'rb') as f:
    topology = pickle.load(f)

# node probabilities from
# https://github.com/xiaoliangchenUCD/DeepRMSA/blob/6708e9a023df1ec05bfdc77804b6829e33cacfe4/Deep_RMSA_A3C.py#L77
node_request_probabilities = np.array([
    0.01801802, 0.04004004, 0.05305305, 0.01901902, 0.04504505, 0.02402402, 0.06706707,
    0.08908909, 0.13813814, 0.12212212, 0.07607608, 0.12012012, 0.01901902, 0.16916917
])

def make_environment(environment_id, subprocess_index, wolf_number, iteration_number, logdir, seed=0):
    """
    Create a Monitor environment - corresponding to a subprocess.

    :param environment_id: (str) the environment ID
    :param subprocess_index: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    """
    def _init():
        environment_args = dict(
            topology=topology,
            seed=10,
            allow_rejection=False, # the agent cannot proactively reject a request
            j=1, # consider only the first suitable spectrum block for the spectrum assignment
            # mean_service_holding_time not set as in the paper to achieve comparable reward values
            mean_service_holding_time=3,
            episode_length=50,
            node_request_probabilities=node_request_probabilities
        )

        log_file = f"{logdir}/gwo_s{subprocess_index}_i{iteration_number}_w{wolf_number}_{uuid.uuid4()}"

        os.system(f"touch {log_file}")

        # os.makedirs(log_file, exist_ok=True)

        env = Monitor(
            gym.make(environment_id, **environment_args),
            log_file,
            info_keywords=(
                'episode_service_blocking_rate',
                'episode_bit_rate_blocking_rate'
            )
        )

        env.seed(seed + subprocess_index)

        return env

    set_random_seed(seed)
    return _init

def objective_function(hyperparameters: List[float], wolf_number: int, iteration_number: int, logdir: str, subprocesses: int) -> float:
    """
        Objective function - This is the function we want to minimize using wolves.

        :param hyperparameters: (list) Chosen hyperparameters for proximal policy optimization,
        in the order shown below:
        [
            neural_network_number_of_nodes_per_layer,
            neural_network_number_of_layers,
            learning_rate,
            entropy_coefficient
        ]
        :returns: (float) Minimum (smallest) episode bit rate blocking rate.
    """

    print(f"Starting PPO using hyperparameters: {hyperparameters}")

    environment = VecExtractDictObs(
        SubprocVecEnv([
            make_environment('DeepRMSA-v0', subprocess_index, wolf_number, iteration_number, logdir)
            for subprocess_index in range(subprocesses)
        ]),
        key="observation"
    )

    policy_args = dict(net_arch=int(hyperparameters[1])*[int(hyperparameters[0])])
    agent = PPO(
        MlpPolicy,
        environment,
        verbose=1,
        ent_coef=hyperparameters[3],
        # tensorboard_log="./tb/PPO-DeepRMSA-v0/",
        policy_kwargs=policy_args,
        gamma=.95,
        learning_rate=hyperparameters[2]
    )

    agent.learn(
        total_timesteps=1500,
        callback=[SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=experiment_log_dir)]
    )

    return min([i['episode_bit_rate_blocking_rate'] for i in list(agent.ep_info_buffer)])

def run(logdir):
    """
        Run.
    """

    topology = "nsfnet_chen_eon_5-paths"

    if USE_US_TOPOLOGY:
        topology = "nsfnet_chen_eon_5-paths_nobel_us"

    dimensions =  4
    intervals = {
        "neural_network_number_of_nodes_per_layer": [8, 512],
        "neural_network_number_of_layers": [1, 8],
        "learning_rate": [10e-5, 0.01],
        "entropy_coefficient": [10e-5, 0.5]
    }
    subprocesses = 4
    num_wolves = 3
    max_iteration_number = 1

    gwo = GreyWolfOptimizer(
        objective_function = objective_function,
        max_iteration_number = max_iteration_number,
        num_wolves = num_wolves,
        intervals = intervals,
        logdir = logdir,
        subprocesses = subprocesses
    )

    print("\nStarting GWO algorithm\n")

    start_time = time.time()

    best_position = gwo.run_gwo()

    print("\nGWO completed\n")
    print("\nBest solution found:")
    print(["%.6f"%best_position[k] for k in range(dimensions)])
    print("fitness of best solution = %.6f" % gwo.evaluate_fitness(best_position, 999, 999))
    print("\nEnd GWO for f\n")
    print(f"\nTopology => {topology}")
    print(f"Wolves => {num_wolves}")
    print(f"Iterations => {max_iteration_number}")
    print(f"Subprocesses => {subprocesses}")
    print(f"Total time taken => {time.time() - start_time} seconds")
    print(f"Log directory => {logdir}")


if __name__ == '__main__':
    t = time.mktime((datetime.now()).timetuple())
    experiment_log_dir = f"{LOG_DIR}/{t}"
    os.makedirs(experiment_log_dir, exist_ok=True)

    # run(experiment_log_dir)

    # best EU
    # a = objective_function(
    #     [57.150171510146116, 1.5973725274316413, 0.002436231245137785, 0.09342223611546041],
    #     999, 999
    # )

    # best US
    # a = objective_function(
    #     [372.7700422923347, 4.558810949514859, 0.00930657739628599, 0.056794321271912244],
    #     999, 999, experiment_log_dir, 4
    # )

    # print(a)

