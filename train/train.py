import gymnasium as gym
import numpy as np
from torch import nn as nn
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList, ProgressBarCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
import os
import sys

# Import your custom environment

import sys
sys.path.insert(0, "/home/asalvi/code_workspace/RL_AdpEst/train/AdaptiveRL-gym")
import adptRL_gym

sys.path.insert(0, "/home/asalvi/code_workspace/RL_AdpEst/train/AdaptiveRL-gym/adptRL_gym/envs/")


#from adaptive_rl_env import AdaptiveRLEnv  # Make sure this path is correct

# Set up logging directory
tmp_path = "/home/asalvi/code_workspace/tmp/sb3_log/AdaptiveRL/test/"
variant = 'AdaptiveRL_test'
os.makedirs(tmp_path, exist_ok=True)
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

total_timesteps = 1e6


# Callback Definitions
class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                self.logger.record('mean_reward', mean_reward)
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose >= 1:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

        return True


# Create environment in multi-process mode
def make_env(env_id, rank, seed=0):
    """
    Utility function to create an environment with unique parameters per process.

    :param env_id: (str) The environment ID
    :param rank: (int) Unique process rank
    :param seed: (int) Random seed for reproducibility
    """
    def _init():
        #port_no = str(24000 + 2 * rank)  # Unique port assignment (adjust as needed)
        #print(f"Initializing env {rank} on port {port_no}")

        env = gym.make(env_id, seed=seed + rank)
        env = Monitor(env)  # Monitor logs stats for training analysis
        return env

    return _init


if __name__ == '__main__':
    env_id = "adptRL_gym/adptRL-v0"
    num_envs = 16  # Number of parallel environments

    # ✅ **Multi-process environment setup with consistent wrapping**

    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_envs)], start_method='fork')
    #env = DummyVecEnv([make_env(env_id, i) for i in range(num_envs)])
    env = VecMonitor(env, filename=tmp_path)  # Logs training stats
    env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=1000.0, gamma=0.99)

    # ✅ **Callbacks for checkpointing & progress monitoring**
    best_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=tmp_path)
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=tmp_path + "checkpoints/", name_prefix=variant, save_replay_buffer=True, save_vecnormalize=True)
    callback = CallbackList([best_callback, checkpoint_callback, ProgressBarCallback()])

    # ✅ **PPO Model Configuration (Matching Structure to HuskyRL)**
    model = PPO(
        "MlpPolicy",  # Using MLP policy since the environment does not require images
        env,
        learning_rate=0.0001,
        n_steps=512,
        batch_size=512,
        n_epochs=5,
        ent_coef=0.005,
        gamma=0.98,
        gae_lambda=0.98,
        clip_range=0.1,
        vf_coef=0.5,
        max_grad_norm=0.5,
        sde_sample_freq=16,
        policy_kwargs=dict(
            net_arch=dict(pi=[64], vf=[64]),  # Consistent network architecture
            activation_fn=nn.ReLU
        ),
        verbose=1,
        tensorboard_log=tmp_path
    )

    # ✅ **Apply Custom Logger**
    model.set_logger(new_logger)

    # ✅ **Start Training**
    model.learn(total_timesteps=int(total_timesteps), callback=callback, progress_bar=False)

    # ✅ **Save Final Model**
    model.save(tmp_path + variant)

    # ✅ **Example Post-Training Evaluation**
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
