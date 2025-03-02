import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
from real_time_plot import RealTimePlot  # Import the real-time plotting class

import sys
sys.path.insert(0, "/home/asalvi/code_workspace/RL_AdpEst/train/AdaptiveRL-gym")
import adptRL_gym

sys.path.insert(0, "/home/asalvi/code_workspace/RL_AdpEst/train/AdaptiveRL-gym/adptRL_gym/envs/")

# Set up paths
model_path = "/home/asalvi/code_workspace/tmp/sb3_log/AdaptiveRL/test/AdaptiveRL_test.zip"
#norm_env_path = "/home/asalvi/code_workspace/tmp/sb3_log/AdaptiveRL/test/vecnormalize.pkl"

def evaluate_model(env_id, model_path, num_episodes=1):
    """
    Evaluate the trained model while visualizing Q, R, GT_Q, and GT_R in real-time.
    """
    # Load environment
    env = DummyVecEnv([lambda: gym.make(env_id, seed=42)])
    #env = VecNormalize.load(norm_env_path, env)
    env.training = False  # Disable reward normalization during evaluation
    env.norm_reward = False
    
    # Load trained model
    model = PPO.load(model_path, env=env)
    
    # Initialize real-time plot
    plotter = RealTimePlot(env.envs[0])
    env.envs[0].plotter = plotter  # Attach plotter to the environment
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        step_no = 0
        total_reward = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            
            # Update the real-time plot after each step
            plotter.update()
            
            total_reward += reward
            step_no += 1
        
        print(f"Episode {episode+1}: Total Reward = {total_reward}")
    
    print("Evaluation complete.")
    plt.show()

if __name__ == "__main__":
    env_id = "adptRL_gym/adptRL-v0"
    evaluate_model(env_id, model_path)