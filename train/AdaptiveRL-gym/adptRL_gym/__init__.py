from gymnasium.envs.registration import register

register(
     id="adptRL_gym/adptRL-v0",
     entry_point="adptRL_gym.envs:AdaptiveRLEnv",
     max_episode_steps=1000,
)
