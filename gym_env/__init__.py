from gym.envs.registration import register


register(
      id='RadarEnv-v0',
      entry_point='gym_env.env:RadarEnv',
      max_episode_steps=1000
  )