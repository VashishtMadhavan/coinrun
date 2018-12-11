import numpy as np
from coinrun import setup_utils, make

config_args = setup_utils.setup_and_load(use_cmd_line_args=False)
env = make('standard', num_envs=4)
for _ in range(1000):
    env.render()
    acts = np.array([env.action_space.sample() for _ in range(env.num_envs)])
    _obs, _rews, _dones, _infos = env.step(acts)
env.close()