import gymnasium as gym
from gymnasium import spaces
import numpy as np

from sim.taxi_env import TaxiDispatchEnv


class PPOTaxiEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self):

        super(PPOTaxiEnv, self).__init__()

        self.base_env = TaxiDispatchEnv(
            grid_size=10,
            num_taxis=3,
            max_steps=50,
            seed=42
        )

        self.action_space = spaces.Discrete(
            self.base_env.num_actions
        )

        self.observation_space = spaces.Box(
            low=0,
            high=20,
            shape=(5,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):

        state = self.base_env.reset()

        state = np.array(state, dtype=np.float32)

        return state, {}

    def step(self, action):

        next_state, reward, done, info = self.base_env.step(action)

        next_state = np.array(
            next_state,
            dtype=np.float32
        )

        terminated = done
        truncated = False

        return (
            next_state,
            reward,
            terminated,
            truncated,
            {
                "wait_time": info["waiting_time"]
            }
        )