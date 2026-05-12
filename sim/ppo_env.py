import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PPOTaxiEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self):

        super(PPOTaxiEnv, self).__init__()

        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            low=0,
            high=10,
            shape=(4,),
            dtype=np.float32
        )

        self.state = np.array([0, 0, 0, 0], dtype=np.float32)

        self.current_step = 0
        self.max_steps = 50

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.state = np.random.randint(
            0,
            10,
            size=(4,)
        ).astype(np.float32)

        self.current_step = 0

        return self.state, {}

    def step(self, action):

        self.current_step += 1

        passenger = self.state[:2]
        taxi = self.state[2:]

        wait_time = np.linalg.norm(passenger - taxi)

        reward = -wait_time

        self.state = np.random.randint(
            0,
            10,
            size=(4,)
        ).astype(np.float32)

        terminated = self.current_step >= self.max_steps

        truncated = False

        info = {
            "wait_time": float(wait_time)
        }

        return (
            self.state,
            reward,
            terminated,
            truncated,
            info
        )