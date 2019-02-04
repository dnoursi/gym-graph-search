# David Noursi

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np

class GraphSearchEnv(gym.Env):
    metadata = {'render.modes':[]}

    def __init__(self):
        n = 10
        N = n**2

        # number of neighbors of each node
        self.n = n
        # number of nodes (can be 2**n?)
        self.N = N

        self.graph_edges = [list(np.random.choice(a=N, size=n)) for _ in range(N)]

        self.observation_space = spaces.Discrete(N)
        self.action_space = spaces.Discrete(n)
        return

    def get_action_space(self):
        return self.graph_edges[self.current_state]

    def step(self, action):
        #print("state", self.current_state, "action", action, "edges", self.graph_edges[self.current_state])

        self.current_state = self.graph_edges[self.current_state][action]

        obs = self.current_state
        reward = 1
        done = (self.current_state == self.N-1)
        info = {}
        # reward: float. done: bool. info: dict.
        # return obs, reward, done, info
        return obs, reward, done, info

    def reset(self):
        self.current_state = 0
        return

    def render(self, mode='human', close=False):
        return



