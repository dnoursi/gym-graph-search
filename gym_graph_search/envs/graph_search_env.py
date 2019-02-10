# David Noursi

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np

# Barabasi Albert Graph
# m0 = initial clique size
# m = number of neighbors to add to each new node
# n = number of total desired nodes in graph
# Begins with graph of m0, adds n-m0 nodes with m neighbors each
def ba_graph(n, m0, m):
    ls0 = list(range(m0))
    edges = [ls0[:i] + ls0[i+1:] for i in ls0]
    # Number of neighbors
    nn = [n-1 for _ in ls0]
    sum_nn = sum(nn)
    for i_new in range(m0, n):
        new = []
        for i in range(len(edges)):
            proba = nn[i]/sum_nn
            if np.random.random() < proba:
                new.append(i)

                edges[i].append(i_new)
                nn[i] += 1

                sum_nn += 2
        edges.append(new)
        nn.append(len(new))
        print(edges)
    return edges

def random_graph(n, m0, m):
    return ba_graph(n, m0, m)

class GraphSearchEnv(gym.Env):
    metadata = {'render.modes':[]}

    def __init__(self):

        # number of nodes
        N = 40
        self.N = N
        m0 = 12
        m = 8

        self.root = 0
        self.target = m0-1

        self.graph_edges = random_graph(N, m0, m)

        self.observation_space = spaces.Discrete(N)
        self.action_space = spaces.Discrete(N)
        return

    def get_action_space(self):
        return self.graph_edges[self.current_state]

    def step(self, action):
        #print("state", self.current_state, "action", action, "edges", self.graph_edges[self.current_state])
        if action not in self.graph_edges[self.current_state]:
            reward = -2
        else:
            self.current_state = action
            reward = -1

        obs = self.current_state
        done = (self.current_state == self.target)
        info = {}
        # reward: float. done: bool. info: dict.
        # return obs, reward, done, info
        return obs, reward, done, info

    def reset(self):
        self.current_state = self.root
        return self.current_state

    def render(self, mode='human', close=False):
        return

# Erdos Renyi Graph
def er_graph(nnodes, proba):
    edges = [[] for _ in range(nnodes)]
    for i in range(nnodes):
        for j in range(i):
            if np.random.random() < proba:
                edges[i].append(j)
                edges[j].append(i)

    print(edges)
    return edges

# ---
# Currently deprecated:
# ---


# i has no possible neighbors remaining itself.
# Go to j < i, and steal a neighbor for i from j
def steal_two_neighbors(edges, i):
    j = int(np.random.random() * i)
    neighbor_index = int(np.random.random() * len(edges[j]))
    k = edges[j][neighbor_index]

    edges[k].remove(j)
    edges[j].remove(k)

    edges[k].append(i)
    edges[i].append(k)

    edges[j].append(i)
    edges[i].append(j)

    return edges

# (Probably requires nneighbors, nnodes to be even integers)
def fixed_degree_random_graph(nneighbors, nnodes):
    remaining_neighbors = list(range(nnodes))
    edges = [[] for _ in range(nnodes)]
    for i in range(nnodes):
        # print(i)
        if i not in remaining_neighbors:
            continue
        remaining_neighbors.remove(i)

        need_neighbors = nneighbors - len(edges[i])
        while need_neighbors > len(remaining_neighbors):
            edges = steal_two_neighbors(edges, i)
            need_neighbors = nneighbors - len(edges[i])
        neighbors = list(np.random.choice(a= remaining_neighbors, size= need_neighbors, replace=False ))
        for neighbor in neighbors:
            edges[i].append(neighbor)
            edges[neighbor].append(i)
            if len(edges[neighbor]) == nneighbors:
                remaining_neighbors.remove(neighbor)

        assert len(edges[i]) == nneighbors
        for neighbor in edges[i]:
            assert i in edges[neighbor]
        # print(edges)
    return edges

class ConstActionSpaceSizeGraphSearchEnv(gym.Env):
    metadata = {'render.modes':[]}

    def __init__(self):

        # number of neighbors of each node
        self.n = 64
        # number of nodes (can be 2**n?)
        self.N = self.n**2

        self.graph_edges = fixed_degree_random_graph(n, N)
        #self.graph_edges = [list(np.random.choice(a = rangeN[:i] + rangeN[i+1:], size=n)) for i in rangeN]

        self.observation_space = spaces.Discrete(N)
        self.action_space = spaces.Discrete(n)
        return

    def get_action_space(self):
        return self.graph_edges[self.current_state]

    def step(self, action):
        #print("state", self.current_state, "action", action, "edges", self.graph_edges[self.current_state])

        self.current_state = self.graph_edges[self.current_state][action]

        obs = self.current_state
        reward = -1
        done = (self.current_state == self.N-1)
        info = {}
        # reward: float. done: bool. info: dict.
        # return obs, reward, done, info
        return obs, reward, done, info

    def reset(self):
        self.current_state = 0
        return self.current_state

    def render(self, mode='human', close=False):
        return
