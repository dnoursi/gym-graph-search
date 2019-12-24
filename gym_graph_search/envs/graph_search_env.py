# David Noursi

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
from sklearn.neighbors import NearestNeighbors

def normalize_proba(p):
    s = sum(p)
    result = [proba/s for proba in p]
    return result

# Barabasi Albert Graph
# m0 = initial clique size
# m = number of neighbors to add to each new node
# n = number of total desired nodes in graph
# Begins with graph of m0, adds n-m0 nodes with m neighbors each
def ba_graph(n, m0, m):
    assert m <= m0
    # Create clique
    ls0 = list(range(m0))
    edges = [set(ls0[:i] + ls0[i+1:]) for i in ls0]
    node_weights = [m0-1 for _ in ls0]
    # Number of neighbors
    for i_new in range(m0, n):
        node_probas = normalize_proba(node_weights)
        new = set(np.random.choice(a=i_new, size=m, p=node_probas, replace=False)) # <-> a = range(i_new)
        assert len(new) == m

        for neighbor in new:
            edges[neighbor].add(i_new)
        edges.append(new)

        node_weights.append(m)
        for i in range(i_new):
            if i in new:
                node_weights[i] += 1
    return edges

# Does not actually assert value m neighbors
def old_ba_graph(n, m0, m):
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
        #print(edges)
    return edges


# n points in \R^d
# center: mu vector,
# covariance: sigma matrix (just scaled identity right now)
def gaussian_random_graph(n,k,d,cov,clip=None,center=None):
    # Validate inputs
    if center == None:
        center = np.zeros(d)
    if clip == None:
        clip = cov * d

    covM = cov * np.identity(d)

    # Create graph
    vectors = np.random.multivariate_normal(mean=center, cov=covM, size=(n))
      # Clip vectors
    for vector in vectors:
        vector = np.array([np.sign(v) * min(abs(v), clip) for v in vector])
    # , algorithm='ball_tree'
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(vectors)
    _, indicess = nbrs.kneighbors(vectors)
    edges = [set() for _ in range(n)]
    for i, indices in enumerate(indicess):
        # The nearest neighbor of each point is itself
        indices = indices[1:]
        # Set union
        edges[i] = edges[i] | set(indices)
        # Graph is undirected
        for j in indices:
            edges[j].add(i)
    return edges, vectors

def grg_s(ns,covs,k,d):
    vectors = [v for v in grps(n, cov, d) for n, cov in zip(ns, covs)]

    nbrs = NearestNeighbors(n_neighbors=k+1).fit(vectors)
    _, indicess = nbrs.kneighbors(vectors)

    edges = [set() for _ in range(sum(ns))]
    for i, indices in enumerate(indicess):
        # The nearest neighbor of each point is itself
        indices = indices[1:]
        # Set union
        edges[i] = edges[i] | set(indices)
        # Graph is undirected
        for j in indices:
            edges[j].add(i)
    return edges, vectors

# clipped gaussian random points
def grps(n,cov,d):
    # Create pts
    covM = [cov * np.identity(d) for cov in covs]
    vectors = np.random.multivariate_normal(mean=center, cov=covM, size=(n))
    # Clip vectors
    clip = cov * d
    for vector in vectors:
        vector = np.array([np.sign(v) * min(abs(v), clip) for v in vector])
    #
    return vectors

def check_connected(graph_edges, nnodes):
    visited = set()
    visitNext = set([0])
    while visitNext:
        current = visitNext.pop()
        visited.add(current)
        visitNext |= (set(graph_edges[current]) - visited)

    if len(visited) == nnodes:
        return True
    return False

class RdGraphSearchEnv(gym.Env):
    metadata = {'render.modes':[]}

    #def __init__(self, n=10, k=5, d=2, cov=1., clip=5, output=True):
    def __init__(self, n=10, k=5, d=2, cov=1., output=True):
        clip = cov * d

        self.nsteps = 0

        # number of nodes
        self.n = n
        assert clip > 0

        self.root = 0
        self.target = 1

        self.graph_edges, self.state_vectors = gaussian_random_graph(n=n, k=k, d=d, cov=cov, clip=clip)

        if output:
            print("Initialized Random Rd Embedded Graph with Parameters n,k,d,cov,clip:", n,k,d,cov,clip)

        self.observation_space = spaces.Box(low=-clip, high=clip, shape = (d,), dtype=np.float32)
        self.action_space = spaces.Discrete(n)
        return

    def is_connected(self):
        return check_connected(self.graph_edges, self.n)

    def get_action_space(self):
        return self.graph_edges[self.current_state]

    def step(self, action):
        #print("state", self.current_state, "action", action, "edges", self.graph_edges[self.current_state])
        if action not in self.graph_edges[self.current_state]:
            reward = -2
        else:
            self.current_state = action
            reward = -1

        obs = self.state_vectors[self.current_state]
        done = (self.current_state == self.target)
        info = {}
        # reward: float. done: bool. info: dict.
        # return obs, reward, done, info

        self.nsteps += 1
        if self.nsteps == self.n:
            done=True
        return obs, reward, done, info

    def reset(self):
        self.current_state = self.root
        return self.current_state

    def render(self, mode='human', close=False):
        return


class BAGraphSearchEnv(gym.Env):
    metadata = {'render.modes':[]}

    def __init__(self, n=30, m0=20, m=20, initial_mode="clique"):

        # number of nodes
        self.n = n

        self.root = 0
        self.target = m0-1

        self.graph_edges = ba_graph(n, m0, m)
        print("Initialized Random BA Graph with Parameters n,m0,m:", n,m0,m)

        self.observation_space = spaces.Discrete(n)
        self.action_space = spaces.Discrete(n)
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
# Each edge has a fixed number of neighbors (fully valid action space)
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
