
# gym-graph-search

MDP environments for graph search problems. The graph is randomly generated. The environment is fully observable. Action and state space are discrete. 

See demonstration of usage in `./demo`

# General Usage

```
import gym
import gym_graph_search

# This is specific to the environment desired; see below
env = gym.make(...)

# Returns the current node in the graph
# Fully observable, so this is precisely the state vector
obs = env.reset() 

# Returns a list of neighboring nodes in the graph
action_space = env.get_action_space() 

obs, reward, done, info = env.step(action)
...
```


# Environments
There are two primary environments we export currently. Details are below.

There is a third deprecated environment in which each node has the same number of neighbors, such that the full action space is valid. This can be found in the [source code](./gym_graph_search/envs/graph_search_env.py#248).



## Barabasi Albert scale-free graph
This environment is a Barabasi-Albert graph. State vectors are simply one-hot vectors. This environment name `graph-search-ba-v0`.

This environment has args n,m<sub>0</sub>,m, integers with the constraint that n > m<sub>0</sub> >= m. n is the number of nodes in the graph, m<sub>0</sub> is the number of initial nodes, and m is the (relatively tight) lower bound of the average number of neighbors of a node. These parameters are [further described on wikipedia](https://en.wikipedia.org/wiki/Barab%C3%A1si%E2%80%93Albert_model#Algorithm).

This environment can be initialized as follows. 

```
env = gym.make('graph-search-ba-v0', n=n, m0=m0, m=m)
```


## Euclidean embedded graph
This environment is a cluster of points in Euclidean space, R<sup>d</sup>, sampled from a clipped multivariate Gaussian normal distribution. Each point is connected to at least k of its nearest neighbors (k is only a lower bound because the graph is undirected). State vectors are d-dimensional vectors of real numbers. This environment name is `graph-search-rd-v0'.

This environment has primary args n,d,k, integers as described above, with the constraint that n > k. n is the total number of nodes in the graph.

This environment can be initialized as follows. 

```
env = gym.make('graph-search-rd-v0', n=n, d=d, k=k)
```

There are also finer knobs available: covariance and clipping. These can be found in the [source code](./gym_graph_search/envs/graph_search_env.py#L97).





# Installation

```
cd gym-graph-search/
pip install -e .
```




