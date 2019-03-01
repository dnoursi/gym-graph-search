
# gym-graph-search

MDP environment for graph search problems. The graph is randomly generated under the constraint that each node has the same fixed number of neighbors. The environment is fully observable. Action and state space are discrete. 

See demonstration of usage in `./demo`

# Components

There are two primary environments we export currently. 

## Barabasi Albert scale-free graph
One is a Barabasi-Albert graph. State vectors are simply one-hot vectors. This environment name `graph-search-ba-v0`.

## Euclidean embedded graph
The second is a cluster of points in $R^d$ sampled from a clipped multivariate Gaussian normal distribution. Each point is connected to some of its nearest neighbors, according to the ordinary norm on vectors. State vectors are d-dimensional vectors of real numbers. This environment name is `graph-search-rd-v0'.


# Usage

```
import gym
import gym_graph_search

env = gym.make("graph-search-v0")

# Returns the current node in the graph
obs = env.reset() 

# Returns a list of neighboring nodes in the graph
action_space = env.get_action_space() 

obs, reward, done, info = env.step(action)
...
```

# Installation

```
cd gym-graph-search/
pip install -e .
```




