
# gym-graph-search

MDP environment for graph search problems. The graph is randomly generated under the constraint that each node has the same fixed number of neighbors. The environment is fully observable. Action and state space are discrete. 

See demonstration of usage in `./demo`

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




