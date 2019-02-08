# gym-graph-search Demonstrations


## Depth-First Search (DFS)

This code is contained within this library
```
python3 dfs.py
```

## OpenAI Baslines

To use [OpenAI Baselines](https://github.com/openai/baselines) for RL algorithms such as Q-Learning or Policy Gradient, edit the file `/baselines/baselines/run.py` with the addition of the following line of code (for systems where this pip package has been installed):

```
import gym_graph_search
```

After this modification, DQN (for example) can then be trained on this env as follows:

```
python -m baselines.run --alg=deepq --env=graph-search-v0 --num_timesteps=1e5
```
