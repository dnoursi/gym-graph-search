

from gym.envs.registration import register

register(
        id = "graph-search-v0",
        entry_point="gym_graph_search.envs:GraphSearchEnv"
        )

