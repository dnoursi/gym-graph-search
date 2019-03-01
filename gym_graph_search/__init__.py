
from gym.envs.registration import register

register(
        id = "graph-search-ba-v0",
        entry_point="gym_graph_search.envs:BAGraphSearchEnv"
        )

register(
        id = "graph-search-rd-v0",
        entry_point="gym_graph_search.envs:RdGraphSearchEnv"
        )
