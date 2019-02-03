# David Noursi

import gym
import gym_graph_search


def main():
    env = gym.make("graph-search-v0")

    nnodes = env.N
    target = nnodes - 1
    shortest_path = dfs(env, nnodes, target)

def dfs(env, nnodes):
    visited = set()
    current = 0

    for neighbor in env.get_action_space():
        return

if __name__ == "__main__":
    main()







