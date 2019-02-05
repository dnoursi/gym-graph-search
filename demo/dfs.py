# David Noursi

import gym
import gym_graph_search

import numpy as np
from copy import deepcopy


def main():
    env = gym.make("graph-search-v0")
    shortest_path = direct_dfs(env)
    print("DFS directly found the following shortest path", shortest_path)
    nnodes = env.N
    target = nnodes - 1
    paths = dfs(env, nnodes, target)

def dfs_agent(neighbors, path, visited):
    unvisited_neighbors = list(set(neighbors) - visited)
    if not unvisited_neighbors:
        return dfs_backtrack_agent(neighbors, path, visited), True
    return dfs_forward_agent(neighbors, unvisited_neighbors), False

def dfs_backtrack_agent(neighbors, path, visited):
    desired_state = path[-2]
    action = neighbors.index(desired_state)
    return action

# Tries to go directly forward
def dfs_forward_agent(neighbors, unvisited):
    n_actions = len(unvisited)
    i_action = np.random.choice(a=n_actions)
    return neighbors.index(unvisited[i_action])

def dfs(env, nnodes, target):

    obs = env.reset()

    solutions = []

    visited = set()
    path = []

    while len(visited) < nnodes:
        print(path, solutions)

        action, backtracking  = dfs_agent(env.get_action_space(), path, visited)
        if backtracking:
            path = path[:-1]
        else:
            visited.add(obs)
            path.append(obs)
        obs, reward, done, _ = env.step(action)
        if backtracking:
            assert obs == path[-1]
        if done:
            solutions.append(deepcopy(path))
    print(solutions)
    return solutions

def direct_dfs(env):
    graph = env.graph_edges
    target = env.N - 1

    solutions = []

    paths = [(0,[0])]
    while paths:
        (root, path) = paths.pop()
        for neighbor in graph[root]:
            if neighbor in path:
                continue

            if neighbor == target:
                solutions.append(deepcopy(path+[neighbor]))
            else:
                paths.append((neighbor, deepcopy(path+[neighbor])))

    solutions_sort = [(len(s),s) for s in solutions]
    solutions_sort.sort()
    return solutions_sort[0][1]
    print(solutions)

if __name__ == "__main__":
    main()







