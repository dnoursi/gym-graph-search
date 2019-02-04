# David Noursi

import gym
import gym_graph_search

import numpy as np
import copy


def main():
    env = gym.make("graph-search-v0")
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
            solutions.append(copy.deepcopy(path))
    print(solutions)
    return solutions

def direct_dfs(env, nnodes, root, target):
    solutions = []

    paths = [(0,[0])]
    while paths:
        (root, path) = paths.pop()
        for neighbor in env.get_action_space():
            if neighbor in path:
                continue

            if neighbor == target:
                solutions.append(path+[neighbor])
            else:
                paths.append((neighbor, path+[neighbor]))

if __name__ == "__main__":
    main()







