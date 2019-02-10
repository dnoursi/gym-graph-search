# David Noursi

import gym
import gym_graph_search

import numpy as np
from copy import deepcopy

def main():
    env = gym.make("graph-search-v0", n=12, m0=6, m=6)
    print("Graph env is", env.graph_edges)
    return
    shortest_path = direct_dfs(env)
    print("DFS directly found the following shortest path", shortest_path)
    return

    nnodes = env.n
    target = env.target
    paths = dfs(env, nnodes, target)
    print("Stateful DFS found the following paths", paths)

def dfs_agent(neighbors, path, visited):
    unvisited_neighbors = list(set(neighbors) - visited)
    if not unvisited_neighbors:
        return dfs_backtrack_agent(neighbors, path), True
    return dfs_forward_agent(neighbors, unvisited_neighbors), False

def dfs_backtrack_agent(neighbors, path):
    desired_state = path[-2]
    #print(desired_state, neighbors)
    action = neighbors.index(desired_state)
    return action

# Tries to go directly forward
def dfs_forward_agent(neighbors, unvisited):
    n_actions = len(unvisited)
    i_action = np.random.choice(a=n_actions)
    return neighbors.index(unvisited[i_action])

def dfs(env, nnodes, target):
    solutions = []
    visited = set()
    nsteps = 0
    # Traverse every edge once?
    #total_nsteps = int(env.n * env.N / 2)

    obs = env.reset()
    path = [obs]

    while len(visited) < nnodes:
    #while nsteps < total_nsteps:
        #print(path, solutions)

        action, backtracking  = dfs_agent(env.get_action_space(), path, visited)
        obs, _, done, _ = env.step(action)

        if backtracking:
            #print(obs, path)
            assert obs == path[-2]
            path = path[:-1]
        else:
            nsteps += 1
            visited.add(obs)
            path.append(obs)

        if done:
            solutions.append(deepcopy(path))
    return solutions
    print(solutions)

def direct_dfs(env):
    graph = env.graph_edges
    target = env.target

    solutions = []

    paths = [(0,[0])]
    while paths:
        (root, path) = paths.pop()
        for neighbor in graph[root]:
            if neighbor in path:
                continue

            if neighbor == target:
                print(path, len(path), len(solutions))
                solutions.append(deepcopy(path+[neighbor]))
            else:
                paths.append((neighbor, deepcopy(path+[neighbor])))

    solutions_sort = [(len(s),s) for s in solutions]
    solutions_sort.sort()
    return solutions_sort[0][1]
    print(solutions)

if __name__ == "__main__":
    main()
