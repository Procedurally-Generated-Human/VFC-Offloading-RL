import numpy as np

def shortest_queue(obs):
    new_queues = np.array([obs[0], obs[1]+obs[2], obs[3]+obs[4], obs[5]+obs[6]])
    return np.argmin(new_queues)