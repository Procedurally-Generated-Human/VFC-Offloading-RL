import numpy as np


def generate_service_cpu(n):
    hardware_data = np.zeros((n))
    for i in range(n):
        hardware_data[i] = np.random.choice([71120,42820,18375])
    return hardware_data

def generate_cloud_hardware():
    hardware_data = np.zeros((2))
    hardware_data[0] = 0
    hardware_data[1] = np.random.choice([100_000])
    return hardware_data

def generate_rsu_hardware():
    hardware_data = np.zeros((1))
    hardware_data[0] = np.random.choice([18375])
    return hardware_data

def shortest_queue(obs):
    new_queues = np.array([obs[3], obs[0]+obs[4], obs[1]+obs[5], obs[2]+obs[6]])
    action = np.argmin(new_queues)
    return action

def shortest_queue10(obs):
    new_queues = np.array([obs[5], obs[0]+obs[6], obs[7]+obs[1], obs[8]+obs[2], obs[9]+obs[3], obs[10]+obs[4]])
    action = np.argmin(new_queues)
    return action

def shortest_queue20(obs):
    new_queues = np.array([obs[9], obs[0]+obs[10], obs[1]+obs[11], obs[2]+obs[12], obs[3]+obs[13], obs[14]+obs[4], obs[5]+obs[15], obs[6]+obs[16], obs[7]+obs[17], obs[8]+obs[18]])
    action = np.argmin(new_queues)
    return action