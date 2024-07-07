import numpy as np


def generate_service_cpu(n):
    hardware_data = np.zeros((n))
    for i in range(n):
        hardware_data[i] = np.random.choice([71120,18375,15750,11750,10250])
    return hardware_data

def generate_cloud_hardware():
    hardware_data = np.zeros((2))
    hardware_data[0] = 0
    hardware_data[1] = np.random.choice([100_000])
    return hardware_data

def generate_rsu_hardware():
    hardware_data = np.zeros((1))
    hardware_data[0] = np.random.choice([18375,15750,11750,10250])
    return hardware_data

def shortest_queue(obs):
    new_queues = np.array([obs[3], obs[4], obs[5], obs[6]])
    return np.argmin(new_queues)
