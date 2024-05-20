import numpy as np


def generate_service_hardware(n, bw_low, bw_high, cpu_low, cpu_high):
    hardware_data = np.zeros((n,2))
    for i in range(n):
        hardware_data[i][0] = int(np.random.uniform(bw_low, bw_high))
        hardware_data[i][1] = int(np.random.uniform(cpu_low, cpu_high))
    return hardware_data

def generate_cloud_hardware(bw_low, bw_high, cpu_low, cpu_high):
    hardware_data = np.zeros((2))
    hardware_data[0] = int(np.random.uniform(bw_low, bw_high))
    hardware_data[1] = int(np.random.uniform(cpu_low, cpu_high))
    return hardware_data

def generate_rsu_hardware(cpu_low, cpu_high):
    hardware_data = np.zeros((1))
    hardware_data[0] = int(np.random.uniform(cpu_low, cpu_high))
    return hardware_data

def shortest_queue(obs):
    new_queues = np.array([obs[0], obs[1]+obs[2], obs[3]+obs[4], obs[5]+obs[6]])
    return np.argmin(new_queues)
