import numpy as np


def generate_service_cpu(n, cpu_low, cpu_high):
    hardware_data = np.zeros((2))
    for i in range(n):
        hardware_data[i] = np.random.choice([71120,18375,15750,11750,10250])
    return hardware_data

def generate_cloud_hardware(bw_low, bw_high, cpu_low, cpu_high):
    hardware_data = np.zeros((2))
    hardware_data[0] = int(np.random.uniform(bw_low, bw_high))
    hardware_data[1] = np.random.choice([100_000])
    return hardware_data

def generate_rsu_hardware(cpu_low, cpu_high):
    hardware_data = np.zeros((1))
    hardware_data[0] = np.random.choice([18375,15750,11750,10250])
    return hardware_data

def shortest_queue(obs):
    new_queues = np.array([obs[3], obs[0]+obs[4], obs[1]+obs[5], obs[2]+obs[6]])
    return np.argmin(new_queues)
