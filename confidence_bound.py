import numpy as np
import scipy.stats

def mean_confidence_interval(filename, confidence=0.95):
    with open(filename+".txt", 'r') as file:
        numbers = [float(line.strip()) for line in file]
    numbers_array = np.array(numbers)
    a = 1.0 * np.array(numbers_array)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    print(filename,round(m,2),"+-",round(h,2))



mean_confidence_interval("out-5-RL")
mean_confidence_interval("out-10-RL")
mean_confidence_interval("out-20-RL")
mean_confidence_interval("out-5-Greedy")
mean_confidence_interval("out-10-Greedy")
mean_confidence_interval("out-20-Greedy")
mean_confidence_interval("out-5-Cloud")
mean_confidence_interval("out-10-Cloud")
mean_confidence_interval("out-20-Cloud")
mean_confidence_interval("out-5-Random")
mean_confidence_interval("out-10-Random")
mean_confidence_interval("out-20-Random")