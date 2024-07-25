import pandas as pd
import matplotlib.pyplot as plt


def read_and_average_results(file_name):
    with open(file_name+".txt") as f:
        data = [float(line.rstrip('\n')) for line in f]
        average = sum(data)/len(data)
        return average

rl_5 = read_and_average_results("out-5-RL")
rl_10 = read_and_average_results("out-10-RL")
rl_20 = read_and_average_results("out-20-RL")
gr_5 = read_and_average_results("out-5-Greedy")
gr_10 = read_and_average_results("out-10-Greedy")
gr_20 = read_and_average_results("out-20-Greedy")
cl_5 = read_and_average_results("out-5-Cloud")
cl_10 = read_and_average_results("out-10-Cloud")
cl_20 = read_and_average_results("out-20-Cloud")
rn_5 = read_and_average_results("out-5-Random")
rn_10 = read_and_average_results("out-10-Random")
rn_20 = read_and_average_results("out-20-Random")
print(cl_10)

rl_values = [rl_5, rl_10, rl_20]
greedy_values = [gr_5, gr_10, gr_20]
cloud_values = [cl_5, cl_10, cl_20]
random_values = [rn_5, rn_10, rn_20]


index = ['Scenario 1', 'Scenario 2', 'Scenario 3',]
df = pd.DataFrame({'DRL': rl_values,
                    'Greedy': greedy_values,
                    "Cloud-Only":cloud_values,
                    "Random":random_values,
                    }, index=index)
ax = df.plot.bar(rot=0, width=0.8, color={"DRL": "blue", "Greedy": "green", "Cloud-Only": "black", "Random": "red"})
ax.set_axisbelow(True)
bars = ax.patches
hatches = ["\\\\","\\\\","\\\\","//","//","//","..","..","..","..","..",".."]
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)
ax.yaxis.grid(color='gray', linestyle='dashed')
plt.ylabel("Avg. Task Delay (s)")
plt.legend(fontsize="large") # using a named size
plt.show()
