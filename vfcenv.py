import gymnasium as gym
import ciw
import numpy as np
import matplotlib.pyplot as plt 

from movement import create_random_walk
from custom_components import MovingTransmissionDist, ComputationDist, StationaryTransmissionDist, CustomSimulation, CustomNode, CustomArrival, CustomIndividual
from animate import Animator



class VFCOffloadingEnv(gym.Env):

    def __init__(self, n_timesteps, render_mode=None) -> None:
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space =  gym.spaces.Box(0,10,(9,))
        self.n_timesteps = n_timesteps
        self.current_timestep = 0
        self.render_mode = render_mode
        super().__init__()

    def reset(self, seed=None):
        ciw.seed(seed)
        walk_1 = create_random_walk(self.n_timesteps)
        walk_2 = create_random_walk(self.n_timesteps)
        walk_3 = create_random_walk(self.n_timesteps)
        walk_4 = create_random_walk(self.n_timesteps)
        walk_5 = create_random_walk(self.n_timesteps)
        if self.render_mode == "human":
            self.anim = Animator([walk_1,walk_2,walk_3,walk_4,walk_5])
        self.N = ciw.create_network(
            arrival_distributions=[ciw.dists.Exponential(rate=3),   #client-trns-1       1
                           ciw.dists.Exponential(rate=3),           #client-trns-2       2
                           ciw.dists.Exponential(rate=3),           #client-trns-3       3
                           ciw.dists.Exponential(rate=3),           #client-trns-4       4
                           ciw.dists.Exponential(rate=3),           #client-trns-5       5
                           None,                                    #rsu-trns            6
                           None,                                    #rsu-cpu             7
                           None,                                    #trns-to-cloud       8
                           None,                                    #cloud-cpu           9
                           None,                                    #trns-to-service-1  10
                           None,                                    #service-cpu-1      11
                           None,                                    #trns-to-service-2  12
                           None,                                    #service-cpu-2      13
                           ],                                       ######################
            service_distributions=[MovingTransmissionDist(bw=300,coords=walk_1),#client-trns-1       1
                           MovingTransmissionDist(bw=400,coords=walk_2),        #client-trns-2       2
                           MovingTransmissionDist(bw=500,coords=walk_3),        #client-trns-3       3
                           MovingTransmissionDist(bw=600,coords=walk_4),        #client-trns-4       4
                           MovingTransmissionDist(bw=500,coords=walk_5),        #client-trns-5       5
                           ciw.dists.Deterministic(value=0.0000001),                #rsu-trns            6
                           ComputationDist(mips=4500),                        #rsu-cpu             7
                           StationaryTransmissionDist(bw=1000,x=0,y=0),       #trns-to-cloud       8
                           ComputationDist(mips=6000),                        #cloud-cpu           9
                           StationaryTransmissionDist(bw=800,x=0,y=0),        #trns-to-service-1  10
                           ComputationDist(mips=4000),                        #service-cpu-1      11
                           StationaryTransmissionDist(bw=800,x=0,y=0),        #trns-to-service-2  12
                           ComputationDist(mips=3500),                        #service-cpu-2      13
                           ],
                       #1,2,3,4,5,6,7,8,9,0,1,2,3           
            routing = [[0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
               [0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
               [0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
               [0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
               [0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
               [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
               [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
               [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0],
               [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
               [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0],
               [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
               [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0],
               [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],
            number_of_servers=[1,1,1,1,1,1,1,1,1,1,1,1,1]
            )
        self.Q = CustomSimulation(self.N)
        self.Q.simulate_until_decision(self.n_timesteps)
        task_sz = (self.Q.nodes[6].all_individuals[0].sz - 80)/(120-80)
        task_cu = (self.Q.nodes[6].all_individuals[0].cu - 800)/(1200-800)
        obs = np.array(self.Q.statetracker.history[-1][1], dtype=np.float32)[6:]
        obs = np.concatenate((obs, np.array([task_cu,task_sz], dtype=np.float32)))
        info = {}
        self.calculated_inds = []
        return obs, info
    
    def step(self, action):
        self.Q.nodes[6].decision = action
        time = self.Q.simulate_until_decision(self.n_timesteps)
        exitnode = self.Q.nodes[-1]
        rew = 0
        for ind in exitnode.all_individuals:
            if ind not in self.calculated_inds:
                ind_recs = ind.data_records
                arrival_date = ind_recs[0].arrival_date
                exit_date = ind_recs[-1].exit_date
                total_delay = exit_date - arrival_date
                rew += 1/total_delay
                self.calculated_inds.append(ind)
        info = {}
        ter = False
        tur = False
        obs = np.array([0,0,0,0,0,0,0,0,0])
        if self.Q.current_time >= self.n_timesteps:
            ter = True
        if not ter:
            task_sz = (self.Q.nodes[6].all_individuals[0].sz - 80)/(120-80)
            task_cu = (self.Q.nodes[6].all_individuals[0].cu - 800)/(1200-800)
            obs = np.array(self.Q.statetracker.history[-1][1], dtype=np.float32)[6:]
            obs = np.concatenate((obs, np.array([task_cu,task_sz], dtype=np.float32)))
        if self.render_mode == "human":
            self.anim.add_frame(time)
            if ter:
                self.anim.show_animation()
        return obs, rew, ter, tur, info


from stable_baselines3 import PPO
from helpers import shortest_queue
train_env = VFCOffloadingEnv(100)
#model = PPO("MlpPolicy", train_env, verbose=1).learn(200000)
#model.save("200000fullvfc")
#model = PPO("MlpPolicy", train_env, verbose=1).load("200000fullvfc")
com_rew = 0
for i in range(1):
    print(i)
    env = VFCOffloadingEnv(100, render_mode="human")
    obs,_ = env.reset()
    ter = False
    tot_rew = 0
    while not ter:
            #action = 1
            #action = env.action_space.sample()
            #print("Current State:", obs)
            #print("Action Taken:", action)
            #action = model.predict(obs, deterministic=True)[0]
            action = shortest_queue(obs)
            #print("action",action)
            obs,rew,ter,_,_ = env.step(action)
            tot_rew += rew
            #print("Next State:", obs)
            # print("Reward:",rew)
            # print("Terminated:",ter)
            #print("-----------------------------")
    com_rew += tot_rew
print(com_rew/20)
ts = [ts[0] for ts in env.Q.statetracker.history]
n1 = [ts[1][0] for ts in env.Q.statetracker.history]
n2 = [ts[1][1] for ts in env.Q.statetracker.history]
n3 = [ts[1][2] for ts in env.Q.statetracker.history]
n4 = [ts[1][3] for ts in env.Q.statetracker.history]
n5 = [ts[1][4] for ts in env.Q.statetracker.history]
n6 = [ts[1][5] for ts in env.Q.statetracker.history]
n7 = [ts[1][6] for ts in env.Q.statetracker.history]
n8 = [ts[1][7] for ts in env.Q.statetracker.history]
n9 = [ts[1][8] for ts in env.Q.statetracker.history]
n10 = [ts[1][9] for ts in env.Q.statetracker.history]
n11 = [ts[1][10] for ts in env.Q.statetracker.history]
n12 = [ts[1][11] for ts in env.Q.statetracker.history]
n13 = [ts[1][12] for ts in env.Q.statetracker.history]
import matplotlib.pyplot as plt 

plt.title("Client Vehicles")
plt.plot(ts, n1, label="client-1"); 
plt.plot(ts, n2, label="client-2"); 
plt.plot(ts, n3, label="client-3"); 
plt.plot(ts, n4, label="client-4"); 
plt.plot(ts, n5, label="client-5"); 
plt.legend(loc="upper left")
plt.show()

plt.title("RSU")
plt.plot(ts, n6, label="rsu-decision");
plt.plot(ts, n7, label="rsu-cpu");
plt.plot(ts, n8, label="rsu-trns-cloud");
plt.plot(ts, n9, label="cloud-cpu");
plt.plot(ts, n10, label="rsu-trns-service-1");
plt.plot(ts, n11, label="service-1-cpu");
plt.plot(ts, n12, label="rsu-trns-service-2");
plt.plot(ts, n13, label="service-2-cpu");
plt.legend(loc="upper left")
plt.show()