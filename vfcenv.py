import ciw.dists
import gymnasium as gym
import ciw
import numpy as np
import matplotlib.pyplot as plt 
from movement import create_random_walk, create_parked_coords
from custom_components import MovingTransmissionDistNew, ComputationDist, StationaryTransmissionDistNew, CustomSimulation, CustomNode, CustomArrival, CustomIndividual
from animate import Animator
from helpers import generate_service_cpu, generate_rsu_hardware, generate_cloud_hardware
np.set_printoptions(suppress=True)


class VFCOffloadingEnv(gym.Env):

    def __init__(self, n_timesteps, render_mode=None) -> None:
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space =  gym.spaces.Box(0,10,(15,))
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
        self.parked_1 = create_parked_coords()
        self.parked_2 = create_parked_coords()
        self.service_vehicles_cpu = generate_service_cpu(2,3500,4000)
        self.rsu_hardware = generate_rsu_hardware(4500,4500)
        self.cloud_hardware = generate_cloud_hardware(1000,1000,6000,6000)
        if self.render_mode == "human":
            self.anim = Animator([walk_1,walk_2,walk_3,walk_4,walk_5],[self.parked_1,self.parked_2], self.service_vehicles_cpu, self.rsu_hardware, self.cloud_hardware)
        self.N = ciw.create_network(
            arrival_distributions=[ciw.dists.Exponential(rate=1),   #client-trns-1       1
                           ciw.dists.Exponential(rate=1),           #client-trns-2       2
                           ciw.dists.Exponential(rate=1),           #client-trns-3       3
                           ciw.dists.Exponential(rate=1),           #client-trns-4       4
                           ciw.dists.Exponential(rate=1),           #client-trns-5       5
                           None,                                    #rsu-trns            6
                           None,                                    #rsu-cpu             7
                           None,                                    #trns-to-cloud       8
                           None,                                    #cloud-cpu           9
                           None,                                    #trns-to-service-1  10
                           None,                                    #service-cpu-1      11
                           None,                                    #trns-to-service-2  12
                           None,                                    #service-cpu-2      13
                           ],                                       ######################
            service_distributions=[MovingTransmissionDistNew(coords=walk_1),#client-trns-1       1
                           MovingTransmissionDistNew(coords=walk_2),        #client-trns-2       2
                           MovingTransmissionDistNew(coords=walk_3),        #client-trns-3       3
                           MovingTransmissionDistNew(coords=walk_4),        #client-trns-4       4
                           MovingTransmissionDistNew(coords=walk_5),        #client-trns-5       5
                           ciw.dists.Deterministic(value=0.0000001),        #rsu-trns            6
                           ComputationDist(mips=self.rsu_hardware[0]),      #rsu-cpu             7
                           ciw.dists.Normal(mean=0.5, sd=1),          #trns-to-cloud       8
                           ComputationDist(mips=self.cloud_hardware[1]),    #cloud-cpu           9
                           StationaryTransmissionDistNew(x=self.parked_1[0],y=self.parked_1[1]),        #trns-to-service-1  10
                           ComputationDist(mips=self.service_vehicles_cpu[0]),                        #service-cpu-1      11
                           StationaryTransmissionDistNew(x=self.parked_2[0],y=self.parked_2[1]),        #trns-to-service-2  12
                           ComputationDist(mips=self.service_vehicles_cpu[1]),                        #service-cpu-2      13
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
            number_of_servers=[1,1,1,1,1,1,1,1,100,1,1,1,1]
            )
        self.Q = CustomSimulation(self.N)
        self.Q.simulate_until_decision(self.n_timesteps)
        rsu_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[7].all_individuals))
        cloud_trans_queue = sum(tuple(x.sz for x in self.Q.nodes[8].all_individuals))
        cloud_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[9].all_individuals))
        service_1_trans_queue = sum(tuple(x.sz for x in self.Q.nodes[10].all_individuals))
        service_1_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[11].all_individuals))
        service_2_trans_queue = sum(tuple(x.sz for x in self.Q.nodes[12].all_individuals))
        service_2_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[13].all_individuals))

        obs_service_distance = np.array([np.linalg.norm((self.parked_1[0],self.parked_1[1]) - np.array([500,500])), np.linalg.norm((self.parked_2[0],self.parked_2[1]) - np.array([500,500]))])/707
        obs_queues_cpu = np.array([rsu_cpu_queue, cloud_cpu_queue, service_1_cpu_queue, service_2_cpu_queue], dtype=np.float32)/1000000
        obs_queues_trans = np.array([cloud_trans_queue, service_1_trans_queue, service_2_trans_queue], dtype=np.float32)/100
        obs_task_cu = (np.array([self.Q.nodes[6].all_individuals[0].cu])-2690)/(54600-2690) #1
        obs_task_sz = (np.array([self.Q.nodes[6].all_individuals[0].sz])-5)/(30-5) #1
        obs_cpu = (np.concatenate(([self.cloud_hardware[1]],[self.rsu_hardware[0]],self.service_vehicles_cpu))-10250)/(71120-10250) #4
        obs = np.concatenate((obs_queues_trans, obs_queues_cpu, obs_service_distance, obs_task_cu, obs_task_sz, obs_cpu),dtype=np.float32)
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
        if self.Q.current_time >= self.n_timesteps:
            ter = True
        if not ter:
            rsu_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[7].all_individuals))
            cloud_trans_queue = sum(tuple(x.sz for x in self.Q.nodes[8].all_individuals))
            cloud_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[9].all_individuals))
            service_1_trans_queue = sum(tuple(x.sz for x in self.Q.nodes[10].all_individuals))
            service_1_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[11].all_individuals))
            service_2_trans_queue = sum(tuple(x.sz for x in self.Q.nodes[12].all_individuals))
            service_2_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[13].all_individuals))

            obs_service_distance = np.array([np.linalg.norm((self.parked_1[0],self.parked_1[1]) - np.array([500,500])), np.linalg.norm((self.parked_2[0],self.parked_2[1]) - np.array([500,500]))])/707
            obs_queues_cpu = np.array([rsu_cpu_queue,cloud_cpu_queue,service_1_cpu_queue,service_2_cpu_queue], dtype=np.float32)/1000000
            obs_queues_trans = np.array([cloud_trans_queue,service_1_trans_queue,service_2_trans_queue], dtype=np.float32)/100
            obs_task_cu = (np.array([self.Q.nodes[6].all_individuals[0].cu])-2690)/(54600-2690) #1
            obs_task_sz = (np.array([self.Q.nodes[6].all_individuals[0].sz])-5)/(30-5) #1
            obs_cpu = (np.concatenate(([self.cloud_hardware[1]],[self.rsu_hardware[0]],self.service_vehicles_cpu))-10250)/(71120-10250) #4
            obs = np.concatenate((obs_queues_trans, obs_queues_cpu, obs_service_distance, obs_task_cu, obs_task_sz, obs_cpu),dtype=np.float32)
        else:
            obs = np.zeros((15))
        if self.render_mode == "human":
            if not ter:
                emitted_node = self.Q.nodes[6].all_individuals[0].data_records[0].node
                self.anim.add_frame(time, emitted_node, action, [obs_queues_trans*100, obs_queues_cpu*1000000, (obs_task_cu*(54600-2690))+2690, (obs_task_sz*(30-5))+5])
            else:
                self.anim.show_animation()
        return obs, rew, ter, tur, info


from stable_baselines3 import PPO
from helpers import shortest_queue
train_env = VFCOffloadingEnv(100)
model = PPO("MlpPolicy", train_env, verbose=1).learn(400000)
model.save("400000ppo")
#model = PPO("MlpPolicy", train_env, verbose=1).load("400000ppo")
com_rew = 0
for i in range(1):
    print(i)
    env = VFCOffloadingEnv(100, render_mode=None)
    obs,_ = env.reset()
    ter = False
    tot_rew = 0
    action_store = []
    while not ter:
            #action = 1
            #action = env.action_space.sample()
            #print("Current State:", obs)
            action = model.predict(obs, deterministic=True)[0]
            #action = shortest_queue(obs)
            action_store.append(action)
            #print("Action Taken:", action)
            obs,rew,ter,_,_ = env.step(action)
            tot_rew += rew
            #print("Next State:", obs)
            #print("Reward:",rew)
            #print("Terminated:",ter)
            #print("-----------------------------")
    com_rew += tot_rew
print(com_rew/1)
print(action_store)
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