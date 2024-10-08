import ciw.dists
import gymnasium as gym
import ciw
import numpy as np
from movement import create_random_walk, create_parked_coords
from custom_components import MovingTransmissionDistNew, ComputationDist, StationaryTransmissionDistNew, CustomSimulation, CloudCompDelay, CloudTransDelay
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
        self.service_vehicles_cpu = generate_service_cpu(2)
        self.rsu_hardware = generate_rsu_hardware()
        self.cloud_hardware = generate_cloud_hardware()
        if self.render_mode == "human":
            self.anim = Animator([walk_1,walk_2,walk_3,walk_4,walk_5],[self.parked_1,self.parked_2], self.service_vehicles_cpu, self.rsu_hardware, self.cloud_hardware)
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
            service_distributions=[MovingTransmissionDistNew(coords=walk_1),#client-trns-1       1
                           MovingTransmissionDistNew(coords=walk_2),        #client-trns-2       2
                           MovingTransmissionDistNew(coords=walk_3),        #client-trns-3       3
                           MovingTransmissionDistNew(coords=walk_4),        #client-trns-4       4
                           MovingTransmissionDistNew(coords=walk_5),        #client-trns-5       5
                           ciw.dists.Deterministic(value=0.0000001),        #rsu-trns            6
                           ComputationDist(mips=self.rsu_hardware[0]),      #rsu-cpu             7
                           CloudTransDelay(bw=1000),                        #trns-to-cloud       8
                           CloudCompDelay(self.cloud_hardware[1],0.05,0.2), #cloud-cpu           9
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
            number_of_servers=[1,1,1,1,1,1,1,1,float("Inf"),1,1,1,1]
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
        obs_task_cu = (np.array([self.Q.nodes[6].all_individuals[0].cu])-27.5)/(13010-27.5) #1
        obs_task_sz = (np.array([self.Q.nodes[6].all_individuals[0].sz])-20)/(40-20) #1
        obs_cpu = (np.concatenate(([self.cloud_hardware[1]],[self.rsu_hardware[0]],self.service_vehicles_cpu))-18375)/(71120-18375) #4
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
                rew += 5-total_delay
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
            obs_task_cu = (np.array([self.Q.nodes[6].all_individuals[0].cu])-27.5)/(13010-27.5) #1
            obs_task_sz = (np.array([self.Q.nodes[6].all_individuals[0].sz])-20)/(40-20) #1
            obs_cpu = (np.concatenate(([self.cloud_hardware[1]],[self.rsu_hardware[0]],self.service_vehicles_cpu))-18375)/(71120-18375) #4
            obs = np.concatenate((obs_queues_trans, obs_queues_cpu, obs_service_distance, obs_task_cu, obs_task_sz, obs_cpu),dtype=np.float32)
        else:
            obs = np.zeros((15))
        if self.render_mode == "human":
            if not ter:
                emitted_node = self.Q.nodes[6].all_individuals[0].data_records[0].node
                self.anim.add_frame(time, emitted_node, action, [obs_queues_trans*100, obs_queues_cpu*1000000, (obs_task_cu*(52040-110))+110, (obs_task_sz*(15-5))+5])
            else:
                self.anim.show_animation()
        return obs, rew, ter, tur, info



from stable_baselines3 import PPO
from helpers import shortest_queue
number_of_trials = 100
train_env = VFCOffloadingEnv(60)
#model = PPO("MlpPolicy", train_env, verbose=1, gamma=0.85).learn(300000)
#model.save("trained_models/env5-new")
model = PPO("MlpPolicy", train_env, verbose=1).load("trained_models/env5-300000")


def test_offloading_method(n, method_name):
    total_rew = 0
    total_delay = 0
    total_num_tasks = 0
    test_delays = []
    for i in range(n):
        env = VFCOffloadingEnv(60, render_mode=None)
        obs,_ = env.reset()
        ter = False
        tot_rew = 0
        while not ter:
                if method_name == "RL":
                    action = model.predict(obs, deterministic=True)[0]
                elif method_name == "Greedy":
                    action = shortest_queue(obs)
                elif method_name == "RSU":
                    action = 0
                elif method_name == "Cloud":
                    action = 1
                elif method_name == "Random":
                    action = env.action_space.sample()
                else:
                    print("INVALID METHOD")
                obs,rew,ter,_,_ = env.step(action)
                tot_rew += rew
        total_rew += tot_rew
        finished_tasks = env.Q.nodes[-1].all_individuals
        finished_tasks_details = [r.data_records for r in finished_tasks]
        delay = []
        for task in finished_tasks_details:
            delay.append(task[-1].exit_date - task[0].arrival_date)
        total_delay += sum(delay)/len(delay)
        total_num_tasks += len(delay)
        test_delays.append(sum(delay)/len(delay))
    print("--------------------------------------")
    print(method_name, "Average Reward:",total_rew/n)
    print(method_name, "Average Task Delay:",total_delay/n)
    print(method_name, "Average #Tasks Completed:",total_num_tasks/n)
    print(method_name, "Test Delays:", test_delays)
    with open("results/out-5-"+method_name+".txt", 'w') as f:
        for line in test_delays:
            f.write(f"{line}\n")


test_offloading_method(10, "RL")
test_offloading_method(10, "Random")
test_offloading_method(10, "Cloud")
test_offloading_method(10, "RSU")
test_offloading_method(10, "Greedy")