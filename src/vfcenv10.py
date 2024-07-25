import ciw.dists
import gymnasium as gym
import ciw
import numpy as np
import matplotlib.pyplot as plt 
from movement import create_random_walk, create_parked_coords
from custom_components import MovingTransmissionDistNew, ComputationDist, StationaryTransmissionDistNew, CustomSimulation10, CloudTransDelay, CloudCompDelay #CNG
from animate import Animator
from helpers import generate_service_cpu, generate_rsu_hardware, generate_cloud_hardware
np.set_printoptions(suppress=True)


class VFCOffloadingEnv10(gym.Env):

    def __init__(self, n_timesteps, render_mode=None) -> None:
        self.action_space = gym.spaces.Discrete(6) #CNG
        self.observation_space =  gym.spaces.Box(0,10,(23,)) #CNG
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
        walk_5 = create_random_walk(self.n_timesteps) #CNG
        walk_6 = create_random_walk(self.n_timesteps)
        walk_7 = create_random_walk(self.n_timesteps)
        walk_8 = create_random_walk(self.n_timesteps)
        walk_9 = create_random_walk(self.n_timesteps)
        walk_10 = create_random_walk(self.n_timesteps)
        self.parked_1 = create_parked_coords()
        self.parked_2 = create_parked_coords()
        self.parked_3 = create_parked_coords() #CNG
        self.parked_4 = create_parked_coords()
        self.service_vehicles_cpu = generate_service_cpu(4) #CNG
        self.rsu_hardware = generate_rsu_hardware()
        self.cloud_hardware = generate_cloud_hardware()
        if self.render_mode == "human":
            self.anim = Animator([walk_1,walk_2,walk_3,walk_4,walk_5,walk_6,walk_7,walk_8,walk_9,walk_10],[self.parked_1,self.parked_2,self.parked_3,self.parked_4], self.service_vehicles_cpu, self.rsu_hardware, self.cloud_hardware) #CNG
        self.N = ciw.create_network(
            arrival_distributions=[ciw.dists.Exponential(rate=3),   #client-trns-1       1
                           ciw.dists.Exponential(rate=3),           #client-trns-2       2
                           ciw.dists.Exponential(rate=3),           #client-trns-3       3
                           ciw.dists.Exponential(rate=3),           #client-trns-4       4
                           ciw.dists.Exponential(rate=3),           #client-trns-5       5
                           ciw.dists.Exponential(rate=3),           #client-trns-6       6 #CNG
                           ciw.dists.Exponential(rate=3),           #client-trns-7       7
                           ciw.dists.Exponential(rate=3),           #client-trns-8       8
                           ciw.dists.Exponential(rate=3),           #client-trns-9       9
                           ciw.dists.Exponential(rate=3),           #client-trns-10     10
                           None,                                    #rsu-trns           11
                           None,                                    #rsu-cpu            12
                           None,                                    #trns-to-cloud      13
                           None,                                    #cloud-cpu          14
                           None,                                    #trns-to-service-1  15
                           None,                                    #service-cpu-1      16
                           None,                                    #trns-to-service-2  17
                           None,                                    #service-cpu-2      18
                           None,                                    #trns-to-service-3  19 #CNG
                           None,                                    #service-cpu-3      20
                           None,                                    #trns-to-service-4  21
                           None,                                    #service-cpu-4      22
                           ],                                       ######################
            service_distributions=[MovingTransmissionDistNew(coords=walk_1),#client-trns-1       1
                           MovingTransmissionDistNew(coords=walk_2),        #client-trns-2       2
                           MovingTransmissionDistNew(coords=walk_3),        #client-trns-3       3
                           MovingTransmissionDistNew(coords=walk_4),        #client-trns-4       4
                           MovingTransmissionDistNew(coords=walk_5),        #client-trns-5       5
                           MovingTransmissionDistNew(coords=walk_6),        #client-trns-6       6 #CNG
                           MovingTransmissionDistNew(coords=walk_7),        #client-trns-7       7
                           MovingTransmissionDistNew(coords=walk_8),        #client-trns-8       8
                           MovingTransmissionDistNew(coords=walk_9),        #client-trns-9       9
                           MovingTransmissionDistNew(coords=walk_10),       #client-trns-10     10
                           ciw.dists.Deterministic(value=0.0000001),        #rsu-trns           11
                           ComputationDist(mips=self.rsu_hardware[0]),      #rsu-cpu            12
                           CloudTransDelay(bw=1000),                        #trns-to-cloud      13
                           CloudCompDelay(self.cloud_hardware[1],0.05,0.2), #cloud-cpu          14
                           StationaryTransmissionDistNew(x=self.parked_1[0],y=self.parked_1[1]),      #trns-to-service-1    15
                           ComputationDist(mips=self.service_vehicles_cpu[0]),                        #service-cpu-1        16
                           StationaryTransmissionDistNew(x=self.parked_2[0],y=self.parked_2[1]),      #trns-to-service-2    17
                           ComputationDist(mips=self.service_vehicles_cpu[1]),                        #service-cpu-2        18
                           StationaryTransmissionDistNew(x=self.parked_3[0],y=self.parked_3[1]),      #trns-to-service-3    19 #CNG
                           ComputationDist(mips=self.service_vehicles_cpu[2]),                        #service-cpu-3        20
                           StationaryTransmissionDistNew(x=self.parked_4[0],y=self.parked_4[1]),      #trns-to-service-4    21
                           ComputationDist(mips=self.service_vehicles_cpu[3]),                        #service-cpu-4        22
                           ],
                                  
            routing = [ #CNG
                               #1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 
                list(np.float_([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])), #1
                list(np.float_([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])), #2
                list(np.float_([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])), #3
                list(np.float_([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])), #4
                list(np.float_([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])), #5
                list(np.float_([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])), #6
                list(np.float_([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])), #7
                list(np.float_([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])), #8
                list(np.float_([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])), #9
                list(np.float_([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])), #10
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #11
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #12
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])), #13
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #14
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0])), #15
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #16
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])), #17
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #18
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0])), #19
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #20
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])), #21
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #22
               ],
            number_of_servers=[1,1,1,1,1,1,1,1,1,1,1,1,1,float("Inf"),1,1,1,1,1,1,1,1] #CNG
            )
        self.Q = CustomSimulation10(self.N) #CNG
        self.Q.simulate_until_decision(self.n_timesteps)
        rsu_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[12].all_individuals)) #CNG
        cloud_trans_queue = sum(tuple(x.sz for x in self.Q.nodes[13].all_individuals)) #CNG
        cloud_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[14].all_individuals)) #CNG
        service_1_trans_queue = sum(tuple(x.sz for x in self.Q.nodes[15].all_individuals)) #CNG
        service_1_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[16].all_individuals)) #CNG
        service_2_trans_queue = sum(tuple(x.sz for x in self.Q.nodes[17].all_individuals)) #CNG
        service_2_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[18].all_individuals)) #CNG
        service_3_trans_queue = sum(tuple(x.sz for x in self.Q.nodes[19].all_individuals)) #CNG
        service_3_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[20].all_individuals)) #CNG
        service_4_trans_queue = sum(tuple(x.sz for x in self.Q.nodes[21].all_individuals)) #CNG
        service_4_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[22].all_individuals)) #CNG
        obs_service_distance = np.array([np.linalg.norm((self.parked_1[0],self.parked_1[1]) - np.array([500,500])), np.linalg.norm((self.parked_2[0],self.parked_2[1]) - np.array([500,500])), np.linalg.norm((self.parked_3[0],self.parked_3[1]) - np.array([500,500])), np.linalg.norm((self.parked_4[0],self.parked_4[1]) - np.array([500,500]))])/707 #CNG
        obs_queues_cpu = np.array([rsu_cpu_queue, cloud_cpu_queue, service_1_cpu_queue, service_2_cpu_queue, service_3_cpu_queue, service_4_cpu_queue], dtype=np.float32)/1000000 #CNG
        obs_queues_trans = np.array([cloud_trans_queue, service_1_trans_queue, service_2_trans_queue, service_3_trans_queue, service_4_trans_queue], dtype=np.float32)/100 #CNG
        obs_task_cu = (np.array([self.Q.nodes[11].all_individuals[0].cu])-27.5)/(13010-27.5) #CGD
        obs_task_sz = (np.array([self.Q.nodes[11].all_individuals[0].sz])-20)/(40-20) #CGD
        obs_cpu = (np.concatenate(([self.cloud_hardware[1]],[self.rsu_hardware[0]],self.service_vehicles_cpu))-18375)/(71120-18375) 
        obs = np.concatenate((obs_queues_trans, obs_queues_cpu, obs_service_distance, obs_task_cu, obs_task_sz, obs_cpu),dtype=np.float32)
        info = {}
        self.calculated_inds = []
        return obs, info
    
    def step(self, action):
        self.Q.nodes[11].decision = action #CGD
        time = self.Q.simulate_until_decision(self.n_timesteps)
        exitnode = self.Q.nodes[-1]
        rew = 0
        for ind in exitnode.all_individuals:
            if ind not in self.calculated_inds:
                ind_recs = ind.data_records
                arrival_date = ind_recs[0].arrival_date
                exit_date = ind_recs[-1].exit_date
                total_delay = exit_date - arrival_date
                #rew += 1/total_delay
                rew += 5 - total_delay
                self.calculated_inds.append(ind)
        info = {}
        ter = False
        tur = False
        if self.Q.current_time >= self.n_timesteps:
            ter = True
        if not ter:
            rsu_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[12].all_individuals)) #CNG
            cloud_trans_queue = sum(tuple(x.sz for x in self.Q.nodes[13].all_individuals)) #CNG
            cloud_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[14].all_individuals)) #CNG
            service_1_trans_queue = sum(tuple(x.sz for x in self.Q.nodes[15].all_individuals)) #CNG
            service_1_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[16].all_individuals)) #CNG
            service_2_trans_queue = sum(tuple(x.sz for x in self.Q.nodes[17].all_individuals)) #CNG
            service_2_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[18].all_individuals)) #CNG
            service_3_trans_queue = sum(tuple(x.sz for x in self.Q.nodes[19].all_individuals)) #CNG
            service_3_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[20].all_individuals)) #CNG
            service_4_trans_queue = sum(tuple(x.sz for x in self.Q.nodes[21].all_individuals)) #CNG
            service_4_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[22].all_individuals)) #CNG
            obs_service_distance = np.array([np.linalg.norm((self.parked_1[0],self.parked_1[1]) - np.array([500,500])), np.linalg.norm((self.parked_2[0],self.parked_2[1]) - np.array([500,500])), np.linalg.norm((self.parked_3[0],self.parked_3[1]) - np.array([500,500])), np.linalg.norm((self.parked_4[0],self.parked_4[1]) - np.array([500,500]))])/707 #CNG
            obs_queues_cpu = np.array([rsu_cpu_queue, cloud_cpu_queue, service_1_cpu_queue, service_2_cpu_queue, service_3_cpu_queue, service_4_cpu_queue], dtype=np.float32)/1000000 #CNG
            obs_queues_trans = np.array([cloud_trans_queue, service_1_trans_queue, service_2_trans_queue, service_3_trans_queue, service_4_trans_queue], dtype=np.float32)/100 #CNG
            obs_task_cu = (np.array([self.Q.nodes[11].all_individuals[0].cu])-27.5)/(13010-27.5) #CGD
            obs_task_sz = (np.array([self.Q.nodes[11].all_individuals[0].sz])-20)/(40-20) #CGD
            obs_cpu = (np.concatenate(([self.cloud_hardware[1]],[self.rsu_hardware[0]],self.service_vehicles_cpu))-18375)/(71120-18375)
            obs = np.concatenate((obs_queues_trans, obs_queues_cpu, obs_service_distance, obs_task_cu, obs_task_sz, obs_cpu),dtype=np.float32)
        else:
            obs = np.zeros((23)) #CND
        if self.render_mode == "human":
            if not ter:
                emitted_node = self.Q.nodes[11].all_individuals[0].data_records[0].node #CGD
                self.anim.add_frame(time, emitted_node, action, [obs_queues_trans*100, obs_queues_cpu*1000000, (obs_task_cu*(54600-2690))+2690, (obs_task_sz*(30-5))+5])
            else:
                self.anim.show_animation()
        return obs, rew, ter, tur, info


from stable_baselines3 import PPO
from helpers import shortest_queue10
number_of_trials = 100
train_env = VFCOffloadingEnv10(60)
#model = PPO("MlpPolicy", train_env, verbose=1, gamma=0.85).learn(300000)
#model.save("env10-300000")
model = PPO("MlpPolicy", train_env, verbose=1).load("env10-300000")


def test_offloading_method(n, method_name):
    total_rew = 0
    total_delay = 0
    total_num_tasks = 0
    test_delays = []
    for i in range(n):
        env = VFCOffloadingEnv10(60, render_mode=None)
        obs,_ = env.reset()
        ter = False
        tot_rew = 0
        #action_store = []
        while not ter:
                if method_name == "RL":
                    action = model.predict(obs, deterministic=True)[0]
                    #action_store.append(action)
                elif method_name == "Greedy":
                    action = shortest_queue10(obs)
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
        #print(action_store)
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
    with open("out-10-"+method_name+".txt", 'w') as f:
        for line in test_delays:
            f.write(f"{line}\n")

test_offloading_method(100, "RL")
test_offloading_method(100, "Random")
test_offloading_method(100, "Cloud")
test_offloading_method(100, "RSU")
test_offloading_method(100, "Greedy")
