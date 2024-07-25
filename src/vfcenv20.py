import ciw.dists
import gymnasium as gym
import ciw
import numpy as np 
from movement import create_random_walk, create_parked_coords
from custom_components import MovingTransmissionDistNew, ComputationDist, StationaryTransmissionDistNew, CustomSimulation20, CloudCompDelay, CloudTransDelay #CNG
from animate import Animator
from helpers import generate_service_cpu, generate_rsu_hardware, generate_cloud_hardware
np.set_printoptions(suppress=True)


class VFCOffloadingEnv20(gym.Env):

    def __init__(self, n_timesteps, render_mode=None) -> None:
        self.action_space = gym.spaces.Discrete(10) #CNG
        self.observation_space =  gym.spaces.Box(0,10,(39,)) #CNG
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
        walk_11 = create_random_walk(self.n_timesteps)
        walk_12 = create_random_walk(self.n_timesteps)
        walk_13 = create_random_walk(self.n_timesteps)
        walk_14 = create_random_walk(self.n_timesteps)
        walk_15 = create_random_walk(self.n_timesteps)
        walk_16 = create_random_walk(self.n_timesteps)
        walk_17 = create_random_walk(self.n_timesteps)
        walk_18 = create_random_walk(self.n_timesteps)
        walk_19 = create_random_walk(self.n_timesteps)
        walk_20 = create_random_walk(self.n_timesteps)
        self.parked_1 = create_parked_coords()
        self.parked_2 = create_parked_coords()
        self.parked_3 = create_parked_coords() #CNG
        self.parked_4 = create_parked_coords()
        self.parked_5 = create_parked_coords()
        self.parked_6 = create_parked_coords()
        self.parked_7 = create_parked_coords()
        self.parked_8 = create_parked_coords()
        self.service_vehicles_cpu = generate_service_cpu(8) #CNG
        self.rsu_hardware = generate_rsu_hardware()
        self.cloud_hardware = generate_cloud_hardware()
        if self.render_mode == "human":
            self.anim = Animator([walk_1,walk_2,walk_3,walk_4,walk_5,walk_6,walk_7,walk_8,walk_9,walk_10,walk_11,walk_12,walk_13,walk_14,walk_15,walk_16,walk_17,walk_18,walk_19,walk_20],[self.parked_1,self.parked_2,self.parked_3,self.parked_4,self.parked_5,self.parked_6,self.parked_7,self.parked_8], self.service_vehicles_cpu, self.rsu_hardware, self.cloud_hardware) #CNG
        self.N = ciw.create_network(
            arrival_distributions=[ciw.dists.Exponential(rate=3),   #client-trns-1       1
                           ciw.dists.Exponential(rate=3),           #client-trns-2       2
                           ciw.dists.Exponential(rate=3),           #client-trns-3       3
                           ciw.dists.Exponential(rate=3),           #client-trns-4       4
                           ciw.dists.Exponential(rate=3),           #client-trns-5       5
                           ciw.dists.Exponential(rate=3),           #client-trns-6       6 
                           ciw.dists.Exponential(rate=3),           #client-trns-7       7
                           ciw.dists.Exponential(rate=3),           #client-trns-8       8
                           ciw.dists.Exponential(rate=3),           #client-trns-9       9
                           ciw.dists.Exponential(rate=3),           #client-trns-10     10
                           ciw.dists.Exponential(rate=3),           #client-trns-11     11
                           ciw.dists.Exponential(rate=3),           #client-trns-12     12
                           ciw.dists.Exponential(rate=3),           #client-trns-13     13
                           ciw.dists.Exponential(rate=3),           #client-trns-14     14
                           ciw.dists.Exponential(rate=3),           #client-trns-15     15
                           ciw.dists.Exponential(rate=3),           #client-trns-16     16 
                           ciw.dists.Exponential(rate=3),           #client-trns-17     17
                           ciw.dists.Exponential(rate=3),           #client-trns-18     18
                           ciw.dists.Exponential(rate=3),           #client-trns-19     19
                           ciw.dists.Exponential(rate=3),           #client-trns-20     20
                           None,                                    #rsu-trns           21
                           None,                                    #rsu-cpu            22
                           None,                                    #trns-to-cloud      23
                           None,                                    #cloud-cpu          24
                           None,                                    #trns-to-service-1  25
                           None,                                    #service-cpu-1      26
                           None,                                    #trns-to-service-2  27
                           None,                                    #service-cpu-2      28
                           None,                                    #trns-to-service-3  29 
                           None,                                    #service-cpu-3      30
                           None,                                    #trns-to-service-4  31
                           None,                                    #service-cpu-4      32
                           None,                                    #trns-to-service-5  33
                           None,                                    #service-cpu-5      34
                           None,                                    #trns-to-service-6  35
                           None,                                    #service-cpu-6      36
                           None,                                    #trns-to-service-7  37 
                           None,                                    #service-cpu-7      38
                           None,                                    #trns-to-service-8  39
                           None,                                    #service-cpu-8      40
                           ],                                       ######################
            service_distributions=[MovingTransmissionDistNew(coords=walk_1),#client-trns-1       1
                           MovingTransmissionDistNew(coords=walk_2),        #client-trns-2       2
                           MovingTransmissionDistNew(coords=walk_3),        #client-trns-3       3
                           MovingTransmissionDistNew(coords=walk_4),        #client-trns-4       4
                           MovingTransmissionDistNew(coords=walk_5),        #client-trns-5       5
                           MovingTransmissionDistNew(coords=walk_6),        #client-trns-6       6 
                           MovingTransmissionDistNew(coords=walk_7),        #client-trns-7       7
                           MovingTransmissionDistNew(coords=walk_8),        #client-trns-8       8
                           MovingTransmissionDistNew(coords=walk_9),        #client-trns-9       9
                           MovingTransmissionDistNew(coords=walk_10),       #client-trns-10     10
                           MovingTransmissionDistNew(coords=walk_11),       #client-trns-11     11
                           MovingTransmissionDistNew(coords=walk_12),       #client-trns-12     12
                           MovingTransmissionDistNew(coords=walk_13),       #client-trns-13     13
                           MovingTransmissionDistNew(coords=walk_14),       #client-trns-14     14
                           MovingTransmissionDistNew(coords=walk_15),       #client-trns-15     15
                           MovingTransmissionDistNew(coords=walk_16),       #client-trns-16     16
                           MovingTransmissionDistNew(coords=walk_17),       #client-trns-17     17
                           MovingTransmissionDistNew(coords=walk_18),       #client-trns-18     18
                           MovingTransmissionDistNew(coords=walk_19),       #client-trns-19     19
                           MovingTransmissionDistNew(coords=walk_20),       #client-trns-20     20
                           ciw.dists.Deterministic(value=0.0000001),        #rsu-trns           21
                           ComputationDist(mips=self.rsu_hardware[0]),      #rsu-cpu            22
                           CloudTransDelay(bw=1000),                        #trns-to-cloud      23
                           CloudCompDelay(self.cloud_hardware[1],0.05,0.2), #cloud-cpu          24
                           StationaryTransmissionDistNew(x=self.parked_1[0],y=self.parked_1[1]),      #trns-to-service-1    25
                           ComputationDist(mips=self.service_vehicles_cpu[0]),                        #service-cpu-1        26
                           StationaryTransmissionDistNew(x=self.parked_2[0],y=self.parked_2[1]),      #trns-to-service-2    27
                           ComputationDist(mips=self.service_vehicles_cpu[1]),                        #service-cpu-2        28
                           StationaryTransmissionDistNew(x=self.parked_3[0],y=self.parked_3[1]),      #trns-to-service-3    29 
                           ComputationDist(mips=self.service_vehicles_cpu[2]),                        #service-cpu-3        30
                           StationaryTransmissionDistNew(x=self.parked_4[0],y=self.parked_4[1]),      #trns-to-service-4    31
                           ComputationDist(mips=self.service_vehicles_cpu[3]),                        #service-cpu-4        32
                           StationaryTransmissionDistNew(x=self.parked_5[0],y=self.parked_5[1]),      #trns-to-service-5    33
                           ComputationDist(mips=self.service_vehicles_cpu[4]),                        #service-cpu-5        34
                           StationaryTransmissionDistNew(x=self.parked_6[0],y=self.parked_6[1]),      #trns-to-service-6    35
                           ComputationDist(mips=self.service_vehicles_cpu[5]),                        #service-cpu-6        36
                           StationaryTransmissionDistNew(x=self.parked_8[0],y=self.parked_7[1]),      #trns-to-service-7    37 
                           ComputationDist(mips=self.service_vehicles_cpu[6]),                        #service-cpu-7        38
                           StationaryTransmissionDistNew(x=self.parked_8[0],y=self.parked_8[1]),      #trns-to-service-8    39
                           ComputationDist(mips=self.service_vehicles_cpu[7]),                        #service-cpu-8        40
                           ],
            routing = [ #CNG
                               #1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #1
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #2
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #3
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #4
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #5
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #6
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #7
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #8
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #9
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #10
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #11
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #12
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #13
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #14
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #15
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #16
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #17
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #18
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #19
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #20
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #21
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #22
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #23
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #24
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #25
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #26
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0])), #27
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #28
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])), #29
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #30
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])), #31
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #32
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0])), #33
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #34
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])), #35
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #36
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0])), #37
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #38
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])), #39
                list(np.float_([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])), #40
               ],
            number_of_servers= [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,float("Inf"),1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] #CNG
            )
        self.Q = CustomSimulation20(self.N) #CNG
        self.Q.simulate_until_decision(self.n_timesteps)
        rsu_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[22].all_individuals)) #CNG
        cloud_trans_queue = sum(tuple(x.sz for x in self.Q.nodes[23].all_individuals)) #CNG
        cloud_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[24].all_individuals)) #CNG
        service_1_trans_queue = sum(tuple(x.sz for x in self.Q.nodes[25].all_individuals)) #CNG
        service_1_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[26].all_individuals)) #CNG
        service_2_trans_queue = sum(tuple(x.sz for x in self.Q.nodes[27].all_individuals)) #CNG
        service_2_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[28].all_individuals)) #CNG
        service_3_trans_queue = sum(tuple(x.sz for x in self.Q.nodes[29].all_individuals)) #CNG
        service_3_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[30].all_individuals)) #CNG
        service_4_trans_queue = sum(tuple(x.sz for x in self.Q.nodes[31].all_individuals)) #CNG
        service_4_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[32].all_individuals)) #CNG
        service_5_trans_queue = sum(tuple(x.sz for x in self.Q.nodes[33].all_individuals)) #CNG
        service_5_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[34].all_individuals)) #CNG
        service_6_trans_queue = sum(tuple(x.sz for x in self.Q.nodes[35].all_individuals)) #CNG
        service_6_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[36].all_individuals)) #CNG
        service_7_trans_queue = sum(tuple(x.sz for x in self.Q.nodes[37].all_individuals)) #CNG
        service_7_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[38].all_individuals)) #CNG
        service_8_trans_queue = sum(tuple(x.sz for x in self.Q.nodes[39].all_individuals)) #CNG
        service_8_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[40].all_individuals)) #CNG
        obs_service_distance = np.array([np.linalg.norm((self.parked_1[0],self.parked_1[1]) - np.array([500,500])), np.linalg.norm((self.parked_2[0],self.parked_2[1]) - np.array([500,500])), np.linalg.norm((self.parked_3[0],self.parked_3[1]) - np.array([500,500])), np.linalg.norm((self.parked_4[0],self.parked_4[1]) - np.array([500,500])), np.linalg.norm((self.parked_5[0],self.parked_5[1]) - np.array([500,500])),np.linalg.norm((self.parked_6[0],self.parked_6[1]) - np.array([500,500])), np.linalg.norm((self.parked_7[0],self.parked_7[1]) - np.array([500,500])), np.linalg.norm((self.parked_8[0],self.parked_8[1]) - np.array([500,500]))])/707 #CNG
        obs_queues_cpu = np.array([rsu_cpu_queue, cloud_cpu_queue, service_1_cpu_queue, service_2_cpu_queue, service_3_cpu_queue, service_4_cpu_queue, service_5_cpu_queue, service_6_cpu_queue, service_7_cpu_queue, service_8_cpu_queue], dtype=np.float32)/1000000 #CNG
        obs_queues_trans = np.array([cloud_trans_queue, service_1_trans_queue, service_2_trans_queue, service_3_trans_queue, service_4_trans_queue, service_5_trans_queue, service_6_trans_queue, service_7_trans_queue, service_8_trans_queue], dtype=np.float32)/100 #CNG
        obs_task_cu = (np.array([self.Q.nodes[21].all_individuals[0].cu])-27.5)/(13010-27.5) #CGD
        obs_task_sz = (np.array([self.Q.nodes[21].all_individuals[0].sz])-10)/(30-10) #CGD
        obs_cpu = (np.concatenate(([self.cloud_hardware[1]],[self.rsu_hardware[0]],self.service_vehicles_cpu))-18375)/(71120-18375) 
        obs = np.concatenate((obs_queues_trans, obs_queues_cpu, obs_service_distance, obs_task_cu, obs_task_sz, obs_cpu),dtype=np.float32)
        info = {}
        self.calculated_inds = []
        return obs, info
    
    def step(self, action):
        self.Q.nodes[21].decision = action #CGD
        time = self.Q.simulate_until_decision(self.n_timesteps)
        exitnode = self.Q.nodes[-1]
        rew = 0
        for ind in exitnode.all_individuals:
            if ind not in self.calculated_inds:
                ind_recs = ind.data_records
                arrival_date = ind_recs[0].arrival_date
                exit_date = ind_recs[-1].exit_date
                total_delay = exit_date - arrival_date
                rew += 5 - total_delay
                self.calculated_inds.append(ind)
        info = {}
        ter = False
        tur = False
        if self.Q.current_time >= self.n_timesteps:
            ter = True
        if not ter:
            rsu_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[22].all_individuals)) #CNG
            cloud_trans_queue = sum(tuple(x.sz for x in self.Q.nodes[23].all_individuals)) #CNG
            cloud_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[24].all_individuals)) #CNG
            service_1_trans_queue = sum(tuple(x.sz for x in self.Q.nodes[25].all_individuals)) #CNG
            service_1_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[26].all_individuals)) #CNG
            service_2_trans_queue = sum(tuple(x.sz for x in self.Q.nodes[27].all_individuals)) #CNG
            service_2_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[28].all_individuals)) #CNG
            service_3_trans_queue = sum(tuple(x.sz for x in self.Q.nodes[29].all_individuals)) #CNG
            service_3_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[30].all_individuals)) #CNG
            service_4_trans_queue = sum(tuple(x.sz for x in self.Q.nodes[31].all_individuals)) #CNG
            service_4_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[32].all_individuals)) #CNG
            service_5_trans_queue = sum(tuple(x.sz for x in self.Q.nodes[33].all_individuals)) #CNG
            service_5_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[34].all_individuals)) #CNG
            service_6_trans_queue = sum(tuple(x.sz for x in self.Q.nodes[35].all_individuals)) #CNG
            service_6_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[36].all_individuals)) #CNG
            service_7_trans_queue = sum(tuple(x.sz for x in self.Q.nodes[37].all_individuals)) #CNG
            service_7_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[38].all_individuals)) #CNG
            service_8_trans_queue = sum(tuple(x.sz for x in self.Q.nodes[39].all_individuals)) #CNG
            service_8_cpu_queue = sum(tuple(x.cu for x in self.Q.nodes[40].all_individuals)) #CNG
            obs_service_distance = np.array([np.linalg.norm((self.parked_1[0],self.parked_1[1]) - np.array([500,500])), np.linalg.norm((self.parked_2[0],self.parked_2[1]) - np.array([500,500])), np.linalg.norm((self.parked_3[0],self.parked_3[1]) - np.array([500,500])), np.linalg.norm((self.parked_4[0],self.parked_4[1]) - np.array([500,500])), np.linalg.norm((self.parked_5[0],self.parked_5[1]) - np.array([500,500])),np.linalg.norm((self.parked_6[0],self.parked_6[1]) - np.array([500,500])), np.linalg.norm((self.parked_7[0],self.parked_7[1]) - np.array([500,500])), np.linalg.norm((self.parked_8[0],self.parked_8[1]) - np.array([500,500]))])/707 #CNG
            obs_queues_cpu = np.array([rsu_cpu_queue, cloud_cpu_queue, service_1_cpu_queue, service_2_cpu_queue, service_3_cpu_queue, service_4_cpu_queue, service_5_cpu_queue, service_6_cpu_queue, service_7_cpu_queue, service_8_cpu_queue], dtype=np.float32)/1000000 #CNG
            obs_queues_trans = np.array([cloud_trans_queue, service_1_trans_queue, service_2_trans_queue, service_3_trans_queue, service_4_trans_queue, service_5_trans_queue, service_6_trans_queue, service_7_trans_queue, service_8_trans_queue], dtype=np.float32)/100 #CNG
            obs_task_cu = (np.array([self.Q.nodes[21].all_individuals[0].cu])-27.5)/(13010-27.5) #CGD
            obs_task_sz = (np.array([self.Q.nodes[21].all_individuals[0].sz])-10)/(30-10) #CGD
            obs_cpu = (np.concatenate(([self.cloud_hardware[1]],[self.rsu_hardware[0]],self.service_vehicles_cpu))-18375)/(71120-18375) 
            obs = np.concatenate((obs_queues_trans, obs_queues_cpu, obs_service_distance, obs_task_cu, obs_task_sz, obs_cpu),dtype=np.float32)
        else:
            obs = np.zeros((39))
        if self.render_mode == "human":
            if not ter:
                emitted_node = self.Q.nodes[21].all_individuals[0].data_records[0].node #CGD
                self.anim.add_frame(time, emitted_node, action, [obs_queues_trans*100, obs_queues_cpu*1000000, (obs_task_cu*(54600-2690))+2690, (obs_task_sz*(30-5))+5])
            else:
                self.anim.show_animation()
        return obs, rew, ter, tur, info


from stable_baselines3 import PPO
from helpers import shortest_queue20
train_env = VFCOffloadingEnv20(60)
#model = PPO("MlpPolicy", train_env, verbose=1, gamma=0.85).learn(300000)
#model.save("trained_models/env20-new")
model = PPO("MlpPolicy", train_env, verbose=1).load("trained_models/env20-300000")


def test_offloading_method(n, method_name):
    total_rew = 0
    total_delay = 0
    total_num_tasks = 0
    test_delays = []
    for i in range(n):
        print(i)
        env = VFCOffloadingEnv20(60, render_mode=None)
        obs,_ = env.reset()
        ter = False
        tot_rew = 0
        #zorder = 0
        while not ter:
                if method_name == "RL":
                    # color = "blue"
                    # zorder = 10
                    action = model.predict(obs)[0]
                    #action_store.append(action)
                elif method_name == "Greedy":
                    # color = "green"
                    action = shortest_queue20(obs)
                elif method_name == "RSU":
                    # color = "black"
                    action = 0
                elif method_name == "Cloud":
                    # color = "black"
                    action = 1
                elif method_name == "Random":
                    # color = "red"
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
    with open("results/out-20-"+method_name+".txt", 'w') as f:
        for line in test_delays:
            f.write(f"{line}\n")
    # ts = [ts[0] for ts in env.Q.statetracker.history]
    # cld_trns = [ts[1][22] for ts in env.Q.statetracker.history]
    # s1_trns = [ts[1][24] for ts in env.Q.statetracker.history]
    # s2_trns =[ts[1][26] for ts in env.Q.statetracker.history]
    # s3_trns =[ts[1][28] for ts in env.Q.statetracker.history]
    # s4_trns =[ts[1][30] for ts in env.Q.statetracker.history]
    # s5_trns = [ts[1][32] for ts in env.Q.statetracker.history]
    # s6_trns =[ts[1][34] for ts in env.Q.statetracker.history]
    # s7_trns =[ts[1][36] for ts in env.Q.statetracker.history]
    # s8_trns =[ts[1][38] for ts in env.Q.statetracker.history]
    # plt.plot(ts, np.array(cld_trns)+np.array(s1_trns)+np.array(s2_trns)+np.array(s3_trns)+np.array(s4_trns)+np.array(s5_trns)+np.array(s6_trns)+np.array(s7_trns)+np.array(s8_trns), label=method_name, linewidth=3, color=color, zorder=zorder)

test_offloading_method(10, "RL")
test_offloading_method(10, "Greedy")
test_offloading_method(10, "Random")
test_offloading_method(10, "Cloud")
test_offloading_method(10, "RSU")


# plt.legend(loc='best')
# plt.xlabel("Simulation Timestep (s)")
# plt.ylabel("Transmisson Queue Length")
# plt.legend(fontsize="large")
# plt.show()