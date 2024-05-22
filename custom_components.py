import ciw
import numpy as np
import math
import random



class CustomSimulation(ciw.Simulation):
    def __init__(self, network):
        super().__init__(network=network, arrival_node_class=CustomArrival, tracker=ciw.trackers.NodePopulation(), individual_class=CustomIndividual,  node_class=[ciw.Node, ciw.Node, ciw.Node, ciw.Node, ciw.Node, CustomNode,
                                ciw.Node, ciw.Node, ciw.Node, ciw.Node, ciw.Node, ciw.Node,
                                ciw.Node])
    def simulate_until_decision(self, max_simulation_time):
        next_active_node = self.find_next_active_node()
        self.current_time = next_active_node.next_event_date
        while self.current_time < max_simulation_time:
            next_active_node = self.event_and_return_nextnode(next_active_node)
            self.statetracker.timestamp()
            self.current_time = next_active_node.next_event_date
            if next_active_node == self.nodes[6]:
                break
        return self.current_time


class CustomNode(ciw.Node):
    # 7-8-10-12
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decision = 0
        self.number_of_serviced_individuals = 0
        self.stop = 0
    def next_node(self, ind):
        self.number_of_serviced_individuals += 1
        self.stop = 1
        if self.decision == 0: #rsu
            return self.simulation.nodes[7]
        elif self.decision == 1: #cloud
            return self.simulation.nodes[8]
        elif self.decision == 2: #service1
            return self.simulation.nodes[10]
        elif self.decision == 3: #service2
            return self.simulation.nodes[12]


class CustomIndividual(object):
    def __init__(self, id_number, cu, sz, dl, customer_class='Customer', priority_class=0, simulation=False):
        """
        Initialise an individual.
        """
        self.arrival_date = False
        self.service_start_date = False
        self.service_time = False
        self.service_end_date = False
        self.exit_date = False
        self.id_number = id_number
        self.data_records = []
        self.customer_class = customer_class
        self.previous_class = customer_class
        self.priority_class = priority_class
        self.prev_priority_class = priority_class
        self.original_class = customer_class
        self.is_blocked = False
        self.server = False
        self.queue_size_at_arrival = False
        self.queue_size_at_departure = False
        self.destination = False
        self.interrupted = False
        self.node = False
        self.simulation = simulation
        self.cu = cu
        self.sz = sz
        self.dl = dl

    def __repr__(self):
        """Represents an Individual instance as a string.
        """
        return f"Individual {self.id_number}"


class CustomArrival(ciw.ArrivalNode):
    def have_event(self):
        """
        Finds a batch size. Creates that many Individuals and send
        them to the relevent node. Then updates the event_dates_dict.
        """
        batch = self.batch_size(self.next_node, self.next_class)
        for _ in range(batch):
            self.number_of_individuals += 1
            self.number_of_individuals_per_class[self.next_class] += 1
            priority_class = self.simulation.network.priority_class_mapping[self.next_class]
            next_individual = self.simulation.IndividualType(
                self.number_of_individuals,
                random.randint(800,1200), #cu
                random.randint(80,120), #sz
                random.random(), #dl
                self.next_class,
                priority_class,
                simulation=self.simulation,
            )

            next_node = self.simulation.transitive_nodes[self.next_node - 1]
            self.release_individual(next_node, next_individual)

        self.event_dates_dict[self.next_node][self.next_class] = self.increment_time(
            self.event_dates_dict[self.next_node][self.next_class],
            self.inter_arrival(self.next_node, self.next_class),
        )
        self.find_next_event_date()


class ComputationDist(ciw.dists.Distribution):
    def __init__(self, mips):
        self.mips = mips
    def sample(self, t=None, ind=None):
        return ind.cu/self.mips
    

class StationaryTransmissionDist(ciw.dists.Distribution):
    def __init__(self, bw, x, y):
        self.bw = bw
        self.x = x
        self.y = y
    def sample(self, t=None, ind=None):
        return ind.sz/self.bw


class StationaryTransmissionDistNew(ciw.dists.Distribution):
    def __init__(self, bw, x, y):
        self.bw = bw
        self.x = x
        self.y = y
    def sample(self, t=None, ind=None):
        return ind.sz/self.bw
    

class MovingTransmissionDist(ciw.dists.Distribution):
    def __init__(self, bw, coords):
        self.bw = bw
        self.coords = coords
    def sample(self, t=None, ind=None):
        coefficient =  np.linalg.norm(self.coords[math.trunc(t)] - np.array([500,500])) /700
        return (ind.sz/self.bw)*coefficient
    
class MovingTransmissionDistNew(ciw.dists.Distribution):
    def __init__(self, bw, coords):
        self.bw = bw
        self.coords = coords
    def sample(self, t=None, ind=None):
        distance = np.linalg.norm(self.coords[math.trunc(t)] - np.array([500,500]))
        r_div_t = 2*2.15 - 32.44 - 20*math.log10(distance*0.001) - 20*math.log10(5900)
        r = 10*(10**6)*math.log2(1+((0.2*(10**(r_div_t/10)))/(2*(10**6)*(10**(-17.4)))))
        return (ind.sz/(r/1_000_000))

import math

def fspl():
    snr = (40)
    return 10_000_000*math.log2(1+snr)

def bwi(x):
    SNR = (30)/(-100)
    trans_rate = 1_000_000*math.log2(1+SNR)
    return trans_rate

def pls(x):
    oij = x #up
    lij = 32.4+(20*math.log10(59_000_000))+(20*math.log10(oij)) #up
    hij = 20+5+5-lij #down
    snr = (30*hij)/(114) #up
    return snr
    trans_rate = 20_000_000*math.log2(1+snr)
    return trans_rate/8_000_000


def chinese_model(x):
    path_loss = 10*math.log10(((6_000_000_000)/(4*3.14*x))**2)
    return path_loss
    yi = (30*path_loss)/144
    print(yi)
    bw = 10_000_000*math.log2(1+yi)
    return bw/8_000_000



def PL_free(fc, dist, Gt=None, Gr=None):
    # Free Space Path loss Model
    # Inputs:
    #   fc        : carrier frequency [Hz]
    #   dist      : distance between base station and mobile station [m]
    #   Gt        : transmitter gain
    #   Gr        : receiver gain
    # Output:
    #   PL        : path loss [dB]

    lambda_ = 299792458 / fc
    tmp = lambda_ / (4 * math.pi * dist)

    if Gt is not None:
        tmp *= math.sqrt(Gt)
    if Gr is not None:
        tmp *= math.sqrt(Gr)

    PL = -20 * math.log10(tmp)

    return PL


def book_formula(fc, dist,):
    loss = 20*math.log10(fc) + 20*math.log10(dist) - 147.56
    received_power = 30-loss
    #snr = 10*math.log10((received_power)/(-144))
    snr = received_power - (-144)
    snr = (10**(snr/10))/1000
    trans_rate = 10_000_000+math.log2(1+snr)
    print(trans_rate/1_000_000)
    return trans_rate/1_000_000

def stack_overflow(x):
    r_div_t = 2*2.15 - 32.44 - 20*math.log10(x*0.001) - 20*math.log10(2400)
    r = 2*(10**6)*math.log2(1+((0.2*(10**(r_div_t/10)))/(2*(10**6)*(10**(-20.4)))))
    print(r)
    return r


def final(x):
    r_div_t = 2*2.15 - 32.44 - 20*math.log10(x*0.001) - 20*math.log10(5900)
    r = 10*(10**6)*math.log2(1+((0.2*(10**(r_div_t/10)))/(2*(10**6)*(10**(-17.4)))))
    print(r)
    return r/1_000_000

import matplotlib.pyplot as plt

a = []
for i in range(1,1000):
    a.append(final(i))

plt.scatter([i for i in range(1,1000)],a)
plt.show()