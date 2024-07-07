import matplotlib.pyplot as plt
from celluloid import Camera
import numpy as np
from matplotlib.patches import Rectangle
import math
np.set_printoptions(suppress=True)


AREA_SIZE = 1000
RSU_IMAGE_SIZE = 100
RSU_IMAGE_POSITION = [500,500]
CLOUD_IMAGE_SIZE = 40
CLOUD_IMAGE_POSITION = [550,550]
FRAME_INTERVAL = 1000


class Animator(object):
    def __init__(self, walks, parked, parked_hardware, rsu_hardware, cloud_hardware):
        self.walks = np.array(walks)
        self.parked = np.array(parked)
        self.parked_hardware = parked_hardware
        self.rsu_hardware = rsu_hardware
        self.cloud_hardware = cloud_hardware
        self.fig = plt.figure("Vehicular Fog Computing Simulation")
        self.camera = Camera(self.fig)
        self.rsu_image_black = plt.imread('rsu-black-transparent.png')
        self.rsu_image_red = plt.imread('rsu-red-transparent.png')
        self.cloud_image_black = plt.imread('cloud-black.png')
        self.cloud_image_red = plt.imread('cloud-red.png')
        plt.xlim(-10,AREA_SIZE+10)
        plt.ylim(-10,AREA_SIZE+10)
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(200))
        plt.gca().yaxis.set_major_locator(plt.MultipleLocator(200))

    def add_frame(self, t, emitted_node, action, obs):
        emitted_node -= 1
        # plot vehicles
        plt.scatter(self.walks[:,math.trunc(t),0],self.walks[:,math.trunc(t),1],c="blue")
        plt.scatter(self.parked[:,0],self.parked[:,1],c="red")
        # plot serivce vehicle hardware data
        for i,txt in enumerate(self.parked):
            plt.annotate(text=str(self.parked_hardware[i]),xy=self.parked[i])
        # plot line from emmited node to RSU
        plt.plot([self.walks[emitted_node][math.trunc(t)][0],RSU_IMAGE_POSITION[0]], [self.walks[emitted_node][math.trunc(t)][1],RSU_IMAGE_POSITION[1]], marker='', color='cornflowerblue')
        # plot city blocks
        for i in range(5):
            for j in range(5):
                plt.gca().add_patch(Rectangle((200*i+10, 200*j+10), 170, 170, facecolor="silver", zorder=-100.0))
        # plot RSU and cloud (black)
        plt.imshow(self.rsu_image_black, extent=[(RSU_IMAGE_POSITION[0]-RSU_IMAGE_SIZE/2), (RSU_IMAGE_POSITION[0]+RSU_IMAGE_SIZE/2), RSU_IMAGE_POSITION[1]-RSU_IMAGE_SIZE/2, RSU_IMAGE_POSITION[1]+RSU_IMAGE_SIZE/2])
        plt.imshow(self.cloud_image_black, extent=[(CLOUD_IMAGE_POSITION[0]-CLOUD_IMAGE_SIZE/2), (CLOUD_IMAGE_POSITION[0]+CLOUD_IMAGE_SIZE/2), CLOUD_IMAGE_POSITION[1]-CLOUD_IMAGE_SIZE/2, CLOUD_IMAGE_POSITION[1]+CLOUD_IMAGE_SIZE/2])
        # plot action taken: 0:red rsu - 1:red cloud - 2,3: draw line to service car
        if action == 0:
            plt.imshow(self.rsu_image_red, extent=[(RSU_IMAGE_POSITION[0]-RSU_IMAGE_SIZE/2), (RSU_IMAGE_POSITION[0]+RSU_IMAGE_SIZE/2), RSU_IMAGE_POSITION[1]-RSU_IMAGE_SIZE/2, RSU_IMAGE_POSITION[1]+RSU_IMAGE_SIZE/2])
        elif action == 1:
            plt.imshow(self.cloud_image_red, extent=[(CLOUD_IMAGE_POSITION[0]-CLOUD_IMAGE_SIZE/2), (CLOUD_IMAGE_POSITION[0]+CLOUD_IMAGE_SIZE/2), CLOUD_IMAGE_POSITION[1]-CLOUD_IMAGE_SIZE/2, CLOUD_IMAGE_POSITION[1]+CLOUD_IMAGE_SIZE/2])
            plt.plot([CLOUD_IMAGE_POSITION[0],RSU_IMAGE_POSITION[0]], [CLOUD_IMAGE_POSITION[1],RSU_IMAGE_POSITION[1]], marker='', color='pink')
        elif action == 2:
            plt.plot([self.parked[0][0],RSU_IMAGE_POSITION[0]], [self.parked[0][1],RSU_IMAGE_POSITION[1]], marker='', color='pink')
        elif action == 3:
            plt.plot([self.parked[1][0],RSU_IMAGE_POSITION[0]], [self.parked[1][1],RSU_IMAGE_POSITION[1]], marker='', color='pink')
        elif action == 4:
            plt.plot([self.parked[2][0],RSU_IMAGE_POSITION[0]], [self.parked[2][1],RSU_IMAGE_POSITION[1]], marker='', color='pink')
        elif action == 5:
            plt.plot([self.parked[3][0],RSU_IMAGE_POSITION[0]], [self.parked[3][1],RSU_IMAGE_POSITION[1]], marker='', color='pink')
        # place simulation details text
        props = dict(boxstyle='round', facecolor='grey', alpha=0.15)
        details = "-- Simulation Details --"+"\nTime: "+str(round(t,5))+"\nOrigin: "+str(emitted_node)+"\nAction: "+str(action)+"\nRSU Cpu: "+str(self.rsu_hardware[0])+"\nCloud BW: "+str(self.cloud_hardware[0])+"\nCloud CPU: "+str(self.cloud_hardware[1])+"\nTrans Queues: "+str(obs[0])+"\nCPU Queues: "+str(obs[1])+"\nTask CU: "+str(obs[2])+"\nTask SZ: "+str(obs[3])
        plt.text(1050, 1000, details, fontsize=12, verticalalignment='top', bbox=props)
        self.camera.snap()

    def show_animation(self):
        animation = self.camera.animate(interval = FRAME_INTERVAL)
        plt.show()