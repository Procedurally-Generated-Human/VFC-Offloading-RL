import matplotlib.pyplot as plt
from celluloid import Camera
import numpy as np
from matplotlib.patches import Rectangle
import math

AREA_SIZE = 1000
IMAGE_SIZE = 100
CLOUD_SIZE = 40
FRAME_INTERVAL = 1000


class Animator(object):
    def __init__(self, walks, parked, parked_hardware):
        self.walks = np.array(walks)
        self.parked = np.array(parked)
        self.parked_hardware = parked_hardware
        self.fig = plt.figure("Vehicular Fog Computing")
        self.camera = Camera(self.fig)
        plt.xlim(0, AREA_SIZE)
        plt.ylim(0, AREA_SIZE)
        self.rsu_image = plt.imread('rsu-black-transparent.png')
    def add_frame(self, t, emitted_node, action):
        emitted_node -= 1
        plt.scatter(self.walks[:,math.trunc(t),0],self.walks[:,math.trunc(t),1],c="blue")
        plt.scatter(self.parked[:,0],self.parked[:,1],c="red")
        for i,txt in enumerate(self.parked):
            plt.annotate(text=str(self.parked_hardware[i]),xy=self.parked[i])
        plt.plot([self.walks[emitted_node][math.trunc(t)][0],500], [self.walks[emitted_node][math.trunc(t)][1],500], marker='', color='cornflowerblue')
        plt.xlim(-10,1010)
        plt.ylim(-10,1010)
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(200))
        plt.gca().yaxis.set_major_locator(plt.MultipleLocator(200))
        for i in range(5):
            for j in range(5):
                plt.gca().add_patch(Rectangle((200*i+10, 200*j+10), 170, 170, facecolor="silver", zorder=-100.0))
        plt.imshow(self.rsu_image, extent=[(500-IMAGE_SIZE/2), (500+IMAGE_SIZE/2), 500-IMAGE_SIZE/2, 500+IMAGE_SIZE/2])
        plt.imshow(plt.imread('cloud-black.png'), extent=[(550-CLOUD_SIZE/2), (550+CLOUD_SIZE/2), 550-CLOUD_SIZE/2, 550+CLOUD_SIZE/2])
        if action == 0:
            plt.imshow(plt.imread('rsu-red-transparent.png'), extent=[(500-IMAGE_SIZE/2), (500+IMAGE_SIZE/2), 500-IMAGE_SIZE/2, 500+IMAGE_SIZE/2])
        elif action == 1:
            plt.imshow(plt.imread('cloud-red.png'), extent=[(550-CLOUD_SIZE/2), (550+CLOUD_SIZE/2), 550-CLOUD_SIZE/2, 550+CLOUD_SIZE/2])
            plt.plot([550,500], [550,500], marker='', color='pink')
        elif action == 2:
            plt.plot([self.parked[0][0],500], [self.parked[0][1],500], marker='', color='pink')
        else:
            plt.plot([self.parked[1][0],500], [self.parked[1][1],500], marker='', color='pink')
        props = dict(boxstyle='round', facecolor='grey', alpha=0.15)
        details = "-- Simulation Details --"+"\nTime: "+str(round(t,5))+"\nOrigin: "+str(emitted_node)+"\nAction: "+str(action)
        plt.text(1050, 1000, details, fontsize=12, verticalalignment='top', bbox=props)
        self.camera.snap()

    def show_animation(self):
        animation = self.camera.animate(interval = FRAME_INTERVAL)
        plt.show()