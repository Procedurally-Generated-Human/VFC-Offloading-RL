import matplotlib.pyplot as plt
from celluloid import Camera
import numpy as np
from matplotlib.patches import Rectangle
import math

AREA_SIZE = 1000
IMAGE_SIZE = 100
FRAME_INTERVAL = 1


class Animator(object):
    def __init__(self, walks):
        self.walks = np.array(walks)
        self.fig = plt.figure("Vehicular Fog Computing")
        self.camera = Camera(self.fig)
        plt.xlim(0, AREA_SIZE)
        plt.ylim(0, AREA_SIZE)
        self.rsu_image = plt.imread('rsu-red-transparent.png')
    def add_frame(self, t):
        plt.scatter(self.walks[:,math.trunc(t),0],self.walks[:,math.trunc(t),1],c="blue")
        plt.xlim(-10,1010)
        plt.ylim(-10,1010)
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(200))
        plt.gca().yaxis.set_major_locator(plt.MultipleLocator(200))
        for i in range(5):
            for j in range(5):
                plt.gca().add_patch(Rectangle((200*i+10, 200*j+10), 170, 170, facecolor="grey", zorder=-100.0))
        plt.imshow(self.rsu_image, extent=[(500-IMAGE_SIZE/2), (500+IMAGE_SIZE/2), 500-IMAGE_SIZE/2, 500+IMAGE_SIZE/2])
        self.camera.snap()

    def show_animation(self):
        animation = self.camera.animate(interval = FRAME_INTERVAL)
        plt.show()