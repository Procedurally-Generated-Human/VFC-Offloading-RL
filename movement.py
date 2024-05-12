import numpy as np
import random
#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
#from celluloid import Camera
#from matplotlib.patches import Rectangle

def create_random_walk(timesteps):
    timesteps += 100
    x = random.choice([200,400,600,800])
    y = random.choice([200,400,600,800])
    speed = random.choice([8,10])
    dirs = ["up","right","down","left"]
    dir = random.choice(dirs)
    coords = np.zeros((timesteps,2))
    coords[0][0] = x
    coords[0][1] = y
    for t in range(1,timesteps):
        dirs = ["up","right","down","left"]
        if x%200==0 and y%200==0:
            if x==0: 
                dirs.remove("left")
            elif x==1000:
                dirs.remove("right")
            elif y==0:
                dirs.remove("down")
            elif y==1000:
                dirs.remove("up")
            dir = random.choice(dirs)
        if dir=="up":
            y = y+speed
        elif dir=="right":
            x = x+speed
        elif dir=="down":
            y = y-speed
        elif dir=="left":
            x = x-speed
        else:
            print("something went wrong")
        coords[t][0] = x
        coords[t][1] = y
    return coords
