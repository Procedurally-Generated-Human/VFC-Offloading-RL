import numpy as np
import random
#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
#from celluloid import Camera
#from matplotlib.patches import Rectangle


def create_parked_coords():
    blocks = [[20,20,170,170],[220,20,370,170],[420,20,570,170],[620,20,770,170],[820,20,970,170],
              [20,220,170,370],[220,220,370,370],[420,220,570,370],[620,220,770,370],[820,220,970,370],
              [20,420,170,570],[220,420,370,570],[420,420,570,570],[620,420,770,570],[820,420,970,570],
              [20,620,170,770],[220,620,370,770],[420,620,570,770],[620,620,770,770],[820,620,970,770],
              [20,820,170,970],[220,820,370,970],[420,820,570,970],[620,820,770,970],[820,820,970,970],
              ]

    block = blocks[np.random.choice(len(blocks))]
    x_min = block[0]
    y_min = block[1]
    x_max = block[2]
    y_max = block[3]
    width = abs(x_max - x_min)
    height = abs(y_max - y_min)
    
    # Calculate the total length of the rectangle's perimeter
    perimeter = 2 * (width + height)
    
    # Generate a random number between 0 and the perimeter
    rand_num = random.uniform(0, perimeter)
    
    # Determine which side of the rectangle the point is on
    if rand_num < width:
        return (x_min + rand_num, y_min)
    elif rand_num < width + height:
        return (x_max, y_min + rand_num - width)
    elif rand_num < 2 * width + height:
        return (x_max - rand_num + width + height, y_max)
    else:
        return (x_min, y_max - rand_num + 2 * width + height)





def create_random_walk(timesteps):
    timesteps += 100
    x = random.choice([200,400,600,800])
    y = random.choice([200,400,600,800])
    speed = random.choice([10,20,25,40])
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
            if x==1000:
                dirs.remove("right")
            if y==0:
                dirs.remove("down")
            if y==1000:
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


