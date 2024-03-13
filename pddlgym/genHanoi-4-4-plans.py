import pddlgym
from pddlgym_planners.fd import FD
import matplotlib.pyplot as plt
import numpy as np
import random
import itertools
from skimage import exposure, color



def all_colors():
    #colors = [0, 64, 128, 192, 255]
    colors = [0, 128, 255]
    combinations = list(itertools.product(colors, repeat=3))
    return combinations

# Define a function to convert an RGB combination to LAB
def convert_to_lab(rgb):
    # skimage expects RGB values to be normalized between 0 and 1
    normalized_rgb = [value / 255.0 for value in rgb]
    # Reshape the RGB tuple to a 1x1x3 numpy array as required by rgb2lab
    rgb_array = np.array([normalized_rgb]).reshape((1, 1, 3))
    lab_array = color.rgb2lab(rgb_array)
    # Return the LAB value, flattened back into a simple tuple
    return tuple(lab_array.flatten())


lab_combinations = list(map(convert_to_lab, all_colors()))

counntteerr = 0


def convert_to_lab_and_to_color_wt_min_distance(rgb, boolean_matrix):

    global counntteerr

    if counntteerr % 100000 == 0:
        print("counter is {}".format(str(counntteerr)))

    if counntteerr == len(boolean_matrix):
        return rgb

    #if counntteerr < len(boolean_matrix):
    if boolean_matrix[counntteerr]:
        lab_array = color.rgb2lab(rgb)
        all_dists = all_distances(lab_array)
        # put the "0" dist to infinit (coz we dont want to spot this one)
        if 0 in all_dists:
            all_dists[all_dists.index(0)] = 99999
        # retrieve the index of the min distance (expect when is equal)
        index_min_color = np.argmin(all_dists)
        closest_color = all_colors()[index_min_color]
        counntteerr+=1
        return closest_color
    
    else:
        counntteerr+=1
        return rgb



def all_distances(labcolor):

    all_distances_lab = list(map(lambda x: color.deltaE_cie76(labcolor, x), lab_combinations))
    #all_distances_lab = list(map(lambda x: color.deltaE_ciede2000(labcolor, x), lab_combinations))
    #all_distances_lab = list(map(lambda x: color.deltaE_ciede94(labcolor, x), lab_combinations))

    return all_distances_lab




def add_noise(images, seed=0):

    np.random.seed(seed)


    if not isinstance(images, np.ndarray):
        images = np.array(images)

    # print(images.shape) # (768, 25, 70, 3)
    # # boolean_matrix 


    # convert the image dataset into a "lab" format 

    reshaped_rgb = images.reshape(-1, 3)  # Reshape to a 2D array where each row is a pixel's RGB values
    print(reshaped_rgb.shape) # 5 082 000


    # Generate a matrix of random numbers from a uniform distribution
    random_matrix = np.random.uniform(low=0.0, high=1.0, size=(reshaped_rgb.shape[0],))
    # Create a boolean matrix: True if the element is < 0.05, False otherwise
    boolean_matrix = random_matrix < 0.05
    #print(boolean_matrix)

    lab_pixels = np.apply_along_axis(convert_to_lab_and_to_color_wt_min_distance, 1, reshaped_rgb, boolean_matrix)
    


    #print("seed {}, images.shape {}, reshaped_rgb {}, lab_pixels {}, bool matrix {}".format(str(seed), str(images.shape), str(reshaped_rgb.shape), str(lab_pixels.shape), str(boolean_matrix.shape)))


    #dataset_lab = lab_pixels.reshape(768, 25, 70, 3)
    dataset_lab = lab_pixels.reshape(images.shape)

    # NOW holds the closests values  

    return dataset_lab



# Function to reduce resolution of a single image using np.take
def reduce_resolution(image):
    # Use np.take to select every 4th element in the first two dimensions
    reduced_image = np.take(np.take(image, np.arange(0, image.shape[0], 9), axis=0),
                            np.arange(0, image.shape[1], 9), axis=1)
    return reduced_image


# See `pddl/sokoban.pddl` and `pddl/sokoban/problem3.pddl`.
env = pddlgym.make("PDDLEnvHanoi44-v0", dynamic_action_space=True)

print(env)

obs, debug_info = env.reset(_problem_idx=0)
img, peg_to_disc_list = env.render()
img = img[:,:,:3]
print(peg_to_disc_list)

counntteerr = 0
#img = add_noise(img, seed=1)


plt.imsave("hanoi_lol_0.png", reduce_resolution(img))
plt.close()


planner = FD()
plan = planner(env.domain, obs)

for i, act in enumerate(plan):

    print("Obs:", obs)
    print("Act:", act)
    obs, reward, done, debug_info = env.step(act)

    img, peg_to_disc_list = env.render()
    img = img[:,:,:3]

    counntteerr = 0
    #img = add_noise(img, seed=1)
    

    plt.imsave("hanoi_lol_"+str(i+1)+".png", reduce_resolution(img))
    plt.close()

print("Final obs, reward, done:", obs, reward, done)
