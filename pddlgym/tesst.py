import pddlgym
import imageio
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_images

import numpy as np




def normalize(x):
    print(np.array(x).shape)
    mean               = np.mean(x, axis=0)
    std                = np.std(x, axis=0)
    #print("normalized shape:",mean.shape,std.shape)
    print(mean.shape)
    return (x - mean)/(std+1e-20), mean, std

# Convert to grayscale
def rgb_to_grayscale(rgb_images):
    rgb_images = np.array(rgb_images)
    grayscale = np.dot(rgb_images[...,:3], [0.21, 0.72, 0.07])
    # Stack the grayscale values to create an (H, W, 3) image
    return np.stack((grayscale,)*3, axis=-1)


nb_samplings_per_starting_state = 50


def export_dataset():




    ## Loading the env
    env = pddlgym.make("PDDLEnvHanoi-v0", dynamic_action_space=True)
    obs, debug_info = env.reset(_problem_idx=1)

    all_images = []
    unique_transitions = []
    all_obs = []
    all_actions = []

    # looping over the number of Towers (for ToH)
    for ii in range(4):

        # Initializing the first State
        obs, debug_info = env.reset(_problem_idx=ii)

        #Retrieve the 1st image
        img = env.render()
        img = img[:,:,:3]

        # # adding the img and obs to all_*
        # all_images.append(img)
        # all_obs.append(obs.literals)

        # # looping over the nber of states to sample for each starting tower
        # for jjj in range(nb_samplings_per_starting_state):

        #     # sample an action
        #     action = env.action_space.sample(obs)

        #     # add the actions to all_*
        #     all_actions.append(action)

        #     # apply the action and retrieve img and obs
        #     obs, reward, done, debug_info = env.step(action)
        #     img = env.render()
        #     img = img[:,:,:3]
        #     all_images.append(img)
        #     all_obs.append(obs.literals)

    # ######### Greying out the images 
    # all_images = rgb_to_grayscale(all_images)

    # ######### Normalizing the images (centered on 0, and with a std)
    # norm_images, mean_, std_ = normalize(all_images)

    # all_pairs_of_images = []
    # all_pairs_of_obs = []

    # # building the array of pairs
    # for iiii, p in enumerate(all_images):
    #     if iiii%2 == 0:
    #         all_pairs_of_images.append([all_images[iiii], all_images[iiii+1]])
    #         all_pairs_of_obs.append([all_obs[iiii], all_obs[iiii+1]])

    # # building the array of UNIQUE transitions
    # for ooo, obss in enumerate(all_pairs_of_obs):
    #     if str(obss) not in unique_transitions:
    #         unique_transitions.append(str(obss))
        
























    datasetss = load_sample_images()     
    print("putain de toiii")
    plt.imshow(datasetss.images[0])
    plt.savefig("kkkkkk-00.png")
    plt.imshow(datasetss.images[1])
    # #plt.gcf()
    plt.savefig("kkkkkkj-01.png")


export_dataset()