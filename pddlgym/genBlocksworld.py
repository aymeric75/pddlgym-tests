import pddlgym
import imageio
#from pddlgym_planners.fd import FD
import numpy as np
import matplotlib.pyplot as plt

def rescale(images):
    return images/255.

#def reduce_resolution(images):


def normalize_colors(images):    
    mean      = np.mean(images, axis=0)
    std       = np.std(images, axis=0)
    return (images - mean)/(std+1e-6), mean, std



def unnormalize_colors(normalized_images, mean, std): 
    # Reverse the normalization process
    unnormalized_images = normalized_images * (std + 1e-6) + mean
    return np.round(unnormalized_images).astype(np.uint8)




# Preprocessing (as recommended by chatpgt, also the order is recommended by chatgpt)

# 0) rescalling pixels value from 0-255 to 0-1 (NO NEED, step 1 aleady rescale)
# 1) reducing the resolution of the images 
# 2) normalizing the colors on each color channel (by substracing the mean and dividing by std)



nb_samplings_per_starting_state = 501 # has to be ODD 

def export_dataset():

    ## 1) Loading the env
    env = pddlgym.make("PDDLEnvBlocks-v0", dynamic_action_space=True)
    obs, debug_info = env.reset(_problem_idx=1)

    all_images = []
    unique_transitions = []
    all_obs = []
    all_actions = []


    # looping over the number of starting positions
    # for blocksworld only the 1st pb has 4 blocks
    for ii in range(1):


        # Initializing the first State
        obs, debug_info = env.reset(_problem_idx=ii)


        # Retrieve the 1st image
        img = env.render()
        img = img[:,:,:3] # remove the transparancy

        print(img.shape)

        # rescaling
        # reducing resolution by 4
        img = img[::8, ::8]

        print(img.shape)
        


        # adding the img and obs to all_*
        all_images.append(img)
        all_obs.append(obs.literals)

        # looping over the nber of states to sample for each starting position
        for jjj in range(nb_samplings_per_starting_state):

            # sample an action
            action = env.action_space.sample(obs)

            # add the actions to all_*
            all_actions.append(action)

            # apply the action and retrieve img and obs
            obs, reward, done, debug_info = env.step(action)
            img = env.render()
            img = img[:,:,:3]

            # rescaling
            # reducing resolution by 4
            img = img[::8, ::8]

            # plt.imsave("blocks_"+str(jjj)+".png", img)
            # plt.close()

            all_images.append(img)
            all_obs.append(obs.literals)


    all_images = np.array(all_images)
    
    # ##### Preprocess the images
    all_images_norm, mean_, std_ = normalize_colors(all_images)
    #all_images_norm_denorm = unnormalize_colors(all_images_norm, mean_, std_)



    all_pairs_of_images = []
    all_pairs_of_images_norm = []
    all_pairs_of_obs = []

    # building the array of pairs
    for iiii, p in enumerate(all_images_norm):
        #if iiii%2 == 0:
        if iiii < len(all_images_norm)-1:
            all_pairs_of_images.append([all_images[iiii], all_images[iiii+1]])
            all_pairs_of_images_norm.append([all_images_norm[iiii], all_images_norm[iiii+1]])
            all_pairs_of_obs.append([all_obs[iiii], all_obs[iiii+1]])

    print(len(all_pairs_of_images_norm))
    print(len(all_actions))


    all_actions_for_each_pair = all_actions

    print(len(all_actions_for_each_pair))
    print(all_actions_for_each_pair[1])
    print()
    plt.imsave("blocks_pair0_0.png", all_pairs_of_images[1][0])
    plt.close()


    plt.imsave("blocks_pair0_1.png", all_pairs_of_images[1][1])
    plt.close()


    #build array containing all actions (no duplicate)
    all_actions_unique = []
    for uuu, act in enumerate(all_actions_for_each_pair):
        if str(act) not in all_actions_unique:
            all_actions_unique.append(str(act))

    #build array containing actions indices of the dataset
    all_actions_indices = []
    for ac in all_actions_for_each_pair:
        all_actions_indices.append(all_actions_unique.index(str(ac)))

    import torch
    import torch.nn.functional as F
    actions_indexes = torch.tensor(all_actions_indices)

    # array of one hot encoded actions
    actions_one_hot = F.one_hot(actions_indexes, num_classes=len(all_actions_unique))

    print(np.array(all_pairs_of_images).shape)
    print(np.array(all_pairs_of_images_norm).shape)
    print(np.array(actions_one_hot).shape)

    return all_pairs_of_images, all_pairs_of_images_norm, actions_one_hot.numpy(), mean_, std_, all_actions_unique



#export_dataset()



