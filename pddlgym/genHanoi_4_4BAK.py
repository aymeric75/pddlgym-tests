import pddlgym
import imageio
#from pddlgym_planners.fd import FD
import numpy as np
import matplotlib.pyplot as plt

def rescale(images):
    return images/255.


def to_black_and_white(images):
    # Initialize an array of zeros with the same shape as the input, but without the color channels
    bw_images = np.zeros(images.shape[:-1], dtype="uint8")

    # Iterate through each image
    for i, img in enumerate(images):
        # Check if all color channels are 255 (white), across all pixels
        # The result is a 2D boolean array where True represents a white pixel
        is_white = np.all(img == 255, axis=-1)

        # Set pixels in the black and white image to 255 (white) where is_white is True, else they remain 0 (black)
        bw_images[i][is_white] = 255

    return bw_images


# Convert to grayscale
def rgb_to_grayscale(rgb_images):
    print("IN rgb_to_grayscale")
    rgb_images = np.array(rgb_images)
    grayscale = np.dot(rgb_images[...,:3], [0.21, 0.72, 0.07]).astype("uint8") 
    # Stack the grayscale values to create an (H, W, 3) image
    #return np.stack((grayscale,)*3, axis=-1)
    return grayscale


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



nb_samplings_per_starting_state = 20001 # has to be ODD 

def export_dataset():

    ## 1) Loading the env
    env = pddlgym.make("PDDLEnvHanoi-v0", dynamic_action_space=True)
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
        img = img[::9, ::9]

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

            # plt.imsave("hanoi_ori_"+str(jjj)+".png", img)
            # plt.close()

            # rescaling
            # reducing resolution by 4
            img = img[::9, ::9]

            all_images.append(img)
            all_obs.append(obs.literals)


    all_images = np.array(all_images)
    
    print(all_images[0])

    # ##### Preprocess the images
    
    ## greys out the images
    all_images = rgb_to_grayscale(all_images)

    #all_images = to_black_and_white(all_images)
    print()
    print(all_images[0].shape)


    plt.imsave("hanoi_NOT.png", all_images[0])
    plt.close()



    ## normalize color
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
    print(all_actions_for_each_pair[2])
    print()
    plt.imsave("hanoi_pair0_0.png", all_pairs_of_images[2][0], cmap='gray')
    plt.close()


    plt.imsave("hanoi_pair0_1.png", all_pairs_of_images[2][1], cmap='gray')
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



