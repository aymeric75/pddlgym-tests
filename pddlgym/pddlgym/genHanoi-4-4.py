import pddlgym
import imageio
#from pddlgym_planners.fd import FD
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os

def rescale(images):
    return images/255.

#def reduce_resolution(images):


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



def normalize_colors(images, mean=None, std=None):    
    if mean is None or std is None:
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


nb_samplings_per_starting_state = 3 # has to be ODD 


def generate_dataset():

    ## 1) Loading the env
    env = pddlgym.make("PDDLEnvHanoi-v0", dynamic_action_space=True)
    # obs, debug_info = env.reset(_problem_idx=0)

    all_traces = []


    # pour 4 images, 3 actions
    # pour N images, N-1 actions


    # looping over the number of starting positions
    # for blocksworld only the 1st pb has 4 blocks
    for ii in range(1):

        all_images_of_a_trace = []
        all_obs_of_a_trace = []
        all_actions_of_a_trace = []
        all_layouts_of_a_trace = []

        # Initializing the first State
        obs, debug_info = env.reset(_problem_idx=ii)


        # Retrieve the 1st image
        img, peg_to_disc_list = env.render()
        img = img[:,:,:3] # remove the transparancy

        print(img.shape)
        

        # rescaling
        # reducing resolution by 4
        #img = img[ , ::4]



        # adding the img and obs to all_*
        all_images_of_a_trace.append(img)
        all_layouts_of_a_trace.append(peg_to_disc_list)
        all_obs_of_a_trace.append(obs.literals)


        # looping over the nber of states to sample for each starting position
        for jjj in range(nb_samplings_per_starting_state):

            start_time = time.time()

            # sample an action
            action = env.action_space.sample(obs)



            # Capture the end time
            inter_time = time.time()
            # Calculate the duration by subtracting the start time from the end time
            duration = inter_time - start_time
            print(f"The inter time 0 is {duration} seconds.")

            # add the actions to all_*
            all_actions_of_a_trace.append(action)

            # apply the action and retrieve img and obs
            obs, reward, done, debug_info = env.step(action)


            # Capture the end time
            inter_time = time.time()
            # Calculate the duration by subtracting the start time from the end time
            duration = inter_time - start_time
            print(f"The inter time 1 is {duration} seconds.")


            img, peg_to_disc_list = env.render()
            img = img[:,:,:3]



            # Capture the end time
            inter_time = time.time()
            # Calculate the duration by subtracting the start time from the end time
            duration = inter_time - start_time
            print(f"The inter time 2 is {duration} seconds.")


            # plt.imsave("hanoi_"+str(jjj)+".png", img)
            # plt.close()

            all_images_of_a_trace.append(img)
            all_layouts_of_a_trace.append(peg_to_disc_list)
            all_obs_of_a_trace.append(obs.literals)


            # Capture the end time
            end_time = time.time()
            # Calculate the duration by subtracting the start time from the end time
            duration = end_time - start_time
            print(f"The code block took {duration} seconds to execute.")


        all_traces.append([all_images_of_a_trace, all_actions_of_a_trace, all_obs_of_a_trace, all_layouts_of_a_trace])
    

    return all_traces




# modify the images
# and construct pairs (of images)
# and construct the array of action (for each pair of images)
# and possibly construct special actions 
# Input: all_images, all_actions, all_layouts
# Return: pairs of images, corresponding actions (one-hot)
def modify_one_trace(all_images, all_actions, all_layouts, mean_all, std_all):
    
    # 1) preprocess images
    all_images = np.array(all_images)

    all_images_orig = np.copy(all_images)

    ## reduce dimension
    all_images = all_images[:, ::4, ::4, :]

    ## grey the images (optional) or black&white
    # all_images = rgb_to_grayscale(all_images)
    # all_images = to_black_and_white(all_images)

    ## normalize by color
    all_images, mean_, std_ = normalize_colors(all_images, mean_all, std_all)

    
    # building the array of pairs
    all_pairs_of_images = []
    all_pairs_of_images_orig = []
    for iiii, p in enumerate(all_images):
        #if iiii%2 == 0:
        if iiii < len(all_images)-1:
            all_pairs_of_images.append([all_images[iiii], all_images[iiii+1]])
            all_pairs_of_images_orig.append([all_images_orig[iiii], all_images_orig[iiii+1]])

    # plt.imsave("soko_pair0_1.png", all_pairs_of_images[1][1])
    # plt.close()


    #build array containing all actions (no duplicate)
    all_actions_unique = []
    for uuu, act in enumerate(all_actions):
        if str(act) not in all_actions_unique:
            all_actions_unique.append(str(act))

    #build array containing actions indices of the dataset
    all_actions_indices = []
    for ac in all_actions:
        all_actions_indices.append(all_actions_unique.index(str(ac)))
    
    print("all accc")
    print(len(all_actions))

    import torch
    import torch.nn.functional as F
    actions_indexes = torch.tensor(all_actions_indices)

    # array of one hot encoded actions
    actions_one_hot = F.one_hot(actions_indexes, num_classes=len(all_actions_unique))

    return all_pairs_of_images, all_pairs_of_images_orig, actions_one_hot, all_actions_unique


def load_dataset(path_to_file):
    # Load data.
    with open(path_to_file, mode="rb") as f:
        loaded_data = pickle.load(f)
    return loaded_data


# actions: a 1D list containing the action of each pair
# mean and sdt, numpy arrays
# image_pairs: a 2D list containing each pair of images of each transition
def save_dataset(dire, traces):


    # Save data.
    # data = {
    #     "observations": obs,
    #     "actions": actions,
    #     "images": images,
    #     "layouts": layouts
    # }
    data = {
        "traces": traces,
    }



    if not os.path.exists(dire):
        os.makedirs(dire) 

    filename = "data.p"
    with open(dire+"/"+filename, mode="wb") as f:
        pickle.dump(data, f)



# # 1) generate dataset (only once normally)
all_traces = generate_dataset()

# 2) save dataset
save_dataset("hanoi_dataset", all_traces)


def export_dataset():

    # 3) load
    loaded_dataset = load_dataset("/workspace/pddlgym-tests/pddlgym/hanoi_dataset/data.p")

    # for trace in load_dataset:
    #     all_images = trace[0]
    #     # all_images_of_a_trace, all_actions_of_a_trace, all_obs_of_a_trace, all_layouts_of_a_trace

    # 4) modify the dataset

    # modify_one_trace

    all_images = []
    all_pairs_of_images = []
    all_pairs_of_images_orig = []
    all_actions_one_hot = []
    all_actions_unique = []

    # first loop to compute the whole dataset mean and std
    for trace in loaded_dataset["traces"]:
        # all_images[:, ::4, ::4, :]
       
        all_images.extend(np.array(trace[0])[:, ::4, ::4, :])

    _, mean_all, std_all = normalize_colors(all_images, mean=None, std=None)


    # second loop to construct the pairs
    for trace in loaded_dataset["traces"]:

        all_images_tr = trace[0]
        all_actions_tr = trace[1]
        all_obs_tr = trace[2] 
        all_layouts_tr = trace[3]
        # all_images_of_a_trace, all_actions_of_a_trace, all_obs_of_a_trace, all_layouts_of_a_trace

        all_pairs_of_images_of_trace, all_pairs_of_images_orig_of_trace, actions_one_hot_of_trace, all_actions_unique_of_trace = modify_one_trace(all_images_tr, all_actions_tr, all_layouts_tr, mean_all, std_all)

        print("ici")
        print(len(all_pairs_of_images_of_trace))
        print(len(actions_one_hot_of_trace))

        print("la")
        all_pairs_of_images.extend(all_pairs_of_images_of_trace)
        all_pairs_of_images_orig.extend(all_pairs_of_images_orig_of_trace)
        all_actions_one_hot.extend(actions_one_hot_of_trace)





    all_actions_unique = all_actions_unique_of_trace

    return all_pairs_of_images, all_pairs_of_images_orig, all_actions_one_hot, mean_all, std_all, all_actions_unique

