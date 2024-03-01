import pddlgym
import imageio
#from pddlgym_planners.fd import FD
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os
import json
from concurrent.futures import ProcessPoolExecutor



def normalize(image):
    # into 0-1 range
    if image.max() == image.min():
        return image - image.min()
    else:
        return (image - image.min())/(image.max() - image.min()), image.max(), image.min()

def equalize(image):
    from skimage import exposure
    return exposure.equalize_hist(image)

def enhance(image):
    return np.clip((image-0.5)*3,-0.5,0.5)+0.5

def preprocess(image):
    image = np.array(image)
    image = image / 255.
    image = image.astype(float)
    image = equalize(image)
    image, orig_max, orig_min = normalize(image)
    image = enhance(image)
    return image, orig_max, orig_min



# Function to reduce resolution of a single image using np.take
def reduce_resolution(image):
    # Use np.take to select every 4th element in the first two dimensions
    reduced_image = np.take(np.take(image, np.arange(0, image.shape[0], 9), axis=0),
                            np.arange(0, image.shape[1], 9), axis=1)
    return reduced_image



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



def deenhance(enhanced_image):
    # Reverse the final shift by subtracting 0.5
    temp_image = enhanced_image - 0.5
    
    # Reverse the clipping: Since clipping limits values, we cannot fully recover original values if they were outside the [-0.5, 0.5] range. 
    # However, for values within the range, we can reverse the scale by dividing by 3.
    # We assume that the enhanced image has values within the range that the clip function allowed.
    temp_image = temp_image / 3
    
    # Reverse the initial shift by adding 0.5 back
    original_image = temp_image + 0.5
    
    return original_image

def denormalize(normalized_image, original_min, original_max):
    if original_max == original_min:
        return normalized_image + original_min
    else:
        return (normalized_image * (original_max - original_min)) + original_min




def unnormalize_colors(normalized_images, mean, std):

    # # Reverse the normalization process
    # unnormalized_images = normalized_images * (std + 1e-6) + mean
    # return np.round(unnormalized_images).astype(np.uint8)

    return (normalized_images*std)+mean




def normalize_colors(images, mean=None, std=None):    
    if mean is None or std is None:
        mean      = np.mean(images, axis=0)
        std       = np.std(images, axis=0)
        print("was heree")
        print(len(images))
        print(mean.shape)
        print(std.shape)

    return (images - mean)/(std+1e-6), mean, std





# Preprocessing (as recommended by chatpgt, also the order is recommended by chatgpt)

# 0) rescalling pixels value from 0-255 to 0-1 (NO NEED, step 1 aleady rescale)
# 1) reducing the resolution of the images 
# 2) normalizing the colors on each color channel (by substracing the mean and dividing by std)


import re

def extract_substrings(input_string):
    # Define the regular expression pattern to match 'd3:default' and 'peg2:default'
    # This pattern looks for any word character (alphanumeric + '_') followed by a digit,
    # a colon, and the word 'default'
    pattern = r'\w+\d:default'

    # Find all matches in the input string
    matches = re.findall(pattern, input_string)

    # Return the matched substrings
    return matches



obs_tmp = [] # array of the last two observations, 1st ele is the last

img_tmp = []

action_tmp = []

blocks_data_tmp = []

obs_occurences = {}



def generate_dataset():



    ## 1) Loading the env
    env = pddlgym.make("PDDLEnvBlocks-v0", dynamic_action_space=True)
    # obs, debug_info = env.reset(_problem_idx=0)

    all_traces = []



    # looping over the number of starting positions
    # for blocksworld only the 1st pb has 4 blocks
    for ii in range(7):


        nb_max_of_samplings = 101 # has to be ODD 

        counter = 0

        all_images_of_a_trace = []
        all_obs_of_a_trace = []
        all_actions_of_a_trace = []
        all_layouts_of_a_trace = []

        # Initializing the first State
        obs, debug_info = env.reset(_problem_idx=ii)
        obs_tmp.append(obs)

        #print(obs)
        # Retrieve the 1st image
        img, blocks_data = env.render()
        img = img[:,:,:3] # remove the transparancy

        img_tmp.append(img)
        if len(img_tmp) > 2:
            img_tmp.pop(0)

        blocks_data_tmp.append(blocks_data)
        if len(blocks_data_tmp) > 2:
            blocks_data_tmp.pop(0)
            
        if str(blocks_data) not in obs_occurences.keys():
            obs_occurences[str(blocks_data)] = 1


        # adding the img and obs to all_*
        all_images_of_a_trace.append(img)
        all_layouts_of_a_trace.append(blocks_data)
        all_obs_of_a_trace.append(obs.literals)


        # plt.imsave("blocks_"+str(counter)+".png", img)
        # plt.close()


        # looping over the nber of states to sample for each starting position
        while counter < nb_max_of_samplings:

            action = env.action_space.sample(obs)
            if len(action_tmp) == 2:

                counterr=0
                while str(action_tmp[0]) == str(action):

                    action = env.action_space.sample(obs)
                    if counterr > 10:
                        break
                    counterr+=1

            obs, reward, done, debug_info = env.step(action)


            img, blocks_data = env.render(action_was=action)
            img = img[:,:,:3]

            img_tmp.append(img)
            if len(img_tmp) > 2:
                img_tmp.pop(0)

            blocks_data_tmp.append(blocks_data)
            if len(blocks_data_tmp) > 2:
                blocks_data_tmp.pop(0)


            if str(blocks_data) not in obs_occurences.keys():
                obs_occurences[str(blocks_data)] = 1
            else:
                obs_occurences[str(blocks_data)] += 1
                #if obs_occurences[str(blocks_data)] < 20:


            # add the actions to all_*
            action_tmp.append(action)
            if len(action_tmp) > 2:
                action_tmp.pop(0)

            all_actions_of_a_trace.append(action)
            all_images_of_a_trace.append(img)
            all_layouts_of_a_trace.append(blocks_data)
            all_obs_of_a_trace.append(obs.literals)

            if counter%10 == 0:
                print("counter: {}".format(str(counter)))

            # if counter > 10000:
            #     exit()

            counter += 1




            # print(str(blocks_data_tmp[0][0]) == '[[a:block], [], [c:block, b:block]]')
            # exit()

            # if "unstack" in str(action) and str(blocks_data_tmp[0][0]) == '[[], [a:block], [c:block, b:block]]':
            #     print("action UNSTACK for block {} was {}".format(str(counter), str(action)))

            #     plt.imsave("blocks_unstack_"+str(counter)+"_pre.png", img_tmp[0])
            #     plt.close()

            #     plt.imsave("blocks_unstack_"+str(counter)+"_succ.png", img_tmp[1])
            #     plt.close()


        all_traces.append([all_images_of_a_trace, all_actions_of_a_trace, all_obs_of_a_trace, all_layouts_of_a_trace])

    print("obs_occurences")
    print(obs_occurences)
    print(len(obs_occurences))

    # for kk in obs_occurences.keys():
    #     print(kk)

    # exit()
    return all_traces




# modify the images
# and construct pairs (of images)
# and construct the array of action (for each pair of images)
# and possibly construct special actions 
# Input: all_images, all_actions, all_layouts
# Return: pairs of images, corresponding actions (one-hot)
def modify_one_trace(all_images_transfo_tr, all_images_orig_tr, all_actions, all_layouts, mean_all, std_all, all_actions_unique_):
    
    
    # building the array of pairs
    all_pairs_of_images = []
    all_pairs_of_images_orig = []
    for iiii, p in enumerate(all_images_transfo_tr):
        #if iiii%2 == 0:
        if iiii < len(all_images_transfo_tr)-1:
            all_pairs_of_images.append([all_images_transfo_tr[iiii], all_images_transfo_tr[iiii+1]])
            all_pairs_of_images_orig.append([all_images_orig_tr[iiii], all_images_orig_tr[iiii+1]])

    # plt.imsave("hanoi_pair0_1.png", all_pairs_of_images[1][1])
    # plt.close()

    #build array containing actions indices of the dataset
    all_actions_indices = []
    for ac in all_actions:
        all_actions_indices.append(all_actions_unique_.index(str(ac)))
    
    print("all accc")
    print(len(all_actions))
    print(len(all_actions_unique_))

    import torch
    import torch.nn.functional as F
    actions_indexes = torch.tensor(all_actions_indices)

    # array of one hot encoded actions
    actions_one_hot = F.one_hot(actions_indexes, num_classes=len(all_actions_unique_))


    return all_pairs_of_images, all_pairs_of_images_orig, actions_one_hot.numpy()


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

    print("saved")

    return

# # # 1) generate dataset (only once normally)
# all_traces = generate_dataset()


# # 2) save dataset
# save_dataset("blocks_dataset", all_traces)

# exit()


# where we load the dataset, and adapt it to our needs
def export_dataset():

    # 3) load
    loaded_dataset = load_dataset("/workspace/pddlgym-tests/pddlgym/blocks_dataset/data.p")

    # for trace in load_dataset:
    #     all_images = trace[0]
    #     # all_images_of_a_trace, all_actions_of_a_trace, all_obs_of_a_trace, all_layouts_of_a_trace

    # 4) modify the dataset


    all_images = []
    all_actions = []
    all_pairs_of_images = []
    all_pairs_of_images_orig = []
    all_actions_one_hot = []
    all_actions_unique = []
    traces_indices = []


    # ## black&white
    # #loaded_dataset["traces"][0] = to_black_and_white(np.array(loaded_dataset["traces"][0]))
    # for jjj, trace in enumerate(loaded_dataset["traces"]):
    #     loaded_dataset["traces"][jjj][0] = to_black_and_white(np.array(trace[0]))

    # first loop to compute the whole dataset mean and std
    for iii, trace in enumerate(loaded_dataset["traces"]):
        # all_images[:, ::4, ::4, :]
        traces_indices.append([iii*len(trace[0]), (iii+1)*len(trace[0])])
        #start_time = time.time()


        # all_images_of_a_trace, all_actions_of_a_trace, all_obs_of_a_trace, all_layouts_of_a_trace
        
        print("iii {}".format(str(iii)))


        print(type(trace[0]))
        print(len(trace[0]))

        print()
        print(trace[0][0].shape)

        # reduced_resolution = np.take(np.take(np.take(trace[0], np.arange(0, trace[0].shape[1], 4), axis=1),
        #                                     np.arange(0, trace[0].shape[2], 4), axis=2),
        #                             np.arange(0, trace[0].shape[3], 1), axis=3)

        reduced_images = [reduce_resolution(img) for img in trace[0]]
        all_images.extend(reduced_images)
        #all_images.extend(np.array(trace[0])[:, ::4, ::4, :])


        # inter_time = time.time()
        # duration = inter_time - start_time
        # print(f"The inter time 0 is {duration} seconds.")

        #all_actions.extend(trace[1]) # simplest version


        # concatenate both the simple action array and the layout desc of the pre state array
        paired_gen = (str(a) + str(b) for a, b in zip(trace[1], trace[-1][:-1]))
        all_actions.extend(list(paired_gen))


        # inter_time = time.time()
        # duration = inter_time - start_time
        # print(f"The inter time 1 is {duration} seconds.")

    #all_images = np.array(all_images)

    # print(all_images[0])

    all_images_preproc, orig_max, orig_min = preprocess(all_images)


    all_images_color_norm, mean_all, std_all = normalize_colors(all_images_preproc, mean=None, std=None)




    #build array containing all actions WITHOUT DUPLICATE
    for uuu, act in enumerate(all_actions):
        if str(act) not in all_actions_unique:
            all_actions_unique.append(str(act))

    print(len(all_actions_unique)) # 1048


    # second loop to construct the pairs
    for iiii, trace in enumerate(loaded_dataset["traces"]):


        all_images_transfo_tr = all_images_color_norm[traces_indices[iiii][0]:traces_indices[iiii][1]]
        all_images_orig_tr = all_images[traces_indices[iiii][0]:traces_indices[iiii][1]]


        all_images_tr = trace[0]
        all_actions_tr = trace[1]
        all_obs_tr = trace[2] 
        all_layouts_tr = trace[3]



        # concatenate both the simple action array and the layout desc of the pre state array
        paired_gen = (str(a) + str(b) for a, b in zip(all_actions_tr, all_layouts_tr[:-1]))
        all_actions_transfo = list(paired_gen)


        # all_images_of_a_trace, all_actions_of_a_trace, all_obs_of_a_trace, all_layouts_of_a_trace
        all_pairs_of_images_of_trace, all_pairs_of_images_orig_of_trace, actions_one_hot_of_trace = modify_one_trace(all_images_transfo_tr, all_images_orig_tr, all_actions_transfo, all_layouts_tr, mean_all, std_all, all_actions_unique)


        # exit()

        print("la")
        all_pairs_of_images.extend(all_pairs_of_images_of_trace)
        all_pairs_of_images_orig.extend(all_pairs_of_images_orig_of_trace)
        all_actions_one_hot.extend(actions_one_hot_of_trace)




    return all_pairs_of_images, all_pairs_of_images_orig, all_actions_one_hot, mean_all, std_all, all_actions_unique, orig_max, orig_min



# all_pairs_of_images, all_pairs_of_images_orig, all_actions_one_hot, mean_all, std_all, all_actions_unique, orig_max, orig_min = export_dataset()


# print("lkeeeeeeee")

# # print(len(all_pairs_of_images))
# print("ici")

# for hh in range(10):

#     # unorma = unnormalize_colors(all_pairs_of_images, mean_all, std_all)

#     # dehanced = deenhance(unorma)

#     # denorm = denormalize(dehanced, orig_min, orig_max)



#     print(all_actions_unique[np.argmax(all_actions_one_hot[hh])])
#     plt.imsave("blocks_"+str(hh)+"_00.png", all_pairs_of_images_orig[hh][0])
#     plt.close()

#     plt.imsave("blocks_"+str(hh)+"_11.png", all_pairs_of_images_orig[hh][1])
#     plt.close()

# exit()