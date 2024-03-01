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
    # Reverse the normalization process
    # unnormalized_images = normalized_images * (std + 1e-6) + mean
    # return np.round(unnormalized_images).astype(np.uint8)
    # mean = np.array(self.parameters["mean"])
    # std  = np.array(self.parameters["std"])
    return (normalized_images*std)+mean



def normalize_colors(images, mean=None, std=None):    
    if mean is None or std is None:
        mean      = np.mean(images, axis=0)
        std       = np.std(images, axis=0)
        print("was heree")
        print(len(images))
        print(mean.shape)
        print(std.shape)

    return (images - mean)/(std+1e-20), mean, std




# Function to reduce resolution of a single image using np.take
def reduce_resolution(image):
    # Use np.take to select every 4th element in the first two dimensions
    reduced_image = np.take(np.take(image, np.arange(0, image.shape[0], 9), axis=0),
                            np.arange(0, image.shape[1], 9), axis=1)
    return reduced_image



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




def unnormalize_colors(normalized_images, mean, std): 
    # Reverse the normalization process
    unnormalized_images = normalized_images * (std + 1e-6) + mean
    return np.round(unnormalized_images).astype(np.uint8)




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



# conditions pour "going back":
#  1) last disk moved == present disk moved
#  2) present peg destination == peg where the disk in question was before (in last obs)
### need i) present action, action before, peg_to_disc_list_current

def is_going_back(action_before, action_now, peg_to_disc_list_current):

    print("here1")
    
    if action_before is None:
        return False


    # {peg1:default: [d4:default, d3:default, d2:default], peg2:default: [], peg3:default: [d1:default], peg4:default: []}

    # pb si le disc qu'on déplace (action_now) vers peg_now ETAIT DEJA SITUE dans sur le même PEG 
    # au state précédent (à voir dans peg_to_disc_list_current) 

    # print("here2")

 

    # CONDITION 1: last disk moved == present disk moved ?

    #last_disk_moved 
    last_disk_moved = action_before.variables[0].name
    
    present_disk = action_now.variables[0].name

    if not last_disk_moved == present_disk:
        return False


    # CONDTION 2:
    #### need
    #####   1) current peg destination
    #####   2) peg where the disk in question was before

    #### 1) 

    ###    soit dans l'action (2e arg), soit, récup disk du 2e arg et récup à quel peg il est dans peg_to_disc_list_current

    current_peg_destination = None

    if 'peg' in action_now.variables[1].name:
        current_peg_destination = action_now.variables[1].name
    else:
        destination_disk = action_now.variables[1].name
        
        for keyy, valss in peg_to_disc_list_current.items():

            for dd in valss:
                if destination_disk == dd.name:
                    current_peg_destination = keyy.name


    print("action_before: {}".format(str(action_before)))
    print("action_now: {}".format(str(action_now)))
    print("peg_to_disc_list_current: {}".format(str(peg_to_disc_list_current)))

    print("current_peg_destination : {}".format(str(current_peg_destination)))


    return False


def origin_and_destination(action, peg_to_disc_list_pre, peg_to_disc_list_suc):

    # take the sent disk name
        
    disk_name = action.variables[0].name

    pre_peg = ''
    suc_peg = ''

    # look where it is in the pre lists
    for index, dico in enumerate(peg_to_disc_list_pre.values()):
        for dic_val in dico:
            if disk_name in str(dic_val):
                pre_peg = "peg"+str(index+1)

    # look where it is in the suc list
    for index, dico in enumerate(peg_to_disc_list_suc.values()):
        for dic_val in dico:
            if disk_name in str(dic_val):
                suc_peg = "peg"+str(index+1)

    # return the two "where"

    return pre_peg, suc_peg


# 1401
nb_samplings_per_starting_state = 7001 # has to be ODD 

# 20001
# 64
# 70

##
##
###
###  ça fait les paires et l'action
#### mais aussi toutes les images de ces pairs (pour autre traitement
#### par la suite)

def generate_problems():

    num_of_problems = 5


    action_before = None

    ## 1) Loading the env
    env = pddlgym.make("PDDLEnvHanoi39-v0", dynamic_action_space=True)
    # obs, debug_info = env.reset(_problem_idx=0)

    all_traces = []
    unique_obs = []
    unique_obs_img = []


    unique_transitions = [] # each ele holds a str representing a unique pair of peg_to_disc_list
    

    obs_occurences = {}
    
    counter = 0

    # looping over the number of starting positions
    # for blocksworld only the 1st pb has 4 blocks
    for ii in range(num_of_problems):



        #### 

        last_two_peg_to_disc_lists_str = [] # must contain only two lists that represent a legal transition
        
        last_two_peg_to_disc_lists = []
        
        # 
        last_two_imgs = []

        trace_transitions = []

        # all_images_of_a_trace = []
        # all_obs_of_a_trace = []
        # all_actions_of_a_trace = []
        # all_layouts_of_a_trace = []


        # Initializing the first State
        obs, debug_info = env.reset(_problem_idx=ii)

        # print("ii = {}".format(str(ii)))
        # continue


        # Retrieve the 1st image
        img, peg_to_disc_list = env.render()
        img = img[:,:,:3] # remove the transparancy

        if str(peg_to_disc_list) not in unique_obs:
            unique_obs.append(str(peg_to_disc_list))
            obs_occurences[str(peg_to_disc_list)] = 1
            unique_obs_img.append(img)

        last_two_peg_to_disc_lists_str.append(str(peg_to_disc_list))
        last_two_peg_to_disc_lists.append(peg_to_disc_list)
        last_two_imgs.append(img)

        # # adding the img and obs to all_*
        # all_images_of_a_trace.append(img)
        # all_layouts_of_a_trace.append(peg_to_disc_list)
        # all_obs_of_a_trace.append(obs.literals)

        # looping over the nber of states to sample for each starting position
        for jjj in range(nb_samplings_per_starting_state):
            
            if counter%10 == 0:
                print("counter: {}".format(str(counter)))

            # sample an action
            action = env.action_space.sample(obs)


            if action_before is not None:
                is_going_back(action_before, action, last_two_peg_to_disc_lists[-1])
                exit()




            # apply the action and retrieve img and obs
            obs, reward, done, debug_info = env.step(action)
            img, peg_to_disc_list = env.render()
            img = img[:,:,:3]


            last_two_peg_to_disc_lists_str.append(str(peg_to_disc_list))
            last_two_peg_to_disc_lists.append(peg_to_disc_list)
            if len(last_two_peg_to_disc_lists_str) > 2:
                last_two_peg_to_disc_lists_str.pop(0)
                last_two_peg_to_disc_lists.pop(0)

            last_two_imgs.append(img)
            if len(last_two_imgs) > 2:
                last_two_imgs.pop(0)

            if len(last_two_peg_to_disc_lists_str) == 2:

                #print(len(last_two_imgs))
                # plt.imsave("hanoi_"+str(jjj)+"_0.png", last_two_imgs[0])
                # plt.close()
                # plt.imsave("hanoi_"+str(jjj)+"_1.png", last_two_imgs[1])
                # plt.close()
                if str(last_two_peg_to_disc_lists_str) not in unique_transitions:


                    transition_actions = [] # hold all the version of the action
                    # i.e. loose, semi-loose-v1, semi-loose-v2

                    

                    pre_peg, post_peg = origin_and_destination(action, last_two_peg_to_disc_lists[0], last_two_peg_to_disc_lists[1])

                    transition_actions.append(str(action))
                    transition_actions.append(str(action)+pre_peg+post_peg)
                    transition_actions.append(str(action)+pre_peg)

                    unique_transitions.append(str(last_two_peg_to_disc_lists_str))
                    trace_transitions.append([[last_two_imgs[0], last_two_imgs[1]], transition_actions])
                
                    action_before = action


                # si une seule trace: la première à chaque fois, et 
                # en fin de trace, la dernière

                # comme plusieurs traces

                #all_images.append(last_two_imgs[0])


            if str(peg_to_disc_list) not in unique_obs:
                unique_obs.append(str(peg_to_disc_list))
                obs_occurences[str(peg_to_disc_list)] = 1
                unique_obs_img.append(img)
            else:
                obs_occurences[str(peg_to_disc_list)] += 1

            # all_images_of_a_trace.append(img)
            # all_actions_of_a_trace.append(action)
            # all_obs_of_a_trace.append(obs.literals)
            # all_layouts_of_a_trace.append(peg_to_disc_list)
            
            counter += 1

        all_traces.append(trace_transitions)

    # for kk, tracee in enumerate(all_traces):

    #     for ijij, trrrrrrrrrrr in enumerate(tracee):

    #         plt.imsave("hanoi_trace_"+str(kk)+"_trans_"+str(ijij)+"_im1.png", trrrrrrrrrrr[0][0])
    #         plt.close()
            

    #         plt.imsave("hanoi_trace_"+str(kk)+"_trans_"+str(ijij)+"_im2.png", trrrrrrrrrrr[0][1])
    #         plt.close()

    #         print("for trace_"+str(kk)+"_trans_"+str(ijij)+" action is "+str(trrrrrrrrrrr[1]))

    # exit()

    print("number of unique transitions is : {}".format(str(len(unique_transitions))))

    #print("number of all traces is : {}".format(len(all_traces)))

    return all_traces, obs_occurences, unique_obs_img



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
def save_dataset(dire, traces, obs_occurences, unique_obs_img):


    # Save data.
    # data = {
    #     "observations": obs,
    #     "actions": actions,
    #     "images": images,
    #     "layouts": layouts
    # }
    data = {
        "traces": traces,
        "obs_occurences": obs_occurences,
        "unique_obs_img": unique_obs_img
    }



    if not os.path.exists(dire):
        os.makedirs(dire) 

    filename = "data.p"
    with open(dire+"/"+filename, mode="wb") as f:
        pickle.dump(data, f)


# # # # 1) generate dataset (only once normally)
generate_problems()



exit()


# 2) save dataset
save_dataset("hanoi_3_9_dataset", all_traces, obs_occurences, unique_obs_img)

exit()

# loaded_dataset = load_dataset("/workspace/pddlgym-tests/pddlgym/hanoi_dataset/data.p")

# all_traces = loaded_dataset["traces"]



# # decompte des actions en fct de leur type

# loose_uniques = []

# loose_v1_uniques = []

# loose_v2_uniques = []

# for trace in all_traces:

#     #tr[0] # pair d'images

#     # loose

#     for trans in trace:

#         if trans[1][0] not in loose_uniques:
#             loose_uniques.append(trans[1][0])

#         if trans[1][1] not in loose_v1_uniques:
#             loose_v1_uniques.append(trans[1][1])

#         if trans[1][2] not in loose_v2_uniques:
#             loose_v2_uniques.append(trans[1][2])


# print("start")

# print(len(loose_uniques))
# print(len(loose_v1_uniques))
# print(len(loose_v2_uniques))

# exit()


# where we load the dataset, and adapt it to our needs
# action_type is semi_v1, semi_v2, loose
def export_dataset(action_type="semi_v1"):

    # 3) load
    loaded_dataset = load_dataset("/workspace/pddlgym-tests/pddlgym/hanoi_dataset/data.p")

    all_images_reduced = []
    all_actions = []
    all_pairs_of_images = []
    all_pairs_of_images_orig = []
    all_actions_one_hot = []
    all_actions_unique = []

    traces_indices = []

    # first loop to compute the whole dataset mean and std
    for iii, trace in enumerate(loaded_dataset["traces"]):


        print(type(trace))
        print(len(trace))

        print(type(trace[0]))

        print(len(trace[0]))


        traces_indices.append([iii*len(trace), (iii+1)*len(trace)])



        for transitio in trace:
            all_images_reduced.append(reduce_resolution(transitio[0][0])) # = im1
            all_images_reduced.append(reduce_resolution(transitio[0][1])) # = im2


        #all_actions.extend(trace[1]) # simplest version

        # at the same time, construct the augmented actions total array (to construct the "unique" array below)
        # concatenate both the simple action array and the layout desc of the pre state array
        
        # paired_gen = (str(a) + str(b) for a, b in zip(trace[1], trace[-1][:-1]))
        # all_actions.extend(list(paired_gen))

        if action_type == "semi_v1":
            all_actions.extend(trace[1][1])

        if action_type == "semi_v2":
            all_actions.extend(trace[1][2])

        if action_type == "loose":
            all_actions.extend(trace[1][0])


    unique_obs_img = loaded_dataset["unique_obs_img"]

    unique_obs_img_preproc, orig_max, orig_min = preprocess(unique_obs_img)


    # filtered_values = all_images_preproc[(all_images_preproc != 1.0) & (all_images_preproc != 0.0)]
    # print(filtered_values)

    # values will be centered on 0
    unique_obs_img_color_norm, mean_all, std_all = normalize_colors(unique_obs_img_preproc, mean=None, std=None)

    all_images_reduced_preproc = preprocess(all_images_reduced)
    all_images_reduced_norm, __, __ = normalize_colors(all_images_reduced_preproc, mean=mean_all, std=std_all)
    

    #build array containing all actions WITHOUT DUPLICATE
    for uuu, act in enumerate(all_actions):
        if str(act) not in all_actions_unique:
            all_actions_unique.append(str(act))



    # second loop to process the pairs
    for iiii, trace in enumerate(loaded_dataset["traces"]):


        # all_images_transfo_tr
        # all_images_orig_tr 

        all_images_transfo_tr = all_images_reduced_norm[traces_indices[iiii][0]:traces_indices[iiii][1]]
        all_images_orig_reduced_tr = all_images_reduced[traces_indices[iiii][0]:traces_indices[iiii][1]]



        all_images_tr = trace[0]
        all_actions_tr = trace[1]
        all_obs_tr = trace[2] 
        all_layouts_tr = trace[3]


        # concatenate both the simple action array and the layout desc of the pre state array
        paired_gen = (str(a) + str(b) for a, b in zip(all_actions_tr, all_layouts_tr[:-1]))
        all_actions_transfo = list(paired_gen)
        
        
        #all_actions_transfo = all_actions_tr

        # all_images_of_a_trace, all_actions_of_a_trace, all_obs_of_a_trace, all_layouts_of_a_trace
        all_pairs_of_images_of_trace, all_pairs_of_images_orig_reduced_of_trace, actions_one_hot_of_trace = modify_one_trace(all_images_transfo_tr, all_images_orig_reduced_tr, all_actions_transfo, all_layouts_tr, mean_all, std_all, all_actions_unique)

        
        all_pairs_of_images.extend(all_pairs_of_images_of_trace)
        all_pairs_of_images_reduced_orig.extend(all_pairs_of_images_orig_reduced_of_trace)
        all_actions_one_hot.extend(actions_one_hot_of_trace)




    return all_pairs_of_images, all_pairs_of_images_reduced_orig, all_actions_one_hot, mean_all, std_all, all_actions_unique, orig_max, orig_min


# all_pairs_of_images, all_pairs_of_images_reduced_orig, all_actions_one_hot, mean_all, std_all, all_actions_unique, orig_max, orig_min = export_dataset()



# # # # 352
# # # print(len(all_actions_unique))

# # # exit()

# for hh in range(0, len(all_pairs_of_images_reduced_orig), 500):

#     acc = all_actions_unique[np.argmax(all_actions_one_hot[hh])]
#     print("action for {} is {}".format(str(hh), str(acc)))

#     im1_orig=all_pairs_of_images_reduced_orig[hh][0]
#     im2_orig=all_pairs_of_images_reduced_orig[hh][1]

#     plt.imsave("hanoi_pair_"+str(hh)+"_pre.png", im1_orig)
#     plt.close()

#     plt.imsave("hanoi_pair_"+str(hh)+"_suc.png", im2_orig)
#     plt.close()

#     #


# exit()