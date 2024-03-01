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
import random
import torch
import torch.nn.functional as F
from skimage import exposure

def normalize(image):
    # into 0-1 range
    if image.max() == image.min():
        return image - image.min()
    else:
        return (image - image.min())/(image.max() - image.min()), image.max(), image.min()

def equalize(image):
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


def gaussian(a, sigma=0.3):
    if sigma == 20:
        np.random.seed(1)
    elif sigma == 30:
        np.random.seed(2)
    elif sigma == 40:
        np.random.seed(3)
    return np.random.normal(0,sigma,a.shape) + a




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
    return (normalized_images*std)+mean



def normalize_colors(images, mean=None, std=None):    
    if mean is None or std is None:
        mean      = np.mean(images, axis=0)
        std       = np.std(images, axis=0)
    return (images - mean)/(std+1e-20), mean, std




# Function to reduce resolution of a single image using np.take
def reduce_resolution(image):
    # Use np.take to select every 4th element in the first two dimensions
    reduced_image = np.take(np.take(image, np.arange(0, image.shape[0], 9), axis=0),
                            np.arange(0, image.shape[1], 9), axis=1)

    return reduced_image


def unnormalize_colors(normalized_images, mean, std): 
    # Reverse the normalization process
    unnormalized_images = normalized_images * (std + 1e-6) + mean
    return np.round(unnormalized_images).astype(np.uint8)



# return the origin and destination of a moved disk
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
nb_samplings_per_starting_state = 501 # has to be ODD 




def generate_dataset():

    ## 1) Loading the env
    env = pddlgym.make("PDDLEnvHanoi44-v0", dynamic_action_space=True)

    all_traces = []
    unique_obs = []
    unique_obs_img = []
    unique_transitions = [] # each ele holds a str representing a unique pair of peg_to_disc_list
    obs_occurences = {}
    counter = 0

    # looping over the number of starting positions
    # for blocksworld only the 1st pb has 4 blocks
    for ii in range(1, 52, 1):

        last_two_peg_to_disc_lists_str = [] # must contain only two lists that represent a legal transition
        last_two_peg_to_disc_lists = []
        last_two_imgs = []
        trace_transitions = []

        # Initializing the first State
        obs, debug_info = env.reset(_problem_idx=ii)

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

        # looping over the nber of states to sample for each starting position
        for jjj in range(nb_samplings_per_starting_state):
            
            if counter%10 == 0:
                print("counter: {}".format(str(counter)))

            # sample an action
            action = env.action_space.sample(obs)

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

                if str(last_two_peg_to_disc_lists_str) not in unique_transitions:

                    transition_actions = [] # hold all the version of the action
                    # i.e. loose, semi-loose-v1, semi-loose-v2

                    pre_peg, post_peg = origin_and_destination(action, last_two_peg_to_disc_lists[0], last_two_peg_to_disc_lists[1])

                    transition_actions.append(str(action))
                    transition_actions.append(str(action)+pre_peg+post_peg)
                    transition_actions.append(str(action)+pre_peg)
                    transition_actions.append(str(last_two_peg_to_disc_lists)) # action full description
                    unique_transitions.append(str(last_two_peg_to_disc_lists_str))
                    trace_transitions.append([[last_two_imgs[0], last_two_imgs[1]], transition_actions])

            if str(peg_to_disc_list) not in unique_obs:
                unique_obs.append(str(peg_to_disc_list))
                obs_occurences[str(peg_to_disc_list)] = 1
                unique_obs_img.append(img)
            else:
                obs_occurences[str(peg_to_disc_list)] += 1

            counter += 1

        all_traces.append(trace_transitions)

    print("number of unique transitions is : {}".format(str(len(unique_transitions))))

    with open("resultatHanoi4-4.txt", 'w') as file2:

        file2.write(str(unique_transitions) + '\n')

    return all_traces, obs_occurences, unique_obs_img, unique_transitions



# construct pairs (of images)
# and construct the array of action (for each pair of images) 
# Return: pairs of images, corresponding actions (one-hot)
def modify_one_trace(all_images_transfo_tr, all_images_orig_tr, all_actions, all_actions_unique_):

    # building the array of pairs
    all_pairs_of_images = []
    all_pairs_of_images_orig = []
    for iiii, p in enumerate(all_images_transfo_tr):
        if iiii%2 == 0:
            #if iiii < len(all_images_transfo_tr)-1:
            all_pairs_of_images.append([all_images_transfo_tr[iiii], all_images_transfo_tr[iiii+1]])
            all_pairs_of_images_orig.append([all_images_orig_tr[iiii], all_images_orig_tr[iiii+1]])

    #build array containing actions indices of the dataset
    all_actions_indices = []
    for ac in all_actions:
        all_actions_indices.append(all_actions_unique_.index(str(ac)))
    
    actions_indexes = torch.tensor(all_actions_indices).long()

    # array of one hot encoded actions
    actions_one_hot = F.one_hot(actions_indexes, num_classes=len(all_actions_unique_))

    return all_pairs_of_images, all_pairs_of_images_orig, actions_one_hot.numpy()


def load_dataset(path_to_file):
    # Load data.
    with open(path_to_file, mode="rb") as f:
        loaded_data = pickle.load(f)
    return loaded_data


def save_dataset(dire, traces, obs_occurences, unique_obs_img, unique_transitions):
    data = {
        "traces": traces,
        "obs_occurences": obs_occurences,
        "unique_obs_img": unique_obs_img,
        "unique_transitions": unique_transitions
    }
    if not os.path.exists(dire):
        os.makedirs(dire) 
    filename = "data.p"
    with open(dire+"/"+filename, mode="wb") as f:
        pickle.dump(data, f)


# # 1) generate dataset (only once normally)
# all_traces, obs_occurences, unique_obs_img, unique_transitions = generate_dataset()

# # 2) save dataset
# save_dataset("hanoi_4_4_dataset", all_traces, obs_occurences, unique_obs_img, unique_transitions)

# exit()


def create_a_trace():
    # 3) load
    loaded_dataset = load_dataset("/workspace/pddlgym-tests/pddlgym/hanoi_4_4_dataset/data.p")
    all_images_reduced = []
    all_actions = []
    all_pairs_of_images = []
    all_pairs_of_images_reduced_orig = []
    all_actions_one_hot = []
    all_actions_unique = []
    traces_indices = []
    traces_actions_indices = []
    start_trace_index = 0
    start_trace_action_index = 0
    all_actions_for_trace = []
    # first loop to compute the whole dataset mean and std
    for iii, trace in enumerate(loaded_dataset["traces"]):
        # traces_actions_indices
        for trtrtr, transitio in enumerate(trace):
            plt.imsave("hanoi_pair_"+str(trtrtr)+"_pre.png", reduce_resolution(transitio[0][0]))
            plt.close()
            plt.imsave("hanoi_pair_"+str(trtrtr)+"_suc.png", reduce_resolution(transitio[0][1]))
            plt.close()
            if trtrtr > 7:
                break
        break
    return






def export_dataset(action_type="full"):


    loaded_dataset = load_dataset("/workspace/pddlgym-tests/pddlgym/hanoi_4_4_dataset/data.p")

    all_images_reduced = []
    all_actions = []
    all_pairs_of_images = []
    all_pairs_of_images_reduced_orig = []
    all_actions_one_hot = []
    all_actions_unique = []

    traces_indices = []
    traces_actions_indices = []
    start_trace_index = 0
    start_trace_action_index = 0
    all_actions_for_trace = []

    # first loop to compute the whole dataset mean and std
    for iii, trace in enumerate(loaded_dataset["traces"]):

        traces_indices.append([start_trace_index, start_trace_index+len(trace)*2])
        start_trace_index+=len(trace)*2

        traces_actions_indices.append([start_trace_action_index, start_trace_action_index+len(trace)])
        start_trace_action_index+=len(trace)
        
        actions_for_one_trace = []

        for trtrtr, transitio in enumerate(trace):

            all_images_reduced.append(reduce_resolution(transitio[0][0])) # = im1
            all_images_reduced.append(reduce_resolution(transitio[0][1])) # = im2

            # plt.imsave("hanoi_pair_"+str(trtrtr)+"_pre.png", reduce_resolution(transitio[0][0]))
            # plt.close()
          
            if action_type == "semi_v1":
                all_actions.append(str(transitio[1][1]))
                actions_for_one_trace.append(str(transitio[1][1]))

            elif action_type == "semi_v2":
                all_actions.append(str(transitio[1][2]))
                actions_for_one_trace.append(str(transitio[1][2]))

            elif action_type == "loose":
                all_actions.append(str(transitio[1][0]))
                actions_for_one_trace.append(str(transitio[1][0]))

            elif action_type == "full":
                all_actions.append(str(transitio[1][3]))
                actions_for_one_trace.append(str(transitio[1][3]))

        all_actions_for_trace.append(actions_for_one_trace)

    unique_obs_img = loaded_dataset["unique_obs_img"]


    reduced_uniques = []
    for uniq in unique_obs_img:

        reduced_uniques.append(np.clip(gaussian(reduce_resolution(uniq), 20), 0, 255))
        reduced_uniques.append(np.clip(gaussian(reduce_resolution(uniq), 30), 0, 255))
        reduced_uniques.append(np.clip(gaussian(reduce_resolution(uniq), 40), 0, 255))


    unique_obs_img_preproc, orig_max, orig_min = preprocess(reduced_uniques)
    unique_obs_img_color_norm, mean_all, std_all = normalize_colors(unique_obs_img_preproc, mean=None, std=None)

    # all_images_reduced > gaussian > clip > preprocess > normalize_colors

    all_images_reduced = np.array(all_images_reduced)
    # 
    all_images_reduced_gaussian_20 = np.clip(gaussian(all_images_reduced, 20), 0, 255)
    all_images_reduced_gaussian_30 = np.clip(gaussian(all_images_reduced, 30), 0, 255)
    all_images_reduced_gaussian_40 = np.clip(gaussian(all_images_reduced, 40), 0, 255)

    all_images_reduced_gaussian_20_preproc, _, _ = preprocess(all_images_reduced_gaussian_20)
    all_images_reduced_gaussian_20_norm, __, __ = normalize_colors(all_images_reduced_gaussian_20_preproc, mean=mean_all, std=std_all)
    
    all_images_reduced_gaussian_30_preproc, _, _ = preprocess(all_images_reduced_gaussian_30)
    all_images_reduced_gaussian_30_norm, __, __ = normalize_colors(all_images_reduced_gaussian_30_preproc, mean=mean_all, std=std_all)

    all_images_reduced_gaussian_40_preproc, _, _ = preprocess(all_images_reduced_gaussian_40)
    all_images_reduced_gaussian_40_norm, __, __ = normalize_colors(all_images_reduced_gaussian_40_preproc, mean=mean_all, std=std_all)

    #build array containing all actions WITHOUT DUPLICATE
    for uuu, act in enumerate(all_actions):
        if str(act) not in all_actions_unique:
            all_actions_unique.append(str(act))


    all_pairs_of_images_processed_gaussian20 = []
    all_pairs_of_images_processed_gaussian30 = []
    all_pairs_of_images_processed_gaussian40 = []


    # second loop to process the pairs
    for iiii, trace in enumerate(loaded_dataset["traces"]):


        all_images_transfo_tr_gaussian20 = all_images_reduced_gaussian_20_norm[traces_indices[iiii][0]:traces_indices[iiii][1]]
        all_images_orig_reduced_tr = all_images_reduced[traces_indices[iiii][0]:traces_indices[iiii][1]]

        all_images_transfo_tr_gaussian30 = all_images_reduced_gaussian_30_norm[traces_indices[iiii][0]:traces_indices[iiii][1]]
        all_images_transfo_tr_gaussian40 = all_images_reduced_gaussian_40_norm[traces_indices[iiii][0]:traces_indices[iiii][1]]


        all_actions_tr = all_actions_for_trace[iiii]
        

        # all_images_of_a_trace, all_actions_of_a_trace, all_obs_of_a_trace, all_layouts_of_a_trace
        all_pairs_of_images_of_trace_gaussian20, all_pairs_of_images_orig_reduced_of_trace, actions_one_hot_of_trace = modify_one_trace(all_images_transfo_tr_gaussian20, all_images_orig_reduced_tr, all_actions_tr, all_actions_unique)
        all_pairs_of_images_of_trace_gaussian30, _, _ = modify_one_trace(all_images_transfo_tr_gaussian30, all_images_orig_reduced_tr, all_actions_tr, all_actions_unique)
        all_pairs_of_images_of_trace_gaussian40, _, _ = modify_one_trace(all_images_transfo_tr_gaussian40, all_images_orig_reduced_tr, all_actions_tr, all_actions_unique)

        all_pairs_of_images_processed_gaussian20.extend(all_pairs_of_images_of_trace_gaussian20)
        all_pairs_of_images_processed_gaussian30.extend(all_pairs_of_images_of_trace_gaussian30)
        all_pairs_of_images_processed_gaussian40.extend(all_pairs_of_images_of_trace_gaussian40)
        
        #all_pairs_of_images.extend(all_pairs_of_images_of_trace)
        all_pairs_of_images_reduced_orig.extend(all_pairs_of_images_orig_reduced_of_trace)
        all_actions_one_hot.extend(actions_one_hot_of_trace)



    ###   alternatively take 2 copies from 20/30, 20/40, 30/40 and put them in the training set
    ###   and 1 copy for the other gaussian array (resp. 40 30 20) and put it in the test_val set

    train_set = []
    test_val_set = []
    for i in range(0, len(all_pairs_of_images_processed_gaussian20)):
        if i%3 == 0:
            train_set.append([all_pairs_of_images_processed_gaussian20[i], all_actions_one_hot[i]])
            train_set.append([all_pairs_of_images_processed_gaussian30[i], all_actions_one_hot[i]])
            test_val_set.append([all_pairs_of_images_processed_gaussian40[i], all_actions_one_hot[i]])
        elif (i+1)%3 == 0:
            train_set.append([all_pairs_of_images_processed_gaussian20[i], all_actions_one_hot[i]])
            train_set.append([all_pairs_of_images_processed_gaussian40[i], all_actions_one_hot[i]])
            test_val_set.append([all_pairs_of_images_processed_gaussian30[i], all_actions_one_hot[i]])
        elif (i+2)%3 == 0:
            train_set.append([all_pairs_of_images_processed_gaussian30[i], all_actions_one_hot[i]])
            train_set.append([all_pairs_of_images_processed_gaussian40[i], all_actions_one_hot[i]])
            test_val_set.append([all_pairs_of_images_processed_gaussian20[i], all_actions_one_hot[i]])
    


    return train_set, test_val_set, all_pairs_of_images_reduced_orig, all_actions_one_hot, mean_all, std_all, all_actions_unique, orig_max, orig_min


# create_a_trace()


#train_set, test_val_set, all_pairs_of_images_reduced_orig, all_actions_one_hot, mean_all, std_all, all_actions_unique, orig_max, orig_min = export_dataset()

# # # # 352
# # # print(len(all_actions_unique))




# for hh in range(0, len(train_set), 50):

#     acc = all_actions_unique[np.argmax(train_set[hh][1])]
#     print("action for {} is {}".format(str(hh), str(acc)))

#     # im1_orig=all_pairs_of_images_reduced_orig[hh][0]
#     # im2_orig=all_pairs_of_images_reduced_orig[hh][1]
#     im1 = train_set[hh][0][0]
#     im2 = train_set[hh][0][1]

#     im1 = unnormalize_colors(im1, mean_all, std_all)
#     im1 = deenhance(im1)
#     im1 = denormalize(im1, orig_min, orig_max)
#     im1 = np.clip(im1, 0, 1)
    
#     im2 = unnormalize_colors(im2, mean_all, std_all)
#     im2 = deenhance(im2)
#     im2 = denormalize(im2, orig_min, orig_max)
#     im2 = np.clip(im2, 0, 1)


#     plt.imsave("hanoi_pair_"+str(hh)+"_pre.png", im1)
#     plt.close()

#     plt.imsave("hanoi_pair_"+str(hh)+"_suc.png", im2)
#     plt.close()

#     #


# exit()




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