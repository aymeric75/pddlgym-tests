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
from skimage import exposure, color
import itertools







def all_colors():

    colors = [0, 128, 255]
    #colors =  [ 0, 21,  73,  95, 114, 123, 141, 143, 171, 193, 204, 228, 236, 238, 255 ]
    #combinations = list(itertools.product(colors, repeat=3))
    combinations = [(51, 255, 51), (255, 0, 255)]
    #combinations = [(128,128,128)]

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

    # print(type(rgb))
    # exit()
    global counntteerr

    if counntteerr % 100000 == 0:
        print("counter is {}".format(str(counntteerr)))

    if counntteerr == len(boolean_matrix):
        return rgb

    #if counntteerr < len(boolean_matrix):
    if boolean_matrix[counntteerr]:

        if (rgb == np.array([0,0,0])).all():
            counntteerr+=1
            #return np.array([32,32,32])
            return np.array([255,255,255])

        elif (rgb == np.array([255,255,255])).all():
            counntteerr+=1
            #return np.array([224,224,224])
            return np.array([0,0,0])
            
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
    seed=1
    np.random.seed(seed)


    if not isinstance(images, np.ndarray):
        images = np.array(images)

    # print(images.shape) # (768, 25, 70, 3)
    # # boolean_matrix 


    # convert the image dataset into a "lab" format 

    reshaped_rgb = images.reshape(-1, 3)  # Reshape to a 2D array where each row is a pixel's RGB values
    print(reshaped_rgb.shape) # 


    # Generate a matrix of random numbers from a uniform distribution
    random_matrix = np.random.uniform(low=0.0, high=1.0, size=(reshaped_rgb.shape[0],))
    # Create a boolean matrix: True if the element is < 0.05, False otherwise
    boolean_matrix = random_matrix < 0.01
    #print(boolean_matrix)

    lab_pixels = np.apply_along_axis(convert_to_lab_and_to_color_wt_min_distance, 1, reshaped_rgb, boolean_matrix)
    


    #print("seed {}, images.shape {}, reshaped_rgb {}, lab_pixels {}, bool matrix {}".format(str(seed), str(images.shape), str(reshaped_rgb.shape), str(lab_pixels.shape), str(boolean_matrix.shape)))


    #dataset_lab = lab_pixels.reshape(768, 25, 70, 3)
    dataset_lab = lab_pixels.reshape(images.shape)

    # NOW holds the closests values  

    return dataset_lab




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
    #image = image[:,:,:,0]
    #image = image / 255. #
    #print(np.unique(image))
    image = image.astype(float)
    image = equalize(image) # in present case, from values like [0 0.001 0.97 1]
    #print(np.unique(image))
    image, orig_max, orig_min = normalize(image)
    #print(np.unique(image))
    image = enhance(image)
    #print(np.unique(image))
    return image, orig_max, orig_min


def gaussian(a, sigma=0.3):

    if sigma == 0.01:
        np.random.seed(1)
    elif sigma == 0.02:
        np.random.seed(2)
    elif sigma == 0.03:
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
def origin_and_destination(action, blocks_data_pre, blocks_data_suc):

    # take the sent disk name
        
    disk_name = action.variables[0].name

    pre_peg = ''
    suc_peg = ''

    # look where it is in the pre lists
    for index, dico in enumerate(blocks_data_pre.values()):
        for dic_val in dico:
            if disk_name in str(dic_val):
                pre_peg = "peg"+str(index+1)

    # look where it is in the suc list
    for index, dico in enumerate(blocks_data_suc.values()):
        for dic_val in dico:
            if disk_name in str(dic_val):
                suc_peg = "peg"+str(index+1)

    # return the two "where"
    return pre_peg, suc_peg


# 1401
nb_samplings_per_starting_state = 501 # has to be ODD 


def make_black_and_white(image, threshold=127):
    # Convert the image to black and white
    black_and_white = np.where(image > threshold, 255, 0)
    return black_and_white

def generate_dataset():

    ## 1) Loading the env
    env = pddlgym.make("PDDLEnvBlocks3-v0", dynamic_action_space=True)

    all_traces = []
    unique_obs = []
    unique_obs_img = []
    unique_transitions = [] # each ele holds a str representing a unique pair of blocks_data
    obs_occurences = {}
    counter = 0

    # looping over the number of starting positions
    # for blocksworld only the 1st pb has 4 blocks
    for ii in range(0, 7, 1):

        print("ici")

        last_two_blocks_data_str = [] # must contain only two lists that represent a legal transition
        last_two_blocks_data = []
        last_two_imgs = []
        trace_transitions = []

        # Initializing the first State
        obs, debug_info = env.reset(_problem_idx=ii)

        # Retrieve the 1st image
        img, blocks_data = env.render()

        img = img[:,:,:3] # remove the transparancy

        # gray_image = np.mean(img, axis=2)
        # threshold = 127  # This is a common default value, but you may need to adjust it
        # binary_image = np.where(gray_image > threshold, 255, 0)
        # img = np.stack([binary_image]*3, axis=-1)
    
        #print(img.shape)

        #print(img[:,:,0].shape)
        #print(np.where(img[:,:,0] > 0, 255, 0))

        # plt.imsave("BLACKreeeeeeee.png", np.where(img[:,:,0] > 0, 1, 0).astype(np.uint8))
        # plt.close()
        # exit()
        img = np.where(img[:,:,0] > 0, 1, 0)

        if str(blocks_data) not in unique_obs:
            unique_obs.append(str(blocks_data))
            obs_occurences[str(blocks_data)] = 1
            unique_obs_img.append(img)

        last_two_blocks_data_str.append(str(blocks_data))
        last_two_blocks_data.append(blocks_data)
        last_two_imgs.append(img)

        # looping over the nber of states to sample for each starting position
        for jjj in range(nb_samplings_per_starting_state):
            
            if counter%10 == 0:
                print("counter: {}".format(str(counter)))

            # sample an action
            action = env.action_space.sample(obs)

            # apply the action and retrieve img and obs
            obs, reward, done, debug_info = env.step(action)
            img, blocks_data = env.render()
            img = img[:,:,:3]

            img = np.where(img[:,:,0] > 0, 1, 0)
            # gray_image = np.mean(img, axis=2)
            # threshold = 127  # This is a common default value, but you may need to adjust it
            # binary_image = np.where(gray_image > threshold, 255, 0)
            # img = np.stack([binary_image]*3, axis=-1)

            last_two_blocks_data_str.append(str(blocks_data))
            last_two_blocks_data.append(blocks_data)
            if len(last_two_blocks_data_str) > 2:
                last_two_blocks_data_str.pop(0)
                last_two_blocks_data.pop(0)

            last_two_imgs.append(img)
            if len(last_two_imgs) > 2:
                last_two_imgs.pop(0)

            if len(last_two_blocks_data_str) == 2:

                if str(last_two_blocks_data_str) not in unique_transitions:

                    # i.e. loose, semi-loose-v1, semi-loose-v2


                    #transition_actions.append(str(action))
        
                    unique_transitions.append(str(last_two_blocks_data_str))
                    trace_transitions.append([[last_two_imgs[0], last_two_imgs[1]], str(last_two_blocks_data_str)])

            if str(blocks_data) not in unique_obs:
                unique_obs.append(str(blocks_data))
                obs_occurences[str(blocks_data)] = 1
                unique_obs_img.append(img)
            else:
                obs_occurences[str(blocks_data)] += 1

            counter += 1

        all_traces.append(trace_transitions)

    print("number of unique transitions is : {}".format(str(len(unique_transitions))))

    with open("resultatBlocks3.txt", 'w') as file2:

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



def save_noisy(dire, filename, images):
    data = {
        "images": images
    }
    if not os.path.exists(dire):
        os.makedirs(dire) 
    with open(dire+"/"+filename, mode="wb") as f:
        pickle.dump(data, f)

# # 1) generate dataset (only once normally)
# all_traces, obs_occurences, unique_obs_img, unique_transitions = generate_dataset()


# # 2) save dataset
# save_dataset("blocks_3_dataset", all_traces, obs_occurences, unique_obs_img, unique_transitions)


# exit()


def create_a_trace():
    # 3) load
    loaded_dataset = load_dataset("/workspace/pddlgym-tests/pddlgym/hanoi_4_4_dataset/data.p")

    # first loop to compute the whole dataset mean and std
    for iii, trace in enumerate(loaded_dataset["traces"]):
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

    global counntteerr


    loaded_dataset = load_dataset("/workspace/pddlgym-tests/pddlgym/blocks_3_dataset/data.p")

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

    unique_actions = []


    total_number_transitions = 0

    # first loop to compute the whole dataset mean and std
    for iii, trace in enumerate(loaded_dataset["traces"]):

        traces_indices.append([start_trace_index, start_trace_index+len(trace)*2])
        start_trace_index+=len(trace)*2

        traces_actions_indices.append([start_trace_action_index, start_trace_action_index+len(trace)])
        start_trace_action_index+=len(trace)
        
        actions_for_one_trace = []

        for trtrtr, transitio in enumerate(trace):

            total_number_transitions += 1

            all_images_reduced.append(reduce_resolution(transitio[0][0])) # = im1
            all_images_reduced.append(reduce_resolution(transitio[0][1])) # = im2


            # plt.imsave("hanoi_pair_"+str(trtrtr)+"_pre.png", reduce_resolution(transitio[0][0]))
            # plt.close()
            if transitio[1] not in unique_actions:

                # combined_image = np.concatenate((reduce_resolution(transitio[0][0]), reduce_resolution(transitio[0][1])), axis=1)
                # plt.imsave("block-transi"+str(len(unique_actions))+".png", combined_image)
                # print("action for transi "+str(len(unique_actions))+" is "+transitio[1])
                unique_actions.append(transitio[1])

            all_actions.append(transitio[1])
            actions_for_one_trace.append(transitio[1])

        all_actions_for_trace.append(actions_for_one_trace)

    unique_obs_img = loaded_dataset["unique_obs_img"]

    print("total_number_transitions : {}".format(str(total_number_transitions)))
    
    print("size unique_actions: {}".format(str(len(unique_actions))))


    reduced_uniques = []
    for uniq in unique_obs_img:

        # print(np.unique(reduce_resolution(uniq)))
        # plt.imsave("BLCOKS3pre.png", reduce_resolution(uniq))
        # plt.close()

        # exit()
        # reduced_uniques.append(np.clip(gaussian(reduce_resolution(uniq), 0.01), 0, 255))
        # reduced_uniques.append(np.clip(gaussian(reduce_resolution(uniq), 0.02), 0, 255))
        # reduced_uniques.append(np.clip(gaussian(reduce_resolution(uniq), 0.03), 0, 255))

        reduced_uniques.append(reduce_resolution(uniq))
        reduced_uniques.append(reduce_resolution(uniq))
        reduced_uniques.append(reduce_resolution(uniq))


    # plt.imsave("TTTTTTTTUNIIIIIIIKKK.png", reduced_uniques[0].astype(np.uint8))
    # plt.close()
    # exit()

    # exit()

    # counntteerr=0
    # noisy_uniques = add_noise(reduced_uniques, seed=1)
    # counntteerr=0
    # np.append(noisy_uniques, add_noise(reduced_uniques, seed=2))
    # counntteerr=0
    # np.append(noisy_uniques, add_noise(reduced_uniques, seed=3))

    # save_noisy("blocks_3_dataset", "uniques.p", noisy_uniques)

    # loaded_noisy = load_dataset("/workspace/pddlgym-tests/pddlgym/blocks_3_dataset/uniques.p")
    # noisy_uniques = loaded_noisy["images"]

    # plt.imsave("UNIIIIIIIKKK.png", noisy_uniques[0].astype(np.uint8))
    # plt.close()
    # exit()
    noisy_uniques = reduced_uniques

    unique_obs_img_preproc, orig_max, orig_min = preprocess(noisy_uniques)
    unique_obs_img_color_norm, mean_all, std_all = normalize_colors(unique_obs_img_preproc, mean=None, std=None)

    # for tr1 in unique_obs_img_color_norm:
    #     print(tr1.shape)
    #     print(np.max(tr1))
    #     print(np.min(tr1))
    #     exit()
    # print(unique_obs_img_color_norm)
    # exit()


    # counntteerr=0
    # all_images_reduced_gaussian_20 = add_noise(all_images_reduced, seed=1)
    # counntteerr=0
    # all_images_reduced_gaussian_30 = add_noise(all_images_reduced, seed=2)
    # counntteerr=0
    # all_images_reduced_gaussian_40 = add_noise(all_images_reduced, seed=3)

    # save_noisy("blocks_3_dataset", "all_images_seed1.p", all_images_reduced_gaussian_20)
    # save_noisy("blocks_3_dataset", "all_images_seed2.p", all_images_reduced_gaussian_30)
    # save_noisy("blocks_3_dataset", "all_images_seed3.p", all_images_reduced_gaussian_40)

    # #exit()

    # loaded_noisy1 = load_dataset("/workspace/pddlgym-tests/pddlgym/blocks_3_dataset/all_images_seed1.p")
    # all_images_seed1 = loaded_noisy1["images"]

    # loaded_noisy2 = load_dataset("/workspace/pddlgym-tests/pddlgym/blocks_3_dataset/all_images_seed2.p")
    # all_images_seed2 = loaded_noisy2["images"]

    # loaded_noisy3 = load_dataset("/workspace/pddlgym-tests/pddlgym/blocks_3_dataset/all_images_seed3.p")
    # all_images_seed3 = loaded_noisy3["images"]

    # for indd, imm in enumerate(all_images_seed2):        
    #     plt.imsave("ihhh"+str(indd)+".png", imm.astype(np.uint8))
    #     plt.close()
    #     if indd > 10:
    #         exit()
    # exit()

    all_images_reduced_gaussian_20_preproc, _, _ = preprocess(all_images_reduced)
    all_images_reduced_gaussian_20_norm, __, __ = normalize_colors(all_images_reduced_gaussian_20_preproc, mean=mean_all, std=std_all)
    
    all_images_reduced_gaussian_30_preproc, _, _ = preprocess(all_images_reduced)
    all_images_reduced_gaussian_30_norm, __, __ = normalize_colors(all_images_reduced_gaussian_30_preproc, mean=mean_all, std=std_all)

    all_images_reduced_gaussian_40_preproc, _, _ = preprocess(all_images_reduced)
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
    

    # print("lk")
    # print(np.max(train_set[0][0]))
    # exit()

    return train_set, test_val_set, all_pairs_of_images_reduced_orig, all_actions_one_hot, mean_all, std_all, all_actions_unique, orig_max, orig_min


#train_set, test_val_set, all_pairs_of_images_reduced_orig, all_actions_one_hot, mean_all, std_all, all_actions_unique, orig_max, orig_min = export_dataset()



# for hh in range(0, len(train_set), 10):

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


#     plt.imsave("blocks_3_pair_"+str(hh)+"_pre.png", im1)
#     plt.close()

#     plt.imsave("blocks_3_pair_"+str(hh)+"_suc.png", im2)
#     plt.close()


# exit()


