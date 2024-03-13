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


ref_colors = [
    [ 0 ,  0, 255],
    [  0 ,255 ,0],
    [  0 ,255 ,  0],
    [128  ,128 ,128],
    [255, 0, 0],
    [255 ,255  , 0],
    [128, 255, 255]]


shades = [[128, 0, 255],
    [128, 255, 0],
    [0, 128, 0],
    [255, 128, 128],
    [128, 0, 0],
     [128, 255, 0],
     [128, 128, 255]]


def convert_to_lab_and_to_color_wt_min_distance(rgb, boolean_matrix):

    global counntteerr

    if counntteerr % 100000 == 0:
        print("counter is {}".format(str(counntteerr)))

    if counntteerr == len(boolean_matrix):
        return rgb

    #if counntteerr < len(boolean_matrix):
    if boolean_matrix[counntteerr]:



        if (rgb == np.array(ref_colors[0])).all():
            counntteerr+=1
            return np.array(shades[0])

        elif (rgb == np.array(ref_colors[1])).all():
            counntteerr+=1
            return np.array(shades[1])

        elif (rgb == np.array(ref_colors[2])).all():
            counntteerr+=1
            return np.array(shades[2])

        elif (rgb == np.array(ref_colors[3])).all():
            counntteerr+=1
            return np.array(shades[3])

        elif (rgb == np.array(ref_colors[4])).all():
            counntteerr+=1
            return np.array(shades[4])

        elif (rgb == np.array(ref_colors[5])).all():
            counntteerr+=1
            return np.array(shades[5])

        elif (rgb == np.array(ref_colors[6])).all():
            counntteerr+=1
            return np.array(shades[6])




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

    # Reshape the array to have each row represent a pixel's color
    pixels = images[0].reshape(-1, 3)
    # Find unique color combinations
    unique_colors = np.unique(pixels, axis=0)
    print("luu")
    print(unique_colors)
    #exit()

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

def equalize(image, histo_bins=None):
    return exposure.equalize_hist(image)

def enhance(image):
    return np.clip((image-0.5)*3,-0.5,0.5)+0.5

def preprocess(image, histo_bins=256):

    if not isinstance(image,np.ndarray):
        image = np.array(image)

    # compte le nombre de couleurs
    # pour chacune, tu donne une version modifiée (voir l'autre fct là)
    # pour le nbre de bins, bah c'est le nbre de couleurs


    image = image / 255. # put the image to between 0 and 1 (with the assumption that vals are 0-255)
    image = image.astype(float)
    #print(np.unique(image))
    image = equalize(image, histo_bins=14)
    #print(np.unique(image))
    image, orig_max, orig_min = normalize(image) # put the image back to btween 0 and 1
    #print(np.unique(image))
    image = enhance(image) # i) from btween 0-1 values, center to 0 (so become btween -0.5 +0.5)
    #print(np.unique(image))
    #exit()

    return image, orig_max, orig_min


def gaussian(a, sigma=0.3):
    if sigma == 5:
        np.random.seed(1)
    elif sigma == 7:
        np.random.seed(2)
    elif sigma == 10:
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



def normalize_colors(images, mean=None, std=None, second=False):    

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

                if str(last_two_peg_to_disc_lists_str) not in unique_transitions and str(last_two_peg_to_disc_lists[0]) != str(last_two_peg_to_disc_lists[1]):

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

        file2.write(str(len(unique_transitions)) + '\n')

    return all_traces, obs_occurences, unique_obs_img, unique_transitions



# construct pairs (of images)
# and construct the array of action (for each pair of images) 
# Return: pairs of images, corresponding actions (one-hot)

def modify_one_trace(all_images_transfo_tr, all_images_orig_tr, all_actions, all_actions_unique_):

    #print(type(all_images_transfo_tr))
    # building the array of pairs
    all_pairs_of_images = []
    all_pairs_of_images_orig = []
    for iiii, p in enumerate(all_images_transfo_tr):
        if iiii%2 == 0:


            # mask = all_images_transfo_tr[iiii+1] > 1000
            # result = all_images_transfo_tr[iiii+1][mask]
            # if len(result) > 0:
            #     print(result)
            #     exit()

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
# save_dataset("hanoi_4_4_dataset", all_traces, obs_occurences, unique_obs_img, unique_transitions)

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

    total_number_unique_transitions = 0

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


            if str(transitio[1][3]) not in all_actions_unique:
                all_actions_unique.append(str(transitio[1][3]))

                # if len(all_actions_unique) < 3000:

                #     combined_image = np.concatenate((reduce_resolution(transitio[0][0]), reduce_resolution(transitio[0][1])), axis=1)
                #     plt.imsave("alltransi/hanoi-4-4-transi"+str(len(all_actions_unique))+".png", combined_image)
                
        all_actions_for_trace.append(actions_for_one_trace)

    unique_obs_img = loaded_dataset["unique_obs_img"]




    reduced_uniques = []
    for uniq in unique_obs_img:

        # reduced_uniques.append(np.clip(gaussian(reduce_resolution(uniq), 0.1), 0, 255))
        # reduced_uniques.append(np.clip(gaussian(reduce_resolution(uniq), 0.2), 0, 255))
        # reduced_uniques.append(np.clip(gaussian(reduce_resolution(uniq), 0.3), 0, 255))


        reduced_uniques.append(reduce_resolution(uniq))
        reduced_uniques.append(reduce_resolution(uniq))
        reduced_uniques.append(reduce_resolution(uniq))

    print("ii1")

    #######

    #####
    #####  0 128 256

    print(np.unique(reduced_uniques[0]))
    print(reduced_uniques[0].shape)

    # for i in range(3):
    #     for j in range(25):
    #         for f in range(70):
    #             if reduced_uniques[0][j][f][i] != 255 and reduced_uniques[0][j][f][i] != 0 and reduced_uniques[0][j][f][i] != 128:
    #                 print(reduced_uniques[0][j][f][i])

    # counntteerr=0
    # noisy_uniques = add_noise(reduced_uniques, seed=1)
    # counntteerr=0
    # np.append(noisy_uniques, add_noise(reduced_uniques, seed=2))
    # counntteerr=0
    # np.append(noisy_uniques, add_noise(reduced_uniques, seed=3))

    # save_noisy("hanoi_4_4_dataset", "uniques.p", noisy_uniques)


    # loaded_noisy = load_dataset("/workspace/pddlgym-tests/pddlgym/hanoi_4_4_dataset/uniques.p")
    # noisy_uniques = loaded_noisy["images"]
    # plt.imsave("HHHH_pre.png", iiimages[0])
    # plt.close()

    noisy_uniques = reduced_uniques

    unique_obs_img_preproc, orig_max, orig_min = preprocess(noisy_uniques, histo_bins=24)
    unique_obs_img_color_norm, mean_all, std_all = normalize_colors(unique_obs_img_preproc, mean=None, std=None)



    print("ii2")
    # all_images_reduced > gaussian > clip > preprocess > normalize_colors

   
    # counntteerr=0
    # all_images_reduced_gaussian_20 = add_noise(all_images_reduced, seed=1)
    # counntteerr=0
    # all_images_reduced_gaussian_30 = add_noise(all_images_reduced, seed=2)
    # counntteerr=0
    # all_images_reduced_gaussian_40 = add_noise(all_images_reduced, seed=3)

    # save_noisy("hanoi_4_4_dataset", "all_images_seed1.p", all_images_reduced_gaussian_20)
    # save_noisy("hanoi_4_4_dataset", "all_images_seed2.p", all_images_reduced_gaussian_30)
    # save_noisy("hanoi_4_4_dataset", "all_images_seed3.p", all_images_reduced_gaussian_40)


    # loaded_noisy1 = load_dataset("/workspace/pddlgym-tests/pddlgym/hanoi_4_4_dataset/all_images_seed1.p")
    # all_images_seed1 = loaded_noisy1["images"]

    # loaded_noisy2 = load_dataset("/workspace/pddlgym-tests/pddlgym/hanoi_4_4_dataset/all_images_seed2.p")
    # all_images_seed2 = loaded_noisy2["images"]

    # loaded_noisy3 = load_dataset("/workspace/pddlgym-tests/pddlgym/hanoi_4_4_dataset/all_images_seed3.p")
    # all_images_seed3 = loaded_noisy3["images"]




    all_images_reduced_gaussian_20_preproc, _, _ = preprocess(all_images_reduced, histo_bins=256)
    all_images_reduced_gaussian_20_norm, __, __ = normalize_colors(all_images_reduced_gaussian_20_preproc, mean=mean_all, std=std_all, second=True)


    all_images_reduced_gaussian_30_preproc, _, _ = preprocess(all_images_reduced, histo_bins=256)
    all_images_reduced_gaussian_30_norm, __, __ = normalize_colors(all_images_reduced_gaussian_30_preproc, mean=mean_all, std=std_all)

    all_images_reduced_gaussian_40_preproc, _, _ = preprocess(all_images_reduced, histo_bins=256)
    all_images_reduced_gaussian_40_norm, __, __ = normalize_colors(all_images_reduced_gaussian_40_preproc, mean=mean_all, std=std_all)



    # for tr1 in all_images_reduced_gaussian_40_norm:
    #     print(tr1.shape)
    #     print(np.max(tr1))
    #     print(np.min(tr1))
    #     exit()
    # print(unique_obs_img_color_norm)
    # exit()



    # print("ii3")
    # for im in all_images_reduced_gaussian_30_norm:
    #     mask = im[0] > 1000
    #     result = im[0][mask]
    #     if len(result) > 0:
    #         print(result)

    # exit()


    #build array containing all actions WITHOUT DUPLICATE
    for uuu, act in enumerate(all_actions):
        if str(act) not in all_actions_unique:
            all_actions_unique.append(str(act))


    all_pairs_of_images_processed_gaussian20 = []
    all_pairs_of_images_processed_gaussian30 = []
    all_pairs_of_images_processed_gaussian40 = []


    # second loop to process the pairs
    for iiii, trace in enumerate(loaded_dataset["traces"]):

        print("ii4")
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


#train_set, test_val_set, all_pairs_of_images_reduced_orig, all_actions_one_hot, mean_all, std_all, all_actions_unique, orig_max, orig_min = export_dataset()


# #print(train_set[0][0][0])

# print(len(train_set)) # 2904



# for hh in range(0, len(train_set), 500):

#     acc = all_actions_unique[np.argmax(train_set[hh][1])]
#     print("action for {} is {}".format(str(hh), str(acc)))

#     # im1_orig=all_pairs_of_images_reduced_orig[hh][0]
#     # im2_orig=all_pairs_of_images_reduced_orig[hh][1]
#     im1 = train_set[hh][0][0]
#     im2 = train_set[hh][0][1]

#     im1 = unnormalize_colors(im1, mean_all, std_all)
#     im1 = deenhance(im1)
#     print(im1)

#     plt.imsave("hanoi_4-4-pair_EEEEEEEE"+str(hh)+"_pre.png", im1)
#     plt.close()

#     exit()
#     im1 = deenhance(im1)
#     im1 = denormalize(im1, orig_min, orig_max)
#     im1 = np.clip(im1, 0, 1)

    
#     im2 = unnormalize_colors(im2, mean_all, std_all)
#     im2 = deenhance(im2)
#     im2 = denormalize(im2, orig_min, orig_max)
#     im2 = np.clip(im2, 0, 1)


#     plt.imsave("hanoi_4-4-pair_"+str(hh)+"_pre.png", im1)
#     plt.close()

#     plt.imsave("hanoi_4-4-pair_"+str(hh)+"_suc.png", im2)
#     plt.close()


# exit()


