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



def normalize_colors(images, mean=None, std=None):    
    if mean is None or std is None:
        mean      = np.mean(images, axis=0)
        std       = np.std(images, axis=0)
        print("was heree")
        print(len(images))
        print(mean.shape)
        print(std.shape)

    return (images - mean)/(std+1e-20), mean, std



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


# e.g. of a move move(d3:default,peg2:default)
# if last move involved same disk as present move
# AND if present move peg <=> former peg (... need to now former peg)
# peg_to_disc_list_precedente[str(peg1)+":default"]
# si peg_to_disc_list_prec["peg2:default"] (where "peg2:default" est la 2e partie de l'action présente)
# CONTIENT disque en question
def is_going_back(action_before, action_now, peg_to_disc_list_prec):

    
    if action_before is None:
        return False


    # {peg1:default: [d4:default, d3:default, d2:default], peg2:default: [], peg3:default: [d1:default], peg4:default: []}

    # pb si le disc qu'on déplace (action_now) vers peg_now ETAIT DEJA SITUE dans sur le même PEG 
    # au state précédent (à voir dans peg_to_disc_list_prec) 


    
    #peg_to_disc_list_prec_json = json.loads(peg_to_disc_list_prec)
    # AND the disk (disk_action_now) was in the peg we are now targetting
    for keyy, valss in peg_to_disc_list_prec.items():
        print(keyy.name) # peg du passé
        # si le peg vers lequel on déplace <=> au peg parcouru
        if keyy.name == action_now.variables[1].name:
            # et si le disk envoyer est dans ce peg ancien parcouru
            for dd in valss:
                if action_now.variables[0].name == dd.name:
                    return True



    return False


nb_samplings_per_starting_state = 301 # has to be ODD 

# 64
# 70

def generate_dataset():

    ## 1) Loading the env
    env = pddlgym.make("PDDLEnvHanoi-v0", dynamic_action_space=True)
    # obs, debug_info = env.reset(_problem_idx=0)

    all_traces = []
    unique_obs = []

    obs_occurences = {}
    
    counter = 0

    # looping over the number of starting positions
    # for blocksworld only the 1st pb has 4 blocks
    for ii in range(8):

        all_images_of_a_trace = []
        all_obs_of_a_trace = []
        all_actions_of_a_trace = []
        all_layouts_of_a_trace = []


        # Initializing the first State
        obs, debug_info = env.reset(_problem_idx=ii)

        # Retrieve the 1st image
        img, peg_to_disc_list = env.render()
        img = img[:,:,:3] # remove the transparancy

        if str(peg_to_disc_list) not in unique_obs:
            unique_obs.append(str(peg_to_disc_list))
            obs_occurences[str(peg_to_disc_list)] = 1

        # adding the img and obs to all_*
        all_images_of_a_trace.append(img)
        all_layouts_of_a_trace.append(peg_to_disc_list)
        all_obs_of_a_trace.append(obs.literals)


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

            if str(peg_to_disc_list) not in unique_obs:
                unique_obs.append(str(peg_to_disc_list))
                obs_occurences[str(peg_to_disc_list)] = 1
            else:
                obs_occurences[str(peg_to_disc_list)] += 1

            # plt.imsave("hanoi_"+str(jjj)+".png", img)
            # plt.close()
            # if jjj > 10:
            #     exit()
            # 
            # mmmmh problème... non ?

            all_images_of_a_trace.append(img)
            all_actions_of_a_trace.append(action)
            all_obs_of_a_trace.append(obs.literals)
            all_layouts_of_a_trace.append(peg_to_disc_list)
            

            counter += 1

        all_traces.append([all_images_of_a_trace, all_actions_of_a_trace, all_obs_of_a_trace, all_layouts_of_a_trace])
    

    # print("len unique_obs")
    # print(len(unique_obs))



    return all_traces, obs_occurences




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


# # # 1) generate dataset (only once normally)
# all_traces, obs_occurences = generate_dataset()


# print("len obs_occurences")
# print(len(obs_occurences))

# [all_images_of_a_trace, all_actions_of_a_trace, all_obs_of_a_trace, all_layouts_of_a_trace]


### Nettoyage du dataset
### 

## for each tr, accumulate the already seen obs

# for tr in all_traces:

#     print("icietla")
#     print(len(tr[0]))
#     print(len(tr[-1]))
#     print(tr[-1])
#     exit()

#     # parcours des layouts
#     # for lay in tr[-1]:
        
#     #     for l in lay:
#     #         print(l)
#     #     print(len(lay))

#     #     exit()



# # 2) save dataset
# save_dataset("hanoi_dataset", all_traces)

# exit()


# where we load the dataset, and adapt it to our needs
def export_dataset():

    # 3) load
    loaded_dataset = load_dataset("/workspace/pddlgym-tests/pddlgym/hanoi_dataset/data.p")

    all_images = []
    all_actions = []
    all_pairs_of_images = []
    all_pairs_of_images_orig = []
    all_actions_one_hot = []
    all_actions_unique = []

    traces_indices = []

    # first loop to compute the whole dataset mean and std
    for iii, trace in enumerate(loaded_dataset["traces"]):

        

        traces_indices.append([iii*len(trace[0]), (iii+1)*len(trace[0])])

        reduced_images = [reduce_resolution(img) for img in trace[0]]
        all_images.extend(reduced_images)
      
        #all_actions.extend(trace[1]) # simplest version

        # at the same time, construct the augmented actions total array (to construct the "unique" array below)
        # concatenate both the simple action array and the layout desc of the pre state array
        paired_gen = (str(a) + str(b) for a, b in zip(trace[1], trace[-1][:-1]))
        all_actions.extend(list(paired_gen))


    all_images_preproc, orig_max, orig_min = preprocess(all_images)


    # filtered_values = all_images_preproc[(all_images_preproc != 1.0) & (all_images_preproc != 0.0)]
    # print(filtered_values)

    # values will be centered on 0
    all_images_color_norm, mean_all, std_all = normalize_colors(all_images_preproc, mean=None, std=None)


    # def preprocess(image):
    #     image = np.array(image)
    #     image = image.astype(float)
    #     image = equalize(image)
    #     image, orig_max, orig_min = normalize(image)
    #     image = enhance(image)
    # de = unnormalize_colors(all_images_color_norm, mean_all, std_all)
    # de = deenhance(de)
    # de = denormalize(de, orig_max, orig_min)

    # plt.imsave("hanoi_deNorm.png", de[0])
    # plt.close()

    # filtered_values = all_images_color_norm[(all_images_color_norm != 1.0) & (all_images_color_norm != 0.0)]
    # print(filtered_values)

    # orig_max, orig_min

    # exit()

    #build array containing all actions WITHOUT DUPLICATE
    for uuu, act in enumerate(all_actions):
        if str(act) not in all_actions_unique:
            all_actions_unique.append(str(act))


    # second loop to construct the pairs
    for iiii, trace in enumerate(loaded_dataset["traces"]):


        # all_images_transfo_tr
        # all_images_orig_tr 

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

        
        all_pairs_of_images.extend(all_pairs_of_images_of_trace)
        all_pairs_of_images_orig.extend(all_pairs_of_images_orig_of_trace)
        all_actions_one_hot.extend(actions_one_hot_of_trace)




    return all_pairs_of_images, all_pairs_of_images_orig, all_actions_one_hot, mean_all, std_all, all_actions_unique



# all_pairs_of_images, all_pairs_of_images_orig, all_actions_one_hot, mean_all, std_all, all_actions_unique = export_dataset()

# exit()


# for hh in range(0, 20, 5):

#     acc = all_actions_unique[np.argmax(all_actions_one_hot[hh])]
#     print("action for {} is {}".format(str(hh), str(acc)))

#     im1_orig=all_pairs_of_images_orig[hh][0]
#     im2_orig=all_pairs_of_images_orig[hh][1]

#     plt.imsave("hanoi_pair_"+str(hh)+"_pre.png", im1_orig)
#     plt.close()

#     plt.imsave("hanoi_pair_"+str(hh)+"_suc.png", im2_orig)
#     plt.close()

#     #
