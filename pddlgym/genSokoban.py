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



# def unnormalize_colors(normalized_images, mean, std): 
#     # Reverse the normalization process
#     unnormalized_images = normalized_images * (std + 1e-6) + mean
#     return np.round(unnormalized_images).astype(np.uint8)


def unnormalize_colors(normalized_images, mean, std): 

    return (normalized_images*std)+mean



# Preprocessing (as recommended by chatpgt, also the order is recommended by chatgpt)

# 0) rescalling pixels value from 0-255 to 0-1 (NO NEED, step 1 aleady rescale)
# 1) reducing the resolution of the images 
# 2) normalizing the colors on each color channel (by substracing the mean and dividing by std)


def is_going_back(action_before, action_now):

    if action_before is None:
        return False

    action_before = str(action_before)
    action_now = str(action_now)

    if "left" in action_before and "right" in action_now:
        return True
    
    elif "right" in action_before and "left" in action_now:
        return True

    elif "down" in action_before and "up" in action_now:
        return True

    elif "up" in action_before and "down" in action_now:
        return True

    return False



def stone_is_stuck(obs):

    for ll in obs.literals:

        if str(ll) == "at(stone-01:thing,pos-2-2:location)":
            return True
        elif str(ll) == "at(stone-01:thing,pos-4-2:location)":
            return True
        elif str(ll) == "at(stone-01:thing,pos-2-4:location)":
            return True
        elif str(ll) == "at(stone-01:thing,pos-4-4:location)":
            return True
    return False

nb_samplings_max_after_reset = 20 # has to be ODD 

total_nb_steps = 700


def generate_dataset():

    ## 1) Loading the env
    env = pddlgym.make("PDDLEnvSokoban-v0", dynamic_action_space=True)
    # obs, debug_info = env.reset(_problem_idx=0)

    all_traces = []
    unique_obs = []
    obs_occurences = {}

    steps_done = 0
    loops_done = 0

    # pour 4 images, 3 actions
    # pour N images, N-1 actions
    
    iiiii=0

    # looping over the number of starting positions
    # for blocksworld only the 1st pb has 4 blocks
    while steps_done < total_nb_steps:

        all_images_of_a_trace = []
        all_obs_of_a_trace = []
        all_actions_of_a_trace = []
        all_layouts_of_a_trace = []
        action_before = None

        # Initializing the first State
        obs, debug_info = env.reset(_problem_idx=iiiii)

        iiiii = loops_done%4
        loops_done+=1

        # Retrieve the 1st image
        img, layout = env.render()
        img = img[:,:,:3] # remove the transparancy


        if str(layout) not in unique_obs:
            unique_obs.append(str(layout))
            obs_occurences[str(layout)] = 1

        # plt.imsave("sokoban_111111.png", img)
        # plt.close()


        # adding the img and obs to all_*
        all_images_of_a_trace.append(img)
        all_layouts_of_a_trace.append(layout)
        all_obs_of_a_trace.append(obs.literals)

        nb_times_stone_stuck = 0

        # looping over the nber of states to sample for each starting position
        for jjj in range(nb_samplings_max_after_reset):

            start_time = time.time()

            # sample an action
            action = env.action_space.sample(obs)

            while(is_going_back(action_before, action)):
                action = env.action_space.sample(obs)

            # # Capture the end time
            # inter_time = time.time()
            # # Calculate the duration by subtracting the start time from the end time
            # duration = inter_time - start_time
            # print(f"The inter time 0 is {duration} seconds.")

            # add the actions to all_*
            

            # apply the action and retrieve img and obs
            obs, reward, done, debug_info = env.step(action)
            steps_done += 1


            # if stone stuck
            if stone_is_stuck(obs):
                nb_times_stone_stuck += 1
            

            # # Capture the end time
            # inter_time = time.time()
            # # Calculate the duration by subtracting the start time from the end time
            # duration = inter_time - start_time
            # print(f"The inter time 1 is {duration} seconds.")


            img, layout = env.render()
            img = img[:,:,:3]



            # # Capture the end time
            # inter_time = time.time()
            # # Calculate the duration by subtracting the start time from the end time
            # duration = inter_time - start_time
            # print(f"The inter time 2 is {duration} seconds.")


            # plt.imsave("sokoban_"+str(steps_done)+".png", img)
            # plt.close()


            if str(layout) not in unique_obs:
                unique_obs.append(str(layout))
                obs_occurences[str(layout)] = 1
            else:
                obs_occurences[str(layout)] += 1

            all_actions_of_a_trace.append(action)
            all_images_of_a_trace.append(img)
            all_layouts_of_a_trace.append(layout)
            all_obs_of_a_trace.append(obs.literals)

            if steps_done%20==0:
                print("{} steps done".format(str(steps_done)))

            # # Capture the end time
            # end_time = time.time()
            # # Calculate the duration by subtracting the start time from the end time
            # duration = end_time - start_time
            # print(f"The code block took {duration} seconds to execute.")

            action_before = action


            if nb_times_stone_stuck > 3 or done:
                break

        all_traces.append([all_images_of_a_trace, all_actions_of_a_trace, all_obs_of_a_trace, all_layouts_of_a_trace])
    
    #print(obs_occurences)
    print(len(obs_occurences))

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

    # plt.imsave("soko_pair0_1.png", all_pairs_of_images[1][1])
    # plt.close()

    #build array containing actions indices of the dataset
    all_actions_indices = []
    for ac in all_actions:
        all_actions_indices.append(all_actions_unique_.index(str(ac)))
    
    print("all accc")
    print(len(all_actions))

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
# all_traces = generate_dataset()


# #2) save dataset
# save_dataset("sokoban_dataset", all_traces)

# exit()

def export_dataset():

    # 3) load
    loaded_dataset = load_dataset("/workspace/pddlgym-tests/pddlgym/sokoban_dataset/data.p")

    # all_images_of_a_trace, all_actions_of_a_trace, all_obs_of_a_trace, all_layouts_of_a_trace

    # 4) modify the dataset

    # modify_one_trace

    all_images = []
    all_actions = []
    all_pairs_of_images = []
    all_pairs_of_images_orig = []
    all_actions_one_hot = []
    all_actions_unique = []
    traces_indices = []


    # first loop to compute the whole dataset mean and std
    for iii, trace in enumerate(loaded_dataset["traces"]):
        # all_images[:, ::4, ::4, :]
        traces_indices.append([iii*len(trace[0]), (iii+1)*len(trace[0])])
       


        print(np.array(trace[0]).shape) # (19, 262, 262, 3)

        all_images.extend(np.array(trace[0])[:, ::4, ::4, :])

        all_actions.extend(trace[1])

    all_images = np.array(all_images) / 255


    #all_images_preproc, orig_max, orig_min = preprocess(all_images)

    all_images_color_norm, mean_all, std_all = normalize_colors(all_images, mean=None, std=None)
    # print("onestla")
    # print(all_images_color_norm[0])

    #filtered_values = all_images_color_norm[0][(all_images_color_norm[0] != 1.0) & (all_images_color_norm[0] != 0.0)]
    #print(filtered_values)

    #build array containing all actions (no duplicate)
    for uuu, act in enumerate(all_actions):
        if str(act) not in all_actions_unique:
            all_actions_unique.append(str(act))

    # second loop to construct the pairs
    for iiii, trace in enumerate(loaded_dataset["traces"]):

        all_images_transfo_tr = all_images_color_norm[traces_indices[iiii][0]:traces_indices[iiii][1]]
        all_images_orig_tr = all_images[traces_indices[iiii][0]:traces_indices[iiii][1]]

        all_images_tr = trace[0]
        all_actions_tr = trace[1]
        all_obs_tr = trace[2] 
        all_layouts_tr = trace[3]
        # all_images_of_a_trace, all_actions_of_a_trace, all_obs_of_a_trace, all_layouts_of_a_trace

        all_pairs_of_images_of_trace, all_pairs_of_images_orig_of_trace, actions_one_hot_of_trace = modify_one_trace(all_images_transfo_tr, all_images_orig_tr, all_actions_tr, all_layouts_tr, mean_all, std_all, all_actions_unique)


        all_pairs_of_images.extend(all_pairs_of_images_of_trace)
        all_pairs_of_images_orig.extend(all_pairs_of_images_orig_of_trace)
        all_actions_one_hot.extend(actions_one_hot_of_trace)




    return all_pairs_of_images, all_pairs_of_images_orig, all_actions_one_hot, mean_all, std_all, all_actions_unique





# all_pairs_of_images, all_pairs_of_images_orig, all_actions_one_hot, mean_all, std_all, all_actions_unique = export_dataset()


# for hh in range(0, 20, 5):

#     acc = all_actions_unique[np.argmax(all_actions_one_hot[hh])]
#     print("action for {} is {}".format(str(hh), str(acc)))

    
#     print(all_pairs_of_images[0])
#     # im1_orig = all_pairs_of_images_orig[hh][0]
#     # im2_orig = all_pairs_of_images_orig[hh][1]
#     unorma_images = unnormalize_colors(all_pairs_of_images, mean_all, std_all)
    
#     print("unorma_images[0]")
#     print(unorma_images[0])

#     im1_orig = unorma_images[hh][0]
#     im2_orig = unorma_images[hh][1]

#     # plt.imsave("hanoi_pair_"+str(hh)+"_pre.png", im1_orig)
#     # plt.close()

#     # plt.imsave("hanoi_pair_"+str(hh)+"_suc.png", im2_orig)
#     # plt.close()

#     plt.imsave("hanoi_pair_"+str(hh)+"_pre.png", im1_orig)
#     plt.close()

#     plt.imsave("hanoi_pair_"+str(hh)+"_suc.png", im2_orig)
#     plt.close()
