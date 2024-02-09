import pddlgym
import imageio
#from pddlgym_planners.fd import FD
import numpy as np
import matplotlib.pyplot as plt

# Convert to grayscale
def rgb_to_grayscale(rgb_images):
    print("IN rgb_to_grayscale")
    rgb_images = np.array(rgb_images)
    grayscale = np.dot(rgb_images[...,:3], [0.21, 0.72, 0.07])
    # Stack the grayscale values to create an (H, W, 3) image
    return np.stack((grayscale,)*3, axis=-1)

# turn the image(s) into 0-1 range (?)
def normalizee(image):
    # into 0-1 range
    if image.max() == image.min():
        return image - image.min()
    else:
        return (image - image.min())/(image.max() - image.min())

def equalize(image):
    from skimage import exposure
    return exposure.equalize_hist(image)

def enhance(image):
    return np.clip((image-0.5)*3,-0.5,0.5)+0.5

def preprocess(image):
    image = image.astype(float)
    image = equalize(image)
    image = normalizee(image)
    #image = enhance(image)
    return image


def normalize(x):
    #x = x.astype(np.float64) / 255
    #print(np.array(x).shape)
    mean               = np.mean(x, axis=0)
    std                = np.std(x, axis=0)
    #print("normalized shape:",mean.shape,std.shape)
    #print(mean.shape)
    return (x - mean)/(std+1e-6), mean, std

# x, mean and std must be np array
def unnormalize(x, mean, std):
    unnormalized_images = (x*std) + mean
    #unnormalized_images = np.clip(unnormalized_images, 0, 255)
    return unnormalized_images.astype(int)



nb_samplings_per_starting_state = 100

def export_dataset():

    ## 1) Loading the env
    env = pddlgym.make("PDDLEnvHanoi-v0", dynamic_action_space=True)
    obs, debug_info = env.reset(_problem_idx=1)

    all_images = []
    unique_transitions = []
    all_obs = []
    all_actions = []

    # looping over the number of Towers (for ToH)
    for ii in range(4):

        # Initializing the first State
        obs, debug_info = env.reset(_problem_idx=ii)

        # Retrieve the 1st image
        img = env.render()
        img = img[:,:,:3]

        # adding the img and obs to all_*
        all_images.append(img)
        all_obs.append(obs.literals)

        # looping over the nber of states to sample for each starting tower
        for jjj in range(nb_samplings_per_starting_state):

            # sample an action
            action = env.action_space.sample(obs)

            # add the actions to all_*
            all_actions.append(action)

            # apply the action and retrieve img and obs
            obs, reward, done, debug_info = env.step(action)
            img = env.render()
            img = img[:,:,:3]
            all_images.append(img)
            all_obs.append(obs.literals)

    from PIL import Image
    all_images = np.array(all_images)
    print("lool")
    # 17 42 25 2
    #print(all_images[15,15:35,10:20,1])



    #print(normalized[121][38][73][0])

    # exit()
    # # check the values of mean or std accross whole dataset
    # for lk in range(204):
    #     for jt in range(56):
    #         for tg in range(120):
    #             for tt in range(3):
    #                 ssss = unn[lk, jt, tg, tt] # check mean (replace by std if needed)
    #                 if int(ssss) != 0 and int(ssss) != 255 and int(ssss) != 51:
    #                     print(ssss) # 17 42 25 2
    #                     print("coords: ",str(lk), str(jt), str(tg), str(tt))

    
    
    ######### Greying out the images 
    all_images_grey = rgb_to_grayscale(all_images)

    ##### Preprocess the images
    all_images_prepro = preprocess(all_images)


    image = Image.fromarray(all_images_prepro[0]*255, 'RGB')
    image.save('NOR0.png')


    image = Image.fromarray(all_images_prepro[1]*255, 'RGB')
    image.save('NOR1.png')

    print(np.amin(all_images_prepro[0]))
    print(type(all_images_prepro[0]))
    
    #plt.imshow(all_images_prepro[0])
    #plt.savefig("NOR00000.png", format="png")
    plt.imsave("NOR000001.png", all_images_prepro[0])
    plt.close()

    plt.imsave("NOR000002.png", all_images_prepro[1])
    plt.close()


    exit()

    plt.imshow(all_images_prepro[1])
    plt.savefig("NOR11111.png")
    plt.close()

    ######### Normalizing the images (centered on 0, and with a std)
    all_images_final, mean_, std_ = normalize(all_images_prepro)

    normalized, mean_, std_ = normalize(all_images_grey)

    #all_images_final = all_images

    all_pairs_of_images = []
    all_pairs_of_images_norm = []
    all_pairs_of_obs = []

    # building the array of pairs
    for iiii, p in enumerate(all_images_final):
        if iiii%2 == 0:
            all_pairs_of_images.append([all_images_final[iiii], all_images_final[iiii+1]])
            all_pairs_of_images_norm.append([all_images_prepro[iiii], all_images_prepro[iiii+1]])
            all_pairs_of_obs.append([all_obs[iiii], all_obs[iiii+1]])

    print("uuu1111")
    print(len(all_pairs_of_images_norm))
    print()
    print(len(all_pairs_of_images))

    # building the array of UNIQUE transitions
    for ooo, obss in enumerate(all_pairs_of_obs):
        if str(obss) not in unique_transitions:
            unique_transitions.append(str(obss))
    
    all_actions_unique = []
    for uuu, act in enumerate(all_actions):
        if str(act) not in all_actions_unique:
            all_actions_unique.append(str(act))


    all_actions_indices = []

    for ac in all_actions:
        all_actions_indices.append(all_actions_unique.index(str(ac)))
    
    print(len(all_actions_indices))


    import torch
    import torch.nn.functional as F

    actions_indexes = torch.tensor(all_actions_indices)
    actions_one_hot = F.one_hot(actions_indexes, num_classes=len(all_actions_unique))

    print(len(all_actions_unique))
    
    #print(actions_one_hot.shape)
    print(actions_one_hot.numpy().shape)


    return all_pairs_of_images[:-2], all_pairs_of_images_norm[:-2], actions_one_hot.numpy(), mean_, std_, all_actions_unique

export_dataset()

