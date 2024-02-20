import pddlgym
import numpy as np
import matplotlib.pyplot as plt
import time


nb_max_of_samplings = 5501

# 
# 
# 

counter = 0

# a dict to show the number of occurences of each obs sampled
obs_occurences = {}

## Loading the env
env = pddlgym.make("PDDLEnvBlocks-v0", dynamic_action_space=True)



# Initializing the first State
obs, debug_info = env.reset(_problem_idx=0)

# we count the occurence of the initial obs
obs_occurences[str(obs)] = 1

print("debut")
img, blocks_data = env.render()
print(blocks_data)

img = img[:,:,:3]

plt.imsave("img_0.png", img)
plt.close()

exit()


def opposite_action(last_action, current_action):

    # unstack(c:block)

    # stack(c:block,b:block)


    if last_action.replace('pickup', '') == current_action.replace('putdown', ''):
        #print("opposite actions : {}, {}".format(str(last_action), str(current_action)))
        return True

    if last_action.replace('putdown', '') == current_action.replace('pickup', ''):
        #print("opposite actions : {}, {}".format(str(last_action), str(current_action)))
        return True

    if "unstack" in last_action and ("stack" in current_action and not "unstack" in current_action):
        if last_action.split(":")[0][-1] == current_action.split(":")[0][-1]:
            #print("opposite actions : {}, {}".format(str(last_action), str(current_action)))
            return True
    
    if "unstack" in current_action and ("stack" in last_action and not "unstack" in last_action):
        if last_action.split(":")[0][-1] == current_action.split(":")[0][-1]:
            #print("opposite actions : {}, {}".format(str(last_action), str(current_action)))
            return True


    return False


last_action = None

start_time = time.time()

while counter < nb_max_of_samplings:

    action = env.action_space.sample(obs)

    if last_action != None:
        ccc = 0
        while(opposite_action(str(last_action), str(action))):

            action = env.action_space.sample(obs)
            ccc += 1
            if ccc > 5:
                break


    obs, reward, done, debug_info = env.step(action)

    img, blocks_data = env.render()
    img = img[:,:,:3]

    # print("action for image "+str(counter)+": {}".format(str(action)))

    if str(blocks_data) not in obs_occurences.keys():
        obs_occurences[str(blocks_data)] = 1

        inter_time = time.time()
        duration = inter_time - start_time
        print("sampled a new state at minute: {}".format(str(duration/60)))
    else:
        obs_occurences[str(blocks_data)] += 1
        #if obs_occurences[str(blocks_data)] < 20:


    # if counter > 10:
    #     exit()

    last_action = action

    print("counter: {}".format(str(counter)))

    counter += 1

print("obs_occurences")
print(obs_occurences)
print(len(obs_occurences))



