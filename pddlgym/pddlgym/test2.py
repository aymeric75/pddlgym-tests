import pddlgym
import imageio
#from pddlgym_planners.fd import FD
import numpy as np
import matplotlib.pyplot as plt


nb_samplings_per_starting_state = 201 # has to be ODD 


## 1) Loading the env
env = pddlgym.make("PDDLEnvHanoi-v0", dynamic_action_space=True)
# obs, debug_info = env.reset(_problem_idx=0)


unique_obs = []

counter = 0


# looping over the number of starting positions
# for blocksworld only the 1st pb has 4 blocks
for ii in range(1):


    # Initializing the first State
    obs, debug_info = env.reset(_problem_idx=ii)

    if str(obs.literals) not in unique_obs:
        unique_obs.append(str(obs.literals))



    # looping over the nber of states to sample for each starting position
    for jjj in range(nb_samplings_per_starting_state):

        
        print("counter: {}".format(str(counter)))


        # sample an action
        action = env.action_space.sample(obs)


        # apply the action and retrieve img and obs
        obs, reward, done, debug_info = env.step(action)


        if str(obs.literals) not in unique_obs:
            unique_obs.append(str(obs.literals))

        counter += 1


print("len unique_obs")
print(len(unique_obs))


