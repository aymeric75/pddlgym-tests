import pddlgym
import imageio
import numpy as np
import matplotlib.pyplot as plt

nb_samplings_per_starting_state = 201

env = pddlgym.make("PDDLEnvHanoi-v0", dynamic_action_space=True, seed=1)

env.seed(1)

unique_obs = []

for ii in range(1):

    # Initializing the first State
    obs, debug_info = env.reset(_problem_idx=0)

    img, peg_to_disc_list = env.render()

    if str(peg_to_disc_list) not in unique_obs:
        unique_obs.append(str(peg_to_disc_list))


    # looping over the nber of states to sample for each starting position
    for jjj in range(nb_samplings_per_starting_state):
        
        print("counter: {}".format(str(jjj)))

        # sample an action
        action = env.action_space.sample(obs)

        # apply the action and retrieve obs
        obs, reward, done, debug_info = env.step(action)

        img, peg_to_disc_list = env.render()

        if str(peg_to_disc_list) not in unique_obs:
            unique_obs.append(str(peg_to_disc_list))


print("len unique_obs")
print(len(unique_obs))


