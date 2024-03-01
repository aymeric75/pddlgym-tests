import pddlgym
from pddlgym_planners.fd import FD
import matplotlib.pyplot as plt
import numpy as np



# Function to reduce resolution of a single image using np.take
def reduce_resolution(image):
    # Use np.take to select every 4th element in the first two dimensions
    reduced_image = np.take(np.take(image, np.arange(0, image.shape[0], 9), axis=0),
                            np.arange(0, image.shape[1], 9), axis=1)
    return reduced_image


# See `pddl/sokoban.pddl` and `pddl/sokoban/problem3.pddl`.
env = pddlgym.make("PDDLEnvHanoi44-v0", dynamic_action_space=True)

print(env)

obs, debug_info = env.reset(_problem_idx=0)
img, peg_to_disc_list = env.render()

print(peg_to_disc_list)



plt.imsave("hanoi_lol_0.png", reduce_resolution(img))
plt.close()


planner = FD()
plan = planner(env.domain, obs)

for i, act in enumerate(plan):

    print("Obs:", obs)
    print("Act:", act)
    obs, reward, done, debug_info = env.step(act)

    img, peg_to_disc_list = env.render()
    img = img[:,:,:3]

    

    plt.imsave("hanoi_lol_"+str(i+1)+".png", reduce_resolution(img))
    plt.close()

print("Final obs, reward, done:", obs, reward, done)
