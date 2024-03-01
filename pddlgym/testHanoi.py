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


## 1) Loading the env
env = pddlgym.make("PDDLEnvHanoi-v0", dynamic_action_space=True)
# obs, debug_info = env.reset(_problem_idx=0)


# Initializing the first State
obs, debug_info = env.reset(_problem_idx=0)


# Retrieve the 1st image
img, peg_to_disc_list = env.render()


print(peg_to_disc_list)