import pddlgym
import imageio
#from pddlgym_planners.fd import FD
import matplotlib
env = pddlgym.make("PDDLEnvHanoi-v0", dynamic_action_space=True)
print(env)
obs, debug_info = env.reset(_problem_idx=1)


action = env.action_space.sample(obs)

exit()

all_images = []

unique_transitions = []

all_obs = []

#env.seed(15)
for ii in range(4):

    print("new starting state")

    print("ii : {}".format(str(ii)))
    obs, debug_info = env.reset(_problem_idx=ii)

    img = env.render()
    # imageio.imsave("frame_"+str(ii)+"_0.png", img)
    # matplotlib.pyplot.close()

    all_images.append(img)
    all_obs.append(obs.literals)

    tmp_pair_img = [img]

    for jjj in range(3000):

            
        action = env.action_space.sample(obs)

        #print(obs.literals)
        print(action)
        #print(env.action_space)

        obs, reward, done, debug_info = env.step(action)
        img = env.render()
        all_images.append(img)
        all_obs.append(obs.literals)

        
        # imageio.imsave("frame_"+str(ii)+"_"+str(jjj+1)+".png", img)
        # matplotlib.pyplot.close()


# 
#        

all_pairs_of_images = []
all_pairs_of_obs = []

for iiii, p in enumerate(all_images):

    # imageio.imsave("p"+str(iiii)+"_0.png", p)
    # matplotlib.pyplot.close()   

    if iiii%2 == 0:
        all_pairs_of_images.append([all_images[iiii], all_images[iiii+1]])
        all_pairs_of_obs.append([all_obs[iiii], all_obs[iiii+1]])
        # all_pairs_of_obs

for ooo, obss in enumerate(all_pairs_of_obs):
        
    if str(obss) not in unique_transitions:
        unique_transitions.append(str(obss))
    

# for jjjj, pair in enumerate(all_pairs_of_images):

#     imageio.imsave("pair"+str(jjjj)+"_0.png", pair[0])
#     matplotlib.pyplot.close()
#     imageio.imsave("pair"+str(jjjj)+"_1.png", pair[1])
#     matplotlib.pyplot.close()


print(len(all_pairs_of_images))
print(len(all_images))
print(len(unique_transitions))
exit()

########
########        bon, tu fais les paires, PUIS ? à chaque paire, tu associe
########


###################
######## 
########     pour chaque depart, tu sample des states prends 100 steps, 
##########################          tu store les pairs (avec un tmp_pair)
#######################################   (et tu store les hash )





# # See `pddl/sokoban.pddl` and `pddl/sokoban/problem3.pddl`.
# #env = pddlgym.make("PDDLEnvSokoban-v0")
# env = pddlgym.make("PDDLEnvHanoi_operator_actions-v0")
# env.fix_problem_index(2)
# obs, debug_info = env.reset()
# planner = FD()
# plan = planner(env.domain, obs)
# for i, act in enumerate(plan):
#     print("Obs:", obs)
#     print("Act:", act)
#     obs, reward, done, info = env.step(act)
#     img = env.render()
#     imageio.imsave("frame"+str(i)+".png", img)
# print("Final obs, reward, done:", obs, reward, done)