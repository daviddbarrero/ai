
# The classic

import gym

env = gym.make("SpaceInvaders-v0")
env.reset() # Start agent Open eyes

for _ in range(2000):
    env.render()
    env.step(env.action_space.sample()) # Action and get result
    # Next state state --> Object next state of the position
    # Reward --> Float
    # Done ---> Boolean
    # Info --> Optional


env.close()
