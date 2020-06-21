import gym

env = gym.make("MountainCar-v0")
env.reset()

# print(env.observation_space.high)
# print(env.observation_space.low)
# print(env.action_space.h)


done = False

while not done:
    action = 2
    new_state, reward, done, _ = env.step(action)
    print()
    env.render()

env.close()
