
import gym

env = gym.make("Qbert-v0")

maxNum = 10
maxStepsEpisode = 500

for episode in range(maxNum):
    obs = env.reset()
    for step in range(maxStepsEpisode):
        env.render()
        action = env.action_space.sample() # Random action
        next_state, reward, done, info = env.step(action)
        obs = next_state

        if done is True:
            print("\n Episode #{} finished in :  {} steps".format(episode, step+1))
            break
env.close()
