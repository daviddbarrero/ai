
import gym
import sys

def run_gym_env(arg):
    env = gym.make(arg[1])
    env.reset()
    for _ in range(int(arg[2])):
        env.render()
        env.step(env.action_space.sample())

    env.close()


if __name__ == "__main__":
    run_gym_env(sys.argv)
