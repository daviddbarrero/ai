import gym
from gym.spaces import *
import sys

def print_spaces(space):
    print(space)
    if isinstance(space, Box):# Check if is type box
        print("\n Cota inferior: ", space.low)
        print("\n Cota superior: ", space.high)


if __name__ == "__main__":
    environment = gym.make(sys.argv[1])
    print("Espacio de estados:")
    print_spaces(environment.observation_space)
    print("Espacio de acciones: ")
    print_spaces(environment.action_space)
    try:
        print("Descripción de las acciones: ", environment.unwrapped.get_action_meanings())
    except AttributeError:
        pass






