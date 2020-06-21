
import gym

env = gym.make("MountainCar-v0")

MaxEp = 1000

for episode in range(MaxEp):
    done = False
    obs = env.reset()
    total_reward = 0.0
    step = 0
    while not done:
        env.render()
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        step += 1
        obs = next_state
    print("\n Episode num {} ended with {} iterations. Reward final={}".format(episode, step+1, total_reward))
env.close()


# Q-LEARNER CLASS

# INIT  ( self env )
# DISCREZE
# GET_ACTION
# Learn(self , obs, action, reward, next_obs)

# EPIPSILON_MIN : LEARN WITH THE RATIO OF LEARNING WROTH
# MAX_NUM_EPISODES : ITERACTIONS TO DO
# STEPS_PER_EPISODE : NUMS STEP IN EPISODES

# ALPHA : LEARNING RATIO
# GAMMA : DISCOUNT FOR THE AGENT IN EACH STEP

# NUM_DISCRETE_BINS: número de divisiones en el caso de discretizar el espacio de estados continuo.
import numpy as np

MAX_NUM_EPISODES = 50000
STEPS_PER_EPISODE = 200
EPSILON_MIN = 0.005
max_num_steps = MAX_NUM_EPISODES * STEPS_PER_EPISODE
EPSILON_DECAY = 500 * EPSILON_MIN / max_num_steps
ALPHA = 0.05
GAMMA = 0.98
NUM_DISCRETE_BINS = 30

class QLearner(object):
    def __init__(self,env):

        self.obs_shape = env.observation_space.shape
        self.obs_high = env.observation_space.high
        self.obs_low = env.observation_space.low
        self.obs_bins = NUM_DISCRETE_BINS
        self.obs_width  = ( self.obs_high- self.obs_low)/ self.obs_bins

        self.action_shape = env.action_space.n
        self.Q = np.zeros((self.obs_bins+1, self.obs_bins+1, self.action_shape)) # matriz of 31x31x3

        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = 1.0

       def discretize(self, obs):
        return tuple(((obs-self.obs_low)/self.bin_width).astype(int))

    def get_action(self, obs):
        discrete_obs = self.discretize(obs)
        # Selección de la acción en base a Epsilon-Greedy
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY
        if np.random.random() > self.epsilon: #Con probabilidad 1-epsilon, elegimos la mejor posible
            return np.argmax(self.Q[discrete_obs])
        else:
            return np.random.choice([a for a in range(self.action_shape)])#Con probabilidad epsilon, elegimos una al azar


    def learn(self, obs, action, reward, next_obs):
        discrete_obs = self.discretize(obs)
        discrete_next_obs = self.discretize(next_obs)
        self.Q[discrete_obs][action] += self.alpha*(reward + self.gamma * np.max(self.Q[discrete_next_obs]) - self.Q[discrete_obs][action])



### Method to train agent


def train(agent, env):
    best_reward = -float('inf')
    for episodes in range(MAX_NUM_EPISODES):
        done = False
        obs = env.reset()
        total_reward = 0.0
        while not done:
            action = agent.get_action(obs) # Eleccion pick by the class
            next_obs, reward ,done, info = env.step(action)
            agent.learn(obs, action, reward, next_obs)
            obs = next_obs
            total_reward += reward
        if total_reward > best_reward:
            best_reward = total_reward
        print("Episode  num: {} With a reward : {} , More reward {} , Epsilon {}".format(episode, total_reward , best_reward , agent.epsilon))

        ## Get the best

        return np.argmax(agent.Q , axis=2)


def test(agent, env, policy):
    done = False
    obs = env.reset()
    total_reward = 0.0
    while not done:
        action = policy[agent.discretize(obs)] # Action that dictemine the policy that we have train
        next_obs, reward, done, info = env.step(action)
        obs = next_obs
        total_reward += reward

    return total_reward



if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    agent = QLearner(env)
    learn_policy = train(agent, env)
    monitor_path = "./monitor_output"
    env = gym.wrappers.Monitor(env, monitor_path, force= True)
    for _ in range(1000):
        test(agent, env , learned_policy)

    env.close()


