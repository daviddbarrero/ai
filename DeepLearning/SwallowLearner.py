import numpy as np
import torch
from libs.percepton import SLP



MAX_NUM_EPISODES = 50000
STEPS_PER_EPISODE = 200
max_num_steps = MAX_NUM_EPISODES * STEPS_PER_EPISODE
ALPHA = 0.05
GAMMA = 0.98
NUM_DISCRETE_BINS = 30

class QLearner(object):
    def __init__(self, environment):
        self.obs_shape = environment.observation_space.shape
        self.obs_high = environment.observation_space.high
        self.obs_low = environment.observation_space.low
        self.obs_bins = NUM_DISCRETE_BINS
        self.bin_width = (self.obs_high-self.obs_low)/self.obs_bins

        self.action_shape = environment.action_space.n
        self.Q = SLP(self.obs_shape, self.action_shape)
        self.Q_optimizer = torch.optim.Adam(Self.Q.parameters(), lr = 1e-5)
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon_max = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = LinearDecaySchedule(initial_value = self.epsilon_max,
                                                 final_value = self.epsilon_min, 
        self.step_num = 0
        self.policy = self.epsilon_greedy_Q
        
        self.memory = ExperienceMemory(capacity = int(1e5))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                                         max_steps = 0.5 * MAX_NUM_EPISODES * STEPS_PER_EPISODE)

    def discretize(self, obs):
        return tuple(((obs-self.obs_low)/self.bin_width).astype(int))

    def get_action(self, obs):
        discrete_obs = self.discretize(obs)
        # Selección de la acción en base a Epsilon-Greedy
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY
        if np.random.random() > self.epsilon: #Con probabilidad 1-epsilon, elegimos la mejor posible
            return np.argmax(self.Q(discrete_obs.data.to(torch.device('cpu'))).numpy)

        else:
            return np.random.choice([a for a in range(self.action_shape)])#Con probabilidad epsilon, elegimos una al azar


    def learn(self, obs, action, reward, next_obs):
        td_target = reward + self.gamma * torch.max(self.Q(next_obs))
        td_error = torch.nn.functional.mse_loss(self.Q(obs)[action], td_target)
        self.Q_optimizer.zero_grad()
        td_error.backward()
        self.Q_optimizer.step()
