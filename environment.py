
import gym
import numpy as np


class SampleEnv(gym.Env):
    '''generate synthetic contextual bandit environment'''

    def __init__(self, params={'dim_s': 100, 'low_s': -1, 'high_s': 1,
                               'dim_a': 100, 'low_a': -1, 'high_a': 1, 'num_a': 100,
                               'dim_l': 8, 'arch_r': [64], 'seed': 2022}):
        super().__init__()
        self.__dict__.update(params)
        self.set_random_seed()
        self.restart()

    def set_random_seed(self):
        '''fix random seed for reproducibility'''
        self.rng = np.random.default_rng(seed=self.seed)
        self.seed_s = self.rng.integers(1e+09)
        self.seed_a = self.rng.integers(1e+09)
        self.seed_r = self.rng.integers(1e+09)

    def restart(self):
        '''setup the environment'''
        self.setup_state_space()
        self.setup_action_space()
        self.setup_reward_function()

    def setup_state_space(self):
        '''generate state space'''
        self.rng_s = np.random.default_rng(seed=self.seed_s)
        self.observation_space = gym.spaces.Box(
            low=self.low_s, high=self.high_s, shape=(self.dim_s,), dtype=np.float32)

    def setup_action_space(self):
        '''generate action space'''
        self.rng_a = np.random.default_rng(seed=self.seed_a)
        self.action_space = gym.spaces.Discrete(self.num_a)
        self.actions = self.rng_a.uniform(self.low_a, self.high_a, (self.num_a,self.dim_a)).astype(np.float32)

    def setup_reward_function(self):
        '''generate reward function'''
        self.rng_r = np.random.default_rng(seed=self.seed_r)
        self.generate_state_embedding()
        self.generate_action_embedding()
        self.A_emb = self.feature_a(self.actions)
        self.A_emb_norm = np.linalg.norm(self.A_emb, axis=1, keepdims=True)

    def generate_state_embedding(self):
        '''generate state feature map'''
        params_s_emb = {'dim_in': self.dim_s, 'dim_layers': self.arch_r, 'dim_out': self.dim_l,
                        'seed': self.rng_r.integers(1e+09)}
        self.feature_s = lambda s: SyntheticGaussianMapping(params_s_emb).propagate(s)

    def generate_action_embedding(self):
        '''generate action feature map'''
        self.params_a_emb = {'dim_in': self.dim_a, 'dim_layers': self.arch_r, 'dim_out': self.dim_l,
                             'seed': self.rng_r.integers(1e+09)}
        self.feature_a = lambda a: SyntheticGaussianMapping(self.params_a_emb).propagate(a)

    def compute_reward(self, s, a_ind):
        '''compute the normalized reward value for a given state and an action index'''
        s_emb = self.feature_s(s)
        s_emb_norm = np.linalg.norm(s_emb, axis=1, keepdims=True)
        R = np.matmul(s_emb, self.A_emb.T) / np.matmul(s_emb_norm, self.A_emb_norm.T)
        r = (R[np.arange(s.shape[0]),a_ind] - R.mean(axis=1)) / (R.max(axis=1) - R.mean(axis=1))
        return r

    def observe(self, num=1):
        '''sample observed states'''
        self.state = self.rng_s.uniform(self.low_s, self.high_s, (num,self.dim_s))
        return self.state.astype(np.float32)

    def reset(self):
        '''observe a new state'''
        state = self.observe().flatten()
        return state

    def step(self, action_index):
        '''given an observed state take an action and receive reward'''
        reward = self.compute_reward(self.state, action_index).item()
        done = True
        info = {}
        return self.state, reward, done, info

    def compute_reward_raw(self, s, a):
        '''compute the reward value for a given state-action pair'''
        s_emb = self.feature_s(s)
        a_emb = self.feature_a(a)
        s_emb_norm = np.linalg.norm(s_emb, axis=1, keepdims=True)
        a_emb_norm = np.linalg.norm(a_emb, axis=1, keepdims=True)
        r = np.matmul(s_emb, a_emb.T) / np.matmul(s_emb_norm, a_emb_norm.T)
        return r

    def print_action_histogram(self, num_s=100000):
        '''compute the average/minimum/maximum reward values and optimal actions'''
        self.restart()
        S = self.observe(num=num_s)
        R = self.compute_reward_raw(S, self.actions)
        hist = np.histogram(R.argmax(axis=1), bins=np.arange(self.num_a+1), density=True)[0]
        print(f'optimal action distribution:\n{-np.sort(-hist)}')


class SyntheticGaussianMapping:
    '''generate synthetic feature extractor'''

    def __init__(self, params):
        self.__dict__.update(params)
        self.activation = lambda z: np.exp(-z**2)
        self.rng = np.random.default_rng(seed=self.seed)
        self.initialize_weights()

    def initialize_weights(self):
        '''initialize the network from the normal distribution'''
        self.dims = [self.dim_in, *self.dim_layers, self.dim_out]
        self.num_layers = len(self.dims) - 1
        self.weights = {}
        for l in range(self.num_layers):
            self.weights[l] = self.rng.normal(scale=1., size=(self.dims[l]+1,self.dims[l+1]))

    def propagate(self, x):
        '''propagate input through the network'''
        z = np.array(x, ndmin=2)
        for l in range(self.num_layers):
            z = np.concatenate([np.ones((z.shape[0],1)), z], axis=1)
            if l < self.num_layers - 1:
                z = self.activation(np.matmul(z, self.weights[l]))
            else:
                z = np.tanh(np.matmul(z, self.weights[l]))
        return z

