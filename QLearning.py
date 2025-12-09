import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import pickle
from config import STEP_REWARD, WALL_REWARD, HOLE_REWARD, GOAL_REWARD
from MazeEnv import MazeEnv
from config import ALPHA, GAMMA, EPSILON, SIMMULATION_NUMBER, ALPHA_DECAY, EPSILON_DECAY, DECAY_STEP, TRAIN, RENDERS
from stable_baselines3 import DQN

class QLearningAgent:
    def __init__(self, action_space, learning_rate, discount_factor, epsilon):
        self.q_table = defaultdict(lambda: np.zeros(action_space.n))
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.action_space = action_space

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = defaultdict(lambda: np.zeros(self.action_space.n), pickle.load(f))

    def get_action(self, state):
        if random.random() < self.epsilon:
            return self.action_space.sample()

        q_values = self.q_table[state]
        exp_q = np.exp(q_values - np.max(q_values))
        probs = exp_q / np.sum(exp_q)
        return np.random.choice(len(q_values), p=probs)

    def update(self, state, action, reward, next_state):
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state][action] = new_value
