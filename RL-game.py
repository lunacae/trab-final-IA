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

def test_qlearning():
    # Training Model/Agent
    if TRAIN():
        env = MazeEnv()
        agent = QLearningAgent(env.action_space, ALPHA(), GAMMA(), EPSILON())

        episodes = SIMMULATION_NUMBER()
        total_reward = 0
        sucess = 0
        for episode in range(1, episodes+1):
            state, _ = env.reset(isnumpy = False)
            done = False
            if RENDERS():
                env.render()

            while not done:
                action = agent.get_action(state)
                next_state, reward, done, _, _ = env.step(action, isnumpy = False)

                agent.update(state, action, reward, next_state)
                state = next_state
                total_reward += reward
                if reward == GOAL_REWARD():
                    sucess += 1
                if RENDERS():
                    env.render()

            # Parameter Decay
            if episode % DECAY_STEP() == 0:
                agent.epsilon *= EPSILON_DECAY()
                agent.alpha *= ALPHA_DECAY()
                print(f"Episode {episode}, Mean Reward: {(total_reward/DECAY_STEP()):.2f}, Success Rate: {(sucess/DECAY_STEP()):.2f}")
                print("Explore Chance (epsilon): ", agent.epsilon)
                print("Exploit Chance (1-epsilon): ", 1-agent.epsilon)
                print("Learning Rate (alpha): ", agent.alpha)
                total_reward = 0
                sucess = 0

        # Save the trained agent
        print("Saving agent model")
        agent.save_model('q_learning_model.pkl')
    # Test the trained agent
    else:
        env = MazeEnv()
        state, _ = env.reset(isnumpy = False)
        done = False
        if RENDERS():
            env.render()
        # Load the trained agent
        agent = QLearningAgent(env.action_space, ALPHA(), GAMMA(), 0)
        print("Opening agent model")
        agent.load_model('q_learning_model.pkl')

        while not done:
            action = agent.get_action(state)
            state, reward, done, _, _ = env.step(action, isnumpy = False)
            print(f"Action: {action}, Reward: {reward}")
            if RENDERS():
                env.render()

test_qlearning()