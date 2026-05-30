import math
import os
import torch
import pickle
import numpy as np

from . import dqn_agent

# --- Constants ---
EPISODES = 1500
EPISODE_LENGTH = 50
SEED = 229
ACTION_BUCKETS = [360, 5]       # angle buckets, force buckets
GAMMA = 0.8                     # reward discount factor
EPSILON_SCHEDULE_FACTOR = 100   # larger = slower epsilon decay
LOG_PATH = "output/dqn-log.txt"
SAVE_EVERY = 50                 # save model every N episodes (also saves on best reward)

# Epsilon decay schedule: starts at 1.0, decays slowly, floors at 0.01
get_epsilon = lambda i: max(0.01, min(1.0, 1.0 - math.log10((i + 1) / EPSILON_SCHEDULE_FACTOR)))


def save_model(filepath, model):
    torch.save(model.qnetwork_local.state_dict(), filepath)


def load_model(modelpath, model_params):
    state_size = model_params['s_dim']
    action_buckets = model_params['buckets']
    action_size = action_buckets[0] * action_buckets[1]

    agent = dqn_agent.Agent(state_size, action_size, seed=SEED)
    agent.qnetwork_local.load_state_dict(torch.load(modelpath))
    return agent


def action_to_tuple(action, action_buckets):
    return (float(int(action) % action_buckets[0]), int(action / action_buckets[0]))


def choose_action(state, model, action_space, epsilon=0.):
    action = action_to_tuple(model.act(state, epsilon), action_space.buckets)
    print(f'action was {action}')
    return action

def train(env, model_path, episodes=200, episode_length=50):
    print('DQN training')

    # Initialize DQN Agent
    state_size = env.state_space.n
    action_buckets = [360, 1]
    env.set_buckets(action=action_buckets)
    action_size = action_buckets[0] * action_buckets[1]

    agent = dqn_agent.Agent(state_size, action_size, seed = 229)

    # Learning related constants; factors should be determined by trial-and-error
    get_epsilon = lambda i: max(0.01, min(1, 1.0 - math.log10((i+1)/25))) # epsilon-greedy, factor to explore randomly; discounted over time
    get_lr = lambda i: max(0.01, min(0.5, 1.0 - math.log10((i+1)/25))) # learning rate; discounted over time
    gamma = 0.8 # reward discount factor

    # Q-learning
    for i_episode in range(episodes):
        epsilon = get_epsilon(i_episode)
        state = env.reset()
        rewards = 0
        done = False

        for t in range(episode_length):
            action = agent.act(state, epsilon)
            next_state, reward, done = env.step(action_to_tuple(action, ACTION_BUCKETS))
            rewards += reward
            agent.step(state, action, reward, next_state, done)
            state = next_state

            if done:
                print('Episode {} finished after {} timesteps, total rewards {}'.format(i_episode, t+1, rewards))
                with open("output\\dqn-log.txt", "a") as myfile:
                    myfile.write('Episode {} finished after {} timesteps, total rewards {}\n'.format(i_episode, t+1, rewards))
                break

        if not done:
            print('Episode {} finished after {} timesteps, total rewards {}'.format(i_episode, episode_length, rewards))
            with open("output\\dqn-log.txt", "a") as myfile:
                myfile.write('Episode {} finished after {} timesteps, total rewards {}\n'.format(i_episode, episode_length, rewards))

        save_model(model_path, agent)
        #print(agent.qnetwork_local.state_dict())
