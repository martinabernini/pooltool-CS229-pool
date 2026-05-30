import math
import torch

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


def log_episode(i_episode, timesteps, rewards):
    msg = f'Episode {i_episode} finished after {timesteps} timesteps, total rewards {rewards}'
    print(msg)
    with open(LOG_PATH, "a") as f:
        f.write(msg + '\n')


def train(env, model_path, episodes=EPISODES, episode_length=EPISODE_LENGTH):
    print('DQN training')

    env.set_buckets(action=ACTION_BUCKETS)
    action_size = ACTION_BUCKETS[0] * ACTION_BUCKETS[1]
    agent = dqn_agent.Agent(env.state_space.n, action_size, seed=SEED)

    best_reward = float('-inf')

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
                log_episode(i_episode, t + 1, rewards)
                break

        if not done:
            log_episode(i_episode, episode_length, rewards)

        # Save on best reward or every SAVE_EVERY episodes
        if rewards > best_reward:
            best_reward = rewards
            save_model(model_path, agent)
            print(f'  -> New best reward {best_reward:.2f}, model saved.')
        elif (i_episode + 1) % SAVE_EVERY == 0:
            save_model(model_path, agent)
            print(f'  -> Checkpoint at episode {i_episode + 1}, model saved.')
