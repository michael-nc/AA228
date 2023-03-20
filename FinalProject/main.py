import os
import time
import json
import random

import numpy as np
import torch
import gymnasium as gym

import DQN

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def scale_image(image):
    image_gray = image[:-12, 6:-6, :]
    image_gray = image_gray.astype(np.float32) / 255
    return np.moveaxis(image_gray, -1, 0)


def train(lr, gamma, target_replace_count, epsilon_decrease, experience_size=10000, batch_size=32, device="cpu", seed=0, total_step=1000, total_episode=-1, model="DQN", base="./"):

    env = gym.make("CarRacing-v2", continuous=False)
    set_seed(seed)
    episode_rewards = []
    
    agent = DQN.DQNAgent(input_channel=3, input_size=(84, 84), action_size=env.action_space.n, lr=lr, gamma=gamma, batch_size=batch_size, experience_size=experience_size, target_replace_count=target_replace_count, epsilon=1, epsilon_final=0.01, epsilon_decrease=epsilon_decrease, device=device)
    
    episode = 0
    start_episode = time.perf_counter()
    start_total = time.perf_counter()

    print(f"Start training {model} with lr: {lr}, gamma: {gamma}, replace_count: {target_replace_count}, epsilon_dec: {epsilon_decrease}")

    while True:
        if total_episode < 0:
            if (len(episode_rewards) > 100 and np.mean(episode_rewards[-100:]) >= 50):
                break
        else:
            if (episode > total_episode):
                break

        done = False 
        cumulative_reward = 0
        observation, info = env.reset(seed=seed)
        observation = scale_image(observation)

        for _ in range(total_step):
            action = agent.get_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_observation = scale_image(next_observation)

            cumulative_reward += reward

            done = True if (terminated or truncated) else False

            # agent store experience and learn
            agent.store_experience(action=action, state=observation, next_state=next_observation, reward=reward, done=done)
            agent.learn()

            observation = next_observation

            if done:
                break

        episode_rewards.append(cumulative_reward)

        if episode % 50 == 0:
            average_reward = np.array(episode_rewards[-50:]).mean()
            now = time.perf_counter()
            print(f"episode: {episode}, average reward: {average_reward}, epsilon: {agent.epsilon}, took: {round(now - start_episode, 3)}, in_total: {round(now - start_total, 3)}")
            start_episode = now

        episode += 1

    # print for last episode
    average_reward = np.mean(episode_rewards[-50:])
    now = time.perf_counter()
    print(f"episode: {episode}, average reward: {average_reward}, epsilon: {agent.epsilon}, took: {round(now - start_episode, 3)}, in_total: {round(now - start_total, 3)}")

    # agent.save_model(base)
    env.close()

    return episode_rewards, now - start_total

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_episode = 500
    lr = 0.01
    gamma = 0.99
    target_replace_count = 400
    epsilon_decrease = 2.5e-06
    episode_reward, total_time = train(lr=lr, gamma=gamma, target_replace_count=target_replace_count, epsilon_decrease=epsilon_decrease, device=device, total_episode=total_episode)
    
    print(f"Training DQN for {total_episode} episodes took {round(total_time, 3)} secs")
    with open("DQN_reward.npy", "wb") as f:
        np.save(f, np.array(episode_reward))