import time

import numpy as np
import pandas as pd

def Q_learning(data: str, gamma: float, num_actions: int, num_states: int, learning_rate: float=0.95, epochs: int=0):

    df = pd.read_csv(f"./data/{data}.csv")
    
    Q_table = np.zeros((num_states, num_actions)) 

    for epoch in range(epochs):
        
        print(f"Starting epoch {epoch} for {data} dataset!")

        if epoch >0:
            df = df.sample(frac=1, random_state=epoch)

        for index, row in df.iterrows():

            current_state = row["s"] -1
            action = row["a"] -1
            reward = row["r"]
            next_state = row["sp"] -1
            
            Q_table[current_state, action] += \
                learning_rate * (reward + gamma*np.max(Q_table[next_state, :]) - Q_table[current_state, action])

    optimal_action = np.argmax(Q_table, axis=1) + 1
    np.savetxt(f"./output/{data}.policy", optimal_action.reshape((-1, 1)), fmt="%s")

    return Q_table, optimal_action

def get_position_velocity(linear_indx):
    positions, velocities = np.meshgrid(np.arange(500), np.arange(100))
    pos, vel = positions.flatten(), velocities.flatten()
    linear_map = 1+pos+500*vel  

    idx = np.where(linear_map==(linear_indx))[0]
    return pos[idx].item(), vel[idx].item()

def gradient_Q_learning(data: str, gamma: float, num_actions: int, num_states: int, learning_rate: float=0.05):

    df = pd.read_csv(f"./data/{data}.csv")
    optimal_actions = []

    theta = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=float)

    for epoch in range(2):
        
        if epoch >0:
            df = df.sample(frac=1, random_state=epoch)

        for index, row in df.iterrows():

            current_state = row["s"] 
            action = row["a"]
            reward = row["r"]
            next_state = row["sp"]
            pos, vel = get_position_velocity(current_state)
            next_pos, next_vel = get_position_velocity(next_state)


            beta = np.array([pos, vel, pos*vel, action, 1])

            u = max([np.dot(theta, np.array([next_pos, next_vel, next_pos*next_vel, action_p, 1])) for action_p in range(1, num_actions+1)])
            gradient = (reward + gamma * u - np.dot(beta, theta))*beta
            scaled_gradient = min(1/np.linalg.norm(gradient), 1)*gradient
            theta += learning_rate*scaled_gradient

    for state in range(1, num_states):
        pos, vel = get_position_velocity(state)
        optimal_action = np.argmax([np.dot(theta, np.array([pos, vel, pos*vel, action_p, 1])) for action_p in range(1, num_actions+1)]) + 1
        optimal_actions.append(optimal_action)

    optimal_actions = np.array(optimal_actions, dtype=int)
    np.savetxt(f"./output/{data}.policy", optimal_actions.reshape((-1, 1)), fmt="%s")





if __name__ == "__main__":
    small = "small"
    small_gamma = 0.95
    small_num_actions = 4
    small_num_states = 100
    epochs = 2

    print(f"Working on small dataset!")
    start_time = time.perf_counter()
    Q_star = Q_learning(small, small_gamma, small_num_actions, small_num_states, epochs=epochs)
    elapsed_time = time.perf_counter() - start_time
    print(f"Small dataset took {elapsed_time:.2f} seconds to process using {epochs} epochs with an average time of {elapsed_time/epochs:.2f} seconds per epoch")


    medium = "medium"
    medium_gamma = 1
    medium_num_actions = 7
    medium_num_states = 50000
    epochs = 3

    print(f"Working on medium dataset!")
    start_time = time.perf_counter()
    Q_star = Q_learning(medium, medium_gamma, medium_num_actions, medium_num_states, epochs=epochs)
    elapsed_time = time.perf_counter() - start_time
    print(f"medium dataset took {elapsed_time:.2f} seconds to process using {epochs} epochs with an average time of {elapsed_time/epochs:.2f} seconds per epoch")

    large = "large"
    large_gamma = 0.95
    large_num_actions = 9
    large_num_states = 312020
    epochs = 3

    print(f"Working on large dataset!")
    start_time = time.perf_counter()
    Q_star = Q_learning(large, large_gamma, large_num_actions, large_num_states, epochs=epochs)
    elapsed_time = time.perf_counter() - start_time
    print(f"Large dataset took {elapsed_time:.2f} seconds to process using {epochs} epochs with an average time of {elapsed_time/epochs:.2f} seconds per epoch")
