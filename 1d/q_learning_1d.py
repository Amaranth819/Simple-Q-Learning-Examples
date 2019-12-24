# We have a Q table representing the potential reward of choosing an action at a state. Q(s, a)
# The estimated Q-value: Q_esti = Q(s_old, a)
# The real Q-value: Q_real = R + gamma*max(Q(s_new))
# Difference: diff = Q_real - Q_esti
# Then update the Q table by: Q(s_old, a) += alpha*diff
# Deciding the action is after the update.

# The game is in 1D space.
# W0---T
# Getting T gets reward 1 and hitting W gets reward -1.

import random
import numpy as np
import time
import matplotlib.pyplot as plt

num_states = 6 # States are 0 (wall), 1, 2, ... num_states - 1
actions = ['l', 'r']
epsilon = 0.5
gamma = 0.99
alpha = 0.1
epoch = 15
time_btw_actions = 0.3

def init_q_table():
    return np.zeros(shape = [num_states, len(actions)], dtype = float)

def state_transition(last_state, curr_action):
    terminate = False
    reward = 0

    if curr_action not in actions:
        raise ValueError('Unknown action!')

    if curr_action == 'l':
        if last_state > 0:
            last_state -= 1
            if last_state == 0:
                reward = -1
                terminate = True
    else:
        if last_state < num_states - 1:
            last_state += 1
            if last_state == num_states - 1:
                reward = 1
                terminate = True
    # The next state, terminate, reward
    return last_state, terminate, reward

def choose_action(curr_state, q_table):
    action = random.sample(actions, 1)[0] if random.random() > epsilon else actions[np.argmax(q_table[curr_state])]
    return action

def q_learning():
    # Initialize the Q table
    q_table = init_q_table()

    # Initialize plot
    plt.ion()

    for e in range(epoch):
        # Initialization
        terminate = False
        step = 0
        curr_state = random.randint(1, num_states - 2)

        while not terminate:
            # Render
            plt.cla()
            render(curr_state)
            plt.pause(0.2)

            # Randomly choose action
            action = choose_action(curr_state, q_table)
            next_state, terminate, reward = state_transition(curr_state, action)

            # Calculate Q_esti and Q_real
            q_esti = q_table[curr_state, actions.index(action)]
            q_real = reward if terminate else reward + gamma*np.max(q_table[next_state])

            # Update the Q table
            q_table[curr_state][actions.index(action)] += alpha * (q_real - q_esti)

            print("[Epoch %d|Step %d] -- Action: %s. State: from %d to %d. Terminate: %r" % (e, step, action, curr_state, next_state, terminate))
            step += 1
            curr_state = next_state
        
        # Print the Q table
        print("After epoch %d, the Q table is \n" % e)
        print(q_table)

    return q_table

def render(curr_state):
    plt.grid()
    plt.scatter(curr_state, 0, s = 200)
    plt.xlim([0, num_states - 1])
    plt.ylim([0, 1])

if __name__ == '__main__':
    plt.xticks(np.arange(num_states))
    plt.yticks([0, 1])
    
    q_learning()