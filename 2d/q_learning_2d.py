import numpy as np
import time
from maze_env import Maze_env

def init_q_table(env_shape, num_actions):
    action_shape = [num_actions] + list(env_shape)
    return np.zeros(action_shape, dtype = float)

def choose_action(curr_state, action_space, q_table, epsilon):
    '''
        Apply epsilon-greedy policy
    '''
    # Randomly choose the action
    if np.random.rand() > epsilon:
        action = action_space[np.random.randint(len(action_space))]
    # Use greedy policy and always choose the action of the maximum q value.
    else:
        action = action_space[np.argmax(q_table[:][curr_state[0]][curr_state[1]])]

    return action

def start():
    # Configuration
    env = Maze_env()
    action_space = env.action_space
    q_table = init_q_table(env.state_shape, len(env.action_space))
    epsilon = 0.5
    gamma = 0.95
    learning_rate = 0.1
    epoch = 10

    # Start training
    for e in range(epoch):
        terminate = False
        steps = 0
        curr_state = env.curr_agent_pos

        while not terminate:
            time.sleep(0.5)

            # Choose action
            action = choose_action(curr_state, action_space, q_table, epsilon)

            # Update the environment
            next_state, terminate, reward = env.state_transition(action)

            # Calculate q value
            q_esti = q_table[action_space.index(action)][curr_state[0]][curr_state[1]]
            q_real = reward if terminate else reward + gamma * np.max(q_table[:][next_state[0]][next_state[1]])

            # Update the q table
            q_table[action_space.index(action)][curr_state[0]][curr_state[1]] += learning_rate * (q_real - q_esti)

            print("[Epoch %d|Step %d] Action: %s" % (e, steps, action), ", From ", curr_state, " To ", next_state)
            steps += 1
            curr_state = np.copy(next_state)

        if terminate:
            if reward == 1:
                print("[Epoch %d] Win the game! The reward is %d." % (e, reward))
            else:
                print("[Epoch %d] Lose the game! The reward is %d." % (e, reward))
            env.reset(True)

    env.mainloop()

if __name__ == '__main__':
    start()