import tkinter as tk
import numpy as np
import sys
sys.path.append('./')

NUM_ROW_UNITS = 4
NUM_COLUMN_UNITS = 4
UNIT_SIZE = 40
NUM_BLOCKS = 2

class Maze_env(tk.Tk, object):
    def __init__(self):
        super(Maze_env, self).__init__()
        self.title('Maze')
        self.window_size = [NUM_COLUMN_UNITS * UNIT_SIZE, NUM_ROW_UNITS * UNIT_SIZE]
        self.geometry('%dx%d' % (self.window_size[0], self.window_size[1]))

        self.state_shape = [NUM_COLUMN_UNITS, NUM_ROW_UNITS]
        self.action_space = ['up', 'down', 'left', 'right']
        self.agent_pos, self.blocks_pos, self.treasures_pos = self.create_objs()
        self.curr_agent_pos = np.copy(self.agent_pos)

        self.agent_spr = tk.PhotoImage(file = './sprites/mario.png')
        self.block_spr = tk.PhotoImage(file = './sprites/block.png')
        self.treasure_spr = tk.PhotoImage(file = './sprites/coin.png')
        # self.agent_spr = tk.PhotoImage(file = './rl/2d/sprites/mario.png')
        # self.block_spr = tk.PhotoImage(file = './rl/2d/sprites/block.png')
        # self.treasure_spr = tk.PhotoImage(file = './rl/2d/sprites/coin.png')
        self.build_maze()

    def create_random_coord(self):
        # [x, y]
        return np.array([np.random.randint(NUM_COLUMN_UNITS), np.random.randint(NUM_ROW_UNITS)])

    def create_objs(self):
        # Create the treasure
        treasures = []
        new_treasure = self.create_random_coord()
        treasures.append(new_treasure)

        # Create the blocks
        create_blocks = 0
        blocks = []
        while create_blocks < NUM_BLOCKS:
            new_block = self.create_random_coord()
            duplicate = False
            # Remove duplicate positions
            for t in treasures:
                if (new_block == t).all():
                    duplicate = True
            for b in blocks:
                if (new_block == b).all():
                    duplicate = True

            if not duplicate:
                create_blocks += 1
                blocks.append(new_block)

        # Create the agent
        create_agent = 0
        while create_agent < 1:
            agent = self.create_random_coord()
            duplicate = False
            # Remove duplicate positions
            for t in treasures:
                if (agent == t).all():
                    duplicate = True
            for b in blocks:
                if (agent == b).all():
                    duplicate = True

            if not duplicate:
                create_agent += 1

        return agent, blocks, treasures

    def get_coord_in_canvas(self, coord):
        return np.array([UNIT_SIZE * (coord[0] + 0.5), UNIT_SIZE * (coord[1] + 0.5)])

    def build_maze(self):
        self.canvas = tk.Canvas(self, height = NUM_ROW_UNITS * UNIT_SIZE, width = NUM_COLUMN_UNITS * UNIT_SIZE, bg = 'white')

        # Draw lines
        for r in range(NUM_ROW_UNITS + 1):
            self.canvas.create_line(0, UNIT_SIZE * r, self.window_size[0], UNIT_SIZE * r)

        for c in range(NUM_COLUMN_UNITS + 1):
            self.canvas.create_line(UNIT_SIZE * c, 0, UNIT_SIZE * c, self.window_size[1])

        # Draw objects
        self.block_objs = []
        for b in self.blocks_pos:
            block_coord = self.get_coord_in_canvas(b)
            block_obj = self.canvas.create_image(block_coord[0], block_coord[1], anchor = 'center', image = self.block_spr)
            self.block_objs.append(block_obj)

        self.treasure_objs = []
        for t in self.treasures_pos:
            treasure_coord = self.get_coord_in_canvas(t)
            treasure_obj = self.canvas.create_image(treasure_coord[0], treasure_coord[1], anchor = 'center', image = self.treasure_spr)
            self.treasure_objs.append(treasure_obj)

        agent_coord = self.get_coord_in_canvas(self.agent_pos)
        self.agent_obj = self.canvas.create_image(agent_coord[0], agent_coord[1], anchor = 'center', image = self.agent_spr)

        self.canvas.pack()

    def state_transition(self, action):
        old_pos = np.copy(self.curr_agent_pos)
        terminate = False
        reward = 0.0

        # Move
        if action == 'up':
            if self.curr_agent_pos[1] > 0:
                self.curr_agent_pos[1] = self.curr_agent_pos[1] - 1
            else:
                reward = -0.1
        elif action == 'down':
            if self.curr_agent_pos[1] < NUM_ROW_UNITS - 1:
                self.curr_agent_pos[1] = self.curr_agent_pos[1] + 1
            else:
                reward = -0.1
        elif action == 'left':
            if self.curr_agent_pos[0] > 0:
                self.curr_agent_pos[0] = self.curr_agent_pos[0] - 1  
            else:
                reward = -0.1
        elif action == 'right':
            if self.curr_agent_pos[0] < NUM_COLUMN_UNITS - 1:
                self.curr_agent_pos[0] = self.curr_agent_pos[0] + 1  
            else:
                reward = -0.1
        else:
            raise ValueError('Unknown action!')
        # print("From: ", old_pos, ", Action: ", action, ", To: ", self.curr_agent_pos)

        # Whether the game is terminated
        for b in self.blocks_pos:
            if (self.curr_agent_pos == b).all():
                terminate = True
                reward = -1.0
                break
        
        if not terminate:
            for t in self.treasures_pos:
                if (self.curr_agent_pos == t).all():
                    terminate = True
                    reward = 1.0
                    break

        # Update the canvas
        coord_diff = (self.curr_agent_pos - old_pos) * UNIT_SIZE
        self.canvas.move(self.agent_obj, coord_diff[0], coord_diff[1])
        self.canvas.update()

        return self.curr_agent_pos, terminate, reward

    def reset(self, new_game = False):
        if new_game:
            # Delete all sprites
            self.canvas.delete(self.agent_obj)
            for b in self.block_objs:
                self.canvas.delete(b)
            for t in self.treasure_objs:
                self.canvas.delete(t)
            self.block_objs.clear()
            self.treasure_objs.clear()
            # Create new sprites
            self.agent_pos, self.blocks_pos, self.treasures_pos = self.create_objs()
            self.curr_agent_pos = np.copy(self.agent_pos)

            agent_coord = self.get_coord_in_canvas(self.agent_pos)
            self.agent_obj = self.canvas.create_image(agent_coord[0], agent_coord[1], anchor = 'center', image = self.agent_spr)

            for b in self.blocks_pos:
                block_coord = self.get_coord_in_canvas(b)
                self.block_objs.append(self.canvas.create_image(block_coord[0], block_coord[1], anchor = 'center', image = self.block_spr))
            for t in self.treasures_pos:
                treasure_coord = self.get_coord_in_canvas(t)
                self.treasure_objs.append(self.canvas.create_image(treasure_coord[0], treasure_coord[1], anchor = 'center', image = self.treasure_spr))
        else:
            self.canvas.delete(self.agent_obj)
            agent_coord = self.get_coord_in_canvas(self.agent_pos)
            self.agent_obj = self.canvas.create_image(agent_coord[0], agent_coord[1], anchor = 'center', image = self.agent_spr)
            self.curr_agent_pos = np.copy(self.agent_pos)

if __name__ == '__main__':
    env = Maze_env()
    # import time
    # time.sleep(2)
    # env.state_transition('up')
    # time.sleep(2)
    # env.state_transition('right')
    env.mainloop()