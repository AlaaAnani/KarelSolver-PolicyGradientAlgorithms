import copy
import itertools
from enum import Enum
import random
import numpy as np
import gym
from gym import spaces
from json import load

from torch.utils.data import dataset


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class Action(Enum):
    move = 0
    turnLeft = 1
    turnRight = 2
    pickMarker = 3
    putMarker = 4
    finish = 5


actions = ['move', 'turnLeft', 'turnRight', 'pickMarker', 'putMarker', 'finish']
actions_dict = {'move': 0, 'turnLeft': 1, 'turnRight': 2, 'pickMarker': 3, 'putMarker': 4, 'finish': 5}

direction_dict = {
    "east": Direction.RIGHT.value,
    "west": Direction.LEFT.value,
    "north": Direction.UP.value,
    "south": Direction.DOWN.value
}

space_sizes = {
    "S0": 88,
    "S1": 104,
    "S2": 192
}
moves = {
    Direction.UP.value: (-1, 0),
    Direction.DOWN.value: (1, 0),
    Direction.LEFT.value: (0, -1),
    Direction.RIGHT.value: (0, 1),
}

turn_lefts = {
    Direction.UP.value: Direction.LEFT.value,
    Direction.DOWN.value: Direction.RIGHT.value,
    Direction.LEFT.value: Direction.DOWN.value,
    Direction.RIGHT.value: Direction.UP.value,
}

turn_rights = {
    Direction.UP.value: Direction.RIGHT.value,
    Direction.DOWN.value: Direction.LEFT.value,
    Direction.LEFT.value: Direction.UP.value,
    Direction.RIGHT.value: Direction.DOWN.value,
}


class Karel(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, paths, state_space="S1", sequential_loading=True, ):
        super(Karel, self).__init__()
        n_actions = 6
        self.paths = paths
        self.sequential_loading = sequential_loading
        self.current_ep = -1
        self.solved = False
        self.state_space = state_space

        if sequential_loading:
            path = paths[self.current_ep]
        else:
            path = random.sample(paths, 1)[0]
        self.reset_from_path(path)
        self.cur_binary_state = self.represent_state()
        self.avatar_looks = ['^', '>', 'v', '<']
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.MultiBinary(space_sizes[state_space])

    def reset_from_path(self, path):
        self.solved = False
        with open(path, 'r') as f:
            data = load(f)
        self.W = data['gridsz_num_cols']
        self.H = data['gridsz_num_rows']
        self.map = []
        self.walls = data['walls']
        self.CRASH = False
        self.FINISH = False
        self.init_markers = data['pregrid_markers']
        self.cur_markers = self.init_markers
        self.final_markers = data['postgrid_markers']

        i = data['pregrid_agent_row']
        j = data['pregrid_agent_col']
        dir = data['pregrid_agent_dir']
        self.init_avatar_pos = [i, j, direction_dict[dir]]
        self.cur_avatar_pos = self.init_avatar_pos
        i = data['postgrid_agent_row']
        dir = data['postgrid_agent_dir']
        j = data['postgrid_agent_col']
        self.final_avatar_pos = [i, j, direction_dict[dir]]
        self.cur_human_state = [
            [self.cur_avatar_pos, self.cur_markers, self.walls],
            [self.final_avatar_pos, self.final_markers, self.walls]
        ]

    def represent_state(self):

        pre_grid = self.represent_grid(self.cur_human_state[0])

        s0_pg = self.state_space == 'S0'

        post_grid = self.represent_grid(self.cur_human_state[1], s0_pg=s0_pg)
        return np.array(pre_grid + post_grid).astype('int8')

    def represent_grid(self, human_state, s0_pg=False):
        """
        S0: (most compact)
        For every cell, 1-bit(agent existence) + 1-bit(marker). [2 bits per cell]
        For every grid, 4-bit(agent orientation). [2 ·16 + 4 = 36 bits per grid]
        For both post-grid and pre-grid, 16-bit(wall). [A bit for every cell]

        for every cell, 1-bit(existence) + 1-bit(wall) + 1-bit(marker). [3 bits per cell]
        For every grid, 4-bit(orientation). [3 ·16 + 4 = 52 bits per grid]
        Since we need to have the pre-grid and the post-grid in the state, we end up with:
        state vector size of 52 ·2 = 104 bits.

        :return:
        """
        avatar_pos, markers, walls = human_state
        if self.state_space == 'S0':
            # 16 2-bit cells
            state = [[0, 0] for _ in range(16)]
            # existence bit
            cell_idx = avatar_pos[0] * self.W + avatar_pos[1]
            state[cell_idx][0] = 1
            # marker bit
            for marker in markers:
                mar_i, mar_j = marker
                cell_idx = mar_i * self.W + mar_j
                state[cell_idx][1] = 1
            # orientation bits
            orientation_bits = [0] * 4
            orientation_bits[avatar_pos[2]] = 1

            # for both grids
            walls_bits = [0] * 16
            for wall in walls:
                wall_i, wall_j = wall
                cell_idx = wall_i * self.W + wall_j
                walls_bits[cell_idx] = 1

            state = list(itertools.chain(*state))
            state.extend(orientation_bits)
            if s0_pg:
                state.extend(walls_bits)
        elif self.state_space == "S1":
            state = [[0, 0, 0] for _ in range(16)]
            # Existence bit
            cell_idx = avatar_pos[0] * self.W + avatar_pos[1]
            state[cell_idx][0] = 1
            # Wall bit
            for wall in walls:
                wall_i, wall_j = wall
                cell_idx = wall_i * self.W + wall_j
                state[cell_idx][1] = 1
            # marker bit
            for marker in markers:
                mar_i, mar_j = marker
                cell_idx = mar_i * self.W + mar_j
                state[cell_idx][2] = 1
            # orientation bits
            orientation_bits = [0] * 4
            orientation_bits[avatar_pos[2]] = 1
            state = list(itertools.chain(*state))
            state.extend(orientation_bits)
        elif self.state_space == "S2":
            state = [[0, 0, 0, 0, 0, 0] for _ in range(16)]
            # Existence bit
            cell_idx = avatar_pos[0] * self.W + avatar_pos[1]
            state[cell_idx][avatar_pos[2]] = 1
            # Wall bit
            for wall in walls:
                wall_i, wall_j = wall
                cell_idx = wall_i * self.W + wall_j
                state[cell_idx][4] = 1
            # marker bit
            for marker in markers:
                mar_i, mar_j = marker
                cell_idx = mar_i * self.W + mar_j
                state[cell_idx][5] = 1
            # orientation bits
            state = list(itertools.chain(*state))

        return state

    def make_env(self):
        state = self.cur_human_state
        avatar_pos, markers, walls = state[0]
        target_avatar_pos, target_markers, walls = state[1]
        self.map = [['.' for _ in range(self.W)] for _ in range(self.H)]
        self.post_map = [['.' for _ in range(self.W)] for _ in range(self.H)]
        for wall in walls:
            self.map[wall[0]][wall[1]] = '#'
            self.post_map[wall[0]][wall[1]] = '#'
        for marker in markers:
            self.map[marker[0]][marker[1]] = 'm'
        for marker in target_markers:
            self.post_map[marker[0]][marker[1]] = 'm'
        self.map[avatar_pos[0]][avatar_pos[1]] = self.avatar_looks[avatar_pos[2]]
        self.post_map[target_avatar_pos[0]][target_avatar_pos[1]] = self.avatar_looks[target_avatar_pos[2]]

    def reset(self):
        """
    Important: the observation must be a numpy array
    :return: (np.array)
    """
        # print(self.CRASH)
        if self.sequential_loading:
            self.current_ep = (self.current_ep + 1) % len(self.paths)
        else:
            self.current_ep = random.randint(0,len(self.paths)-1)
        path = self.paths[self.current_ep]
        self.reset_from_path(path)
        return self.represent_state()

    def reward(self, human_state, action):
        self.solved = False
        cur_grid = human_state[0]
        post_grid = human_state[1]
        if action == Action.finish.value:
            if cur_grid == post_grid:
                self.solved = True
                return 1
        elif self.CRASH:
            return -1
        return 0
    def probe(self, s, a):
        saved = copy.deepcopy([self.W ,self.H ,self.map ,self.walls,self.CRASH ,
        self.init_markers,self.cur_markers,self.final_markers ,
        self.init_avatar_pos ,self.cur_avatar_pos ,self.final_avatar_pos ,
        self.cur_human_state ,self.FINISH])
        s_nxt, r, d, i = self.step(a)
        [self.W ,self.H ,self.map ,self.walls,self.CRASH ,
        self.init_markers,self.cur_markers,self.final_markers ,
        self.init_avatar_pos ,self.cur_avatar_pos ,self.final_avatar_pos ,
        self.cur_human_state ,self.FINISH]=saved
        return s_nxt, r, d, i
    def step(self, action):
        [(curr_i, curr_j, curr_dir), cur_markers, walls] = self.cur_human_state[0]
        [(new_i, new_j, new_dir), new_markers, _] = self.cur_human_state[0]
        NO_MARKER_TO_PICK = False
        CANNOT_PUT_MARKER = False
        ON_WALL = False
        OUT_OF_BOUNDS = False
        FINISH = False
        if action == Action.finish.value:
            FINISH = True
            # take action
        elif action == Action.move.value:
            # MOVE
            di, dj = moves[curr_dir]
            [new_i, new_j] = [curr_i + di, curr_j + dj]
            # VALID MOVE?
            OUT_OF_BOUNDS = new_i < 0 or new_i > self.H - 1 or new_j < 0 or new_j > self.W - 1
            for wall in walls:
                if [new_i, new_j] == wall:
                    ON_WALL = True
        elif action == Action.pickMarker.value:
            # PICK MARKER
            NO_MARKER_TO_PICK = True
            for marker in cur_markers:
                if [curr_i, curr_j] == marker:
                    NO_MARKER_TO_PICK = False
                    new_markers.remove(marker)
        elif action == Action.putMarker.value:
            # PUT MARKER
            CANNOT_PUT_MARKER = False
            for marker in cur_markers:
                if [curr_i, curr_j] == marker:
                    CANNOT_PUT_MARKER = True
            new_markers.append([curr_i, curr_j])
        elif action == Action.turnLeft.value:
            new_dir = turn_lefts[curr_dir]
        elif action == Action.turnRight.value:
            new_dir = turn_rights[curr_dir]
        info = {
            "NO_MARKER_TO_PICK": NO_MARKER_TO_PICK,
            "CANNOT_PUT_MARKER": CANNOT_PUT_MARKER,
            "ON_WALL": ON_WALL,
            "OUT_OF_BOUNDS": OUT_OF_BOUNDS,
            "FINISH": FINISH
        }
        done = FINISH or NO_MARKER_TO_PICK or OUT_OF_BOUNDS or ON_WALL or CANNOT_PUT_MARKER
        self.CRASH = NO_MARKER_TO_PICK or OUT_OF_BOUNDS or ON_WALL or CANNOT_PUT_MARKER
        self.FINISH = FINISH
        reward = self.reward(self.cur_human_state, action)

        if not done:
            # Ignore actions leading to crashes
            self.cur_markers = new_markers
            self.cur_avatar_pos = [new_i, new_j, new_dir]
            self.cur_human_state[0] = [self.cur_avatar_pos, self.cur_markers, walls]
        info['ep'] = self.current_ep
        # print("ACTION", action, done, self.cur_markers, info)
        return self.represent_state(), reward, done, info

    def render(self, mode='console'):
        self.make_env()
        for r in range(self.W):
            for c in range(self.H):
                if self.map[r][c] in self.avatar_looks:
                    print(BColors.OKBLUE + self.map[r][c] + BColors.ENDC, " ", end='')
                elif self.map[r][c] == 'm':
                    print(BColors.WARNING + 'm' + BColors.ENDC, " ", end='')
                elif self.map[r][c] == '#':
                    print(BColors.FAIL + '#' + BColors.ENDC, " ", end='')
                else:
                    print(self.map[r][c], " ", end='')
            print("            ", end='')
            for c in range(self.H):
                if self.post_map[r][c] in self.avatar_looks:
                    print(BColors.OKBLUE + self.post_map[r][c] + BColors.ENDC, " ", end='')
                elif self.post_map[r][c] == 'm':
                    print(BColors.WARNING + 'm' + BColors.ENDC, " ", end='')
                elif self.post_map[r][c] == '#':
                    print(BColors.FAIL + '#' + BColors.ENDC, " ", end='')
                else:
                    print(self.post_map[r][c], " ", end='')
            print('\n')

    def close(self):
        pass

