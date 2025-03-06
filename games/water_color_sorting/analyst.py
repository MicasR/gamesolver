from games.water_color_sorting import schemas as s
import numpy as np
from collections import deque


def generate_reduced_game_state(game_state: s.GameState) -> np.ndarray:
    """
    Receives a game state and returns its 2d numpy array representation.
    Where columns are tubes and rows are colors.
    Empty color is equal to 0 and each color has its own number.
    """
    # Create a mapping of color names to unique numbers
    # Empty color is always 0
    color_mapping = {'Empty': 0}
    color_index = 1

    # Scan all tubes to find unique colors and assign them indices
    for tube_name, tube in game_state.tubes.items():
        for color_attr in ['color_1', 'color_2', 'color_3', 'color_4']:
            color = getattr(tube, color_attr)
            if color.name != 'Empty' and color.name not in color_mapping:
                color_mapping[color.name] = color_index
                color_index += 1

    # Create a numpy array to represent the game state
    # 4 rows (one for each color position in a tube) and columns equal to number of tubes
    num_tubes = len(game_state.tubes)
    reduced_state = np.zeros((4, num_tubes), dtype=int)

    # Fill the array with color indices
    for col, (tube_name, tube) in enumerate(game_state.tubes.items()):
        # Bottom to top: color_4 is at the bottom (row 3), color_1 is at the top (row 0)
        reduced_state[0, col] = color_mapping[tube.color_1.name]
        reduced_state[1, col] = color_mapping[tube.color_2.name]
        reduced_state[2, col] = color_mapping[tube.color_3.name]
        reduced_state[3, col] = color_mapping[tube.color_4.name]

    return reduced_state


class Solver:
    def __init__(self, reduced_game_state:np.ndarray):
        self.pos = reduced_game_state

    @staticmethod
    def get_first_non_zero(arr: np.ndarray):
        try:
            return arr[arr != 0][0]
        except IndexError:
            return 0

    @staticmethod
    def get_first_non_zero_index(arr: np.ndarray):
        try:
            return np.where(arr != 0)[0][0]
        except IndexError:
            return 3

    @staticmethod
    def get_last_zero_index(arr: np.ndarray):
        try:
            return np.where(arr == 0)[0][-1]
        except IndexError:
            return 3

    def get_legal_moves_to(self, moveable_to):
        first_non_zero = self.first_non_zero
        n = first_non_zero.shape[0]
        if first_non_zero[moveable_to] == 0:
            return np.where((first_non_zero != 0) & (np.arange(n) != moveable_to))[0], moveable_to
        else:
            return np.where((first_non_zero == first_non_zero[moveable_to]) & (np.arange(n) != moveable_to))[0], moveable_to

    def swap(self, i, j):
            out = self.pos.copy()
            f_row,f_col = (self.get_first_non_zero_index(self.pos[:, i]), i)
            t_row,t_col = (self.get_last_zero_index(self.pos[:, j]), j)
            while f_row<len(self.pos):
                out[f_row][f_col], out[t_row][t_col] = out[t_row][t_col], out[f_row][f_col]
                if f_row + 1 < len(self.pos):
                    if self.pos[f_row + 1][f_col] == self.pos[f_row][f_col]:
                        f_row+=1
                        t_row-=1
                        if t_row <0:
                            break
                    else:
                        break
                else:
                    break
            return Solver(out)

    def isgoal(self):
        return np.array_equiv(self.pos, self.pos[0])

    def __iter__(self):
        self.first_non_zero = np.apply_along_axis(self.get_first_non_zero, 0, self.pos)
        moveable_to = np.where(self.pos[0] == 0)[0]
        legal_moves = tuple(map(self.get_legal_moves_to, moveable_to))

        out = [self.swap(origin, target)
               for origins, target in legal_moves
               for origin in origins]   

        def number_of_full_stacks(pos):
            return np.sum(np.all((pos == [pos[0]]), axis=0))

        def fillings_of_stacks(game):
            pos = game.pos
            return number_of_full_stacks(pos), number_of_full_stacks(pos[1:]), number_of_full_stacks(pos[2:])

        return iter(sorted(out, key=fillings_of_stacks, reverse=True))

    def set_rep(self):
        return frozenset(map(tuple, self.pos.T))

    def __repr__(self):
        return repr(self.pos)

    def solve(pos, depthFirst=False) -> list[s.Move]:
        queue = deque([pos])
        trail = {pos.set_rep(): None}
        move_trail = {}
        solution = deque()
        moves = deque()
        load = queue.append if depthFirst else queue.appendleft

        while not pos.isgoal():
            current_pos = pos
            for idx, m in enumerate(pos):
                if m.set_rep() in trail:
                    continue
                trail[m.set_rep()] = pos

                # Find which tubes were involved in this move
                # We need to compare the current state with the new state
                current_state = current_pos.pos
                new_state = m.pos

                # Find which columns (tubes) changed
                changed_cols = []
                for col in range(current_state.shape[1]):
                    if not np.array_equal(current_state[:, col], new_state[:, col]):
                        changed_cols.append(col)

                # There should be exactly 2 changed columns: source and destination
                if len(changed_cols) == 2:
                    # Determine which is source and which is destination
                    # Source loses liquid (non-zero becomes zero)
                    # Destination gains liquid (zero becomes non-zero)
                    col1, col2 = changed_cols

                    # Check if col1 is source (has more zeros in new state)
                    if np.sum(new_state[:, col1] == 0) > np.sum(current_state[:, col1] == 0):
                        source, dest = col1, col2
                    else:
                        source, dest = col2, col1

                    move_trail[m.set_rep()] = {"select": f"tube {source+1}", "target": f"tube {dest+1}"}

                load(m)
            pos = queue.pop()

        # Reconstruct the solution path
        while pos:
            solution.appendleft(pos)
            if pos.set_rep() in move_trail:
                moves.appendleft(s.Move.model_validate(move_trail[pos.set_rep()]))
            pos = trail[pos.set_rep()]

        # Return the sequence of moves instead of states
        return list(moves)



