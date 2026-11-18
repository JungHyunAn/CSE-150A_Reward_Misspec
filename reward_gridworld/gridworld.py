"""Grid geometry shared by the moving-obstacle simulator."""

LAYOUT = (
    "S.....",
    ".###..",
    "...#..",
    ".#....",
    ".#..#.",
    ".....G",
)
ACTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))  # East, south, west, north.
GAMMA = 0.99
HORIZON = 100

class Gridworld:
    def __init__(self, layout: tuple[str, ...] = LAYOUT) -> None:
        self.positions = tuple(
            (row, col)
            for row, line in enumerate(layout)
            for col, cell in enumerate(line)
            if cell != "#"
        )
        index = {cell: i for i, cell in enumerate(self.positions)}
        self.start = next(i for (r, c), i in index.items() if layout[r][c] == "S")
        self.goal = next(i for (r, c), i in index.items() if layout[r][c] == "G")
        goal_row, goal_col = self.positions[self.goal]
        self.distance = tuple(
            abs(row - goal_row) + abs(col - goal_col) for row, col in self.positions
        )
        self.moves = tuple(
            tuple(index.get((r + dr, c + dc), i) for dr, dc in ACTIONS)
            for i, (r, c) in enumerate(self.positions)
        )
