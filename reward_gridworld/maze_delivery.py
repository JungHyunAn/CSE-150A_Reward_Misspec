"""Start-to-goal scenes with walls and moving obstacles."""

from collections import deque
from dataclasses import dataclass
from random import Random

from .moving_obstacles import MovingObstacleGridworld, State


DEMO_DISTANCE_WEIGHT = 2
OBSTACLE_REGION_SIZE = 5


@dataclass(frozen=True)
class Scene:
    name: str
    title: str
    layout: tuple[str, ...]
    # One anchor per obstacle. Repeated anchors mean the obstacles share a region.
    region_origins: tuple[tuple[int, int], ...]
    demo_actions: str = ""
    demo_trial: int = 0

    @property
    def obstacle_regions(self) -> tuple[tuple[tuple[int, int], ...], ...]:
        """Expand top-left anchors into square movement regions."""
        return tuple(
            tuple((row + dr, col + dc) for dr in range(OBSTACLE_REGION_SIZE) for dc in range(OBSTACLE_REGION_SIZE))
            for row, col in self.region_origins
        )


SCENES = (
    Scene("forked-maze", "Forked maze", (
        "S......#.......",
        ".#.###.#.##.##.",
        "..#....#....#..",
        "..##.#####.##..",
        ".......#.......",
        "..#.........#..",
        "#.##.......##.#",
        "...............",
        "..#.........#..",
        "###.........#.#",
        "...#...###.#...",
        "..###.##....#..",
        ".......#.##.##.",
        "#.##...#....#.#",
        "...#...#......G",
    ), ((5, 5), (5, 5), (5, 5)), demo_actions="SSESSEEEEESEEEESSSSSSESSSEEE"),
    Scene("narrow-crossing", "Narrow crossing", (
        "S..............",
        "......#........",
        "......#........",
        "#.#.##.........",
        "...............",
        "...............",
        "#####.....#####",
        ".............#.",
        "...............",
        ".............#.",
        "#.....#.###.##.",
        "...............",
        "...............",
        "...............",
        "..............G",
    ), ((4, 5), (0, 10), (10, 1)), demo_actions="EEEEEEEEESSSSSSSEEESEESSSSSS"),
    Scene("long-detour", "Long detour", (
        "S......#.......",
        ".......#.......",
        "...#...#.......",
        "...#...........",
        "#########......",
        "...#.......#...",
        "...............",
        "#..........#...",
        "##.....########",
        ".......#...#...",
        ".#.....#.......",
        "...##..........",
        "...#...........",
        "...#...........",
        "...#..........G",
    ), ((6, 2), (0, 9), (10, 9)), demo_actions="EEEEEESSSEEESSSSWWWSSSSEEEEEEEESSS", demo_trial=2),
)


class MazeDeliveryGridworld(MovingObstacleGridworld):
    def __init__(self, scene: Scene, *, demo_guided: bool = False) -> None:
        super().__init__(scene.layout, obstacle_regions=scene.obstacle_regions)
        self.scene = scene
        self.to_goal = self._static_distances(self.grid.goal)
        self.shaping_costs = self._demo_costs() if demo_guided else self.to_goal

    def _static_distances(self, target: int) -> tuple[int, ...]:
        """Find wall-aware distances to a goal or a demonstration waypoint."""
        distances = [-1] * len(self.grid.moves)
        distances[target] = 0
        pending = deque([target])
        while pending:
            cell = pending.popleft()
            for neighbor in self.grid.moves[cell]:
                if distances[neighbor] < 0:
                    distances[neighbor] = distances[cell] + 1
                    pending.append(neighbor)
        if -1 in distances:
            raise ValueError("All free cells in a maze must be connected to the goal.")
        return tuple(distances)

    def demonstration_path(self) -> tuple[int, ...]:
        """Replay the fixed demonstration in its own obstacle scenario."""
        rng = Random(f"maze-demo:{self.scene.name}:{self.scene.demo_trial}")
        state = self.reset(rng)
        path = [state.robot]
        for direction in self.scene.demo_actions:
            if state.robot == self.grid.goal:
                raise ValueError(f"Demonstration in {self.scene.name} continues past the goal.")
            state, _, outcome = self.step(state, "ESWN".index(direction), rng)
            if outcome == "collision":
                raise ValueError(f"Demonstration in {self.scene.name} collides.")
            path.append(state.robot)
        if path[-1] != self.grid.goal:
            raise ValueError(f"Demonstration in {self.scene.name} must reach the goal.")
        return tuple(path)

    def _demo_costs(self) -> tuple[int, ...]:
        distances, progress = self.demonstration_projection()
        length = len(self.scene.demo_actions)
        return tuple(length - index + DEMO_DISTANCE_WEIGHT * distance
                     for distance, index in zip(distances, progress, strict=True))

    def demonstration_projection(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Find the nearest demo distance and index, resolving ties toward the goal."""
        path = self.demonstration_path()
        distances_by_remaining_steps = [self._static_distances(cell) for cell in reversed(path)]
        distances_to_demo = []
        indices = []
        for position in range(len(self.grid.positions)):
            distance, remaining = min((distances[position], steps)
                                      for steps, distances in enumerate(distances_by_remaining_steps))
            distances_to_demo.append(distance)
            indices.append(len(path) - 1 - remaining)
        return tuple(distances_to_demo), tuple(indices)

    def distances_to_demo(self) -> tuple[int, ...]:
        """Return the wall-aware distance to the nearest demonstrated cell."""
        return self.demonstration_projection()[0]

    def remaining_distance(self, state: State) -> int:
        return self.shaping_costs[state.robot]
