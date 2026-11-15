# Reward Misspecification in Tabular Reinforcement Learning

CSE 150A/250A team project. Train tabular Q-learning agents on a deterministic delivery task to measure how progress bonuses affect delivery efficiency and repeated visits.

## Set up and run

Clone the repository, then run from its root. The Conda environment uses Python 3.12 and pins the numerical and plotting dependencies.

```sh
git clone https://github.com/JungHyunAn/CSE-150A_Reward_Misspec.git
cd CSE-150A_Reward_Misspec
conda env create -f environment.yml
conda activate cse150a-gridworld
python -m reward_gridworld
```

The default sweep trains nine conditions with seeds 0–4, 5,000 episodes per seed and condition, and greedy evaluation every 100 episodes. A quick smoke run uses one seed and one bonus strength.

```sh
python -m reward_gridworld --episodes 100 --eval-every 25 --seeds 0 --alphas 2 --out runs/smoke
```

Use `python -m reward_gridworld --help` for the full set of options. After a dependency change, update with `conda env update -f environment.yml --prune`.

## Delivery task

```text
S.....
.###..
...#..
.#....
.#..#.
.....G
```

`S` is the start, `G` is the goal, and `#` marks walls. The state contains the position and the number of actions remaining. North, south, east, and west moves into a wall or the boundary leave the position unchanged and consume one action. Episodes end on delivery or after 100 actions. The shortest delivery takes 10 actions.

The task reward charges 0.1 per action and adds 10 on delivery, with discount γ = 0.99. All agents use learning rate 0.2 and the same epsilon schedule, which decays linearly from 1.0 to 0.05 over the first 80% of training.

| Training reward | Term added to the task reward |
| --- | --- |
| Baseline | 0 |
| Positive-only progress | α × max(0, d(s) − d(s′)) |
| Potential-based | γΦ(s′) − Φ(s) |

Here d is Manhattan distance to the goal. The potential Φ is −αd before termination and zero after either delivery or timeout. The two shaping methods use α ∈ {0.1, 0.5, 1, 2}. Greedy evaluation starts at the fixed start and breaks tied actions in east, south, west, north order.

## Outputs

| File in `runs/default/` | Contents |
| --- | --- |
| `learning_curves.png` | Task return, success rate, episode length, and repeated entries over training. Shows the baseline and shaped conditions at the largest α, with mean and one standard deviation across seeds. The dotted line marks the optimal task return. |
| `alpha_sweep.png` | Final metrics across all bonus strengths, with the baseline mean for reference. |
| `trajectories.png` | Final greedy paths for the first supplied seed, using the same three conditions as the learning curves. Thicker arrows indicate more traversals. |
| `learning.csv`, `final.csv` | Per-seed checkpoint and final metrics, including discounted return under both the task reward and the condition's training reward. |
| `optimal.csv` | Exact finite-horizon greedy reference for each reward, computed by backward induction. This gives a reference for what each reward optimizes under full planning. |
| `config.json` | Layout, hyperparameters, seeds, and evaluation settings. |

A repeated entry counts a move into any previously visited cell, including the start. Actions that leave the agent in place have their own blocked-move count. For a given seed and checkpoint, the evaluation is one deterministic rollout from the fixed start. Success and timeout rates are computed across seeds.

## Contributing

Create a branch for changes and open a pull request into `main`. Put simulator and reward changes in `reward_gridworld/gridworld.py`, learning changes in `reward_gridworld/learning.py`, and figure changes in `reward_gridworld/visualization.py`. Use the smoke command before opening a pull request. Keep generated runs and local environments out of Git.
