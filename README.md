## Project Idea
Quantum Frog is a 2d 8X8 game similar to the original frogger of a frog crossing
from one end of the stree to the other avoiding getting hit by a car. The difference is in way time moves. It is fixed, quantized, time moves 
only when the frog moves, for one time step then stops again. This adds layer of
strategy to the game where players have to stop and think about their
next action. Game is by design a 2 player game both working together
to cross to the other end. One player's action affect the other, in this 
way. So they have to communicate.

Playing the game is not an end by itself. Rather we’re interested in answering big questions about the newly designed game.
How does the game difficulty scale with # of cars?
How can multi agents play the game? 
Is it better to collaborate or compete?
How to come up with the best reward model?
In the commercial version of the game, what would motivate players the best. 
Reward per square advanced, final reward, simultaneous movement.


## Agents
* cars walking side ways right to left and left to right with certain speed (
squares per second)
* two frogs walking from bottom to top and could walk down or right to left

## Objective
We need a way to represent the game using simple matrix based representation.
We also need to train an RL agent(s) that could play the game optimally and
hopefully reveal strcuture about optimal solution and insight to make the
game design better. For example, how difficulty scales as we add more cars. How 
do players' actions affect each other.

## Plan

### 1. Game Environment (Gymnasium API)

The environment follows the [Gymnasium](https://gymnasium.farama.org/) interface (`reset()`, `step(action)`, `render()`).

**State Representation — Multi-channel 8×8 grid:**

| Channel | Contents | Values |
|---------|----------|--------|
| 0 | Frog positions | 1 = frog A, 2 = frog B, 0 = empty |
| 1 | Car positions | 1 = car present, 0 = empty |
| 2 | Car velocities | signed int: +v = right, −v = left |

State shape: `(3, 8, 8)` — compatible with both tabular flattening and conv-net input.

**Action Space:**

Each frog chooses from `{up, down, left, right, stay}` → 5 actions per frog.
- Single-frog mode: `Discrete(5)`
- Two-frog joint mode: `Discrete(25)` or `MultiDiscrete([5, 5])`

**Time Rule:** The environment advances cars by one tick only when `step()` is called (i.e., when a frog acts). Time is frozen between steps.

**Reward Signal:**

| Event | Reward |
|-------|--------|
| Frog reaches top row | +100 |
| Both frogs reach top row | +200 (bonus) |
| Frog hit by car | −100, episode ends |
| Each step taken | −1 (encourages efficiency) |
| Frog moves up one row | +1 (progress shaping) |

**Done Condition:** Both frogs reach the top row (success), or any frog is hit (failure).

### 2. Training — Escalating Difficulty

Training proceeds in stages, each adding complexity:

| Stage | Frogs | Cars | Speed | Algorithm | Goal |
|-------|-------|------|-------|-----------|------|
| 1 | 1 | 1 | 1 sq/step | Tabular Q-Learning | Learn RL basics, validate env |
| 2 | 1 | 2–3 | 1 sq/step | Tabular Q-Learning | Handle multiple obstacles |
| 3 | 1 | 3–4 | mixed (1–2) | DQN | Generalize over more complex traffic |
| 4 | 2 | 2 | 1 sq/step | Independent Q / DQN | Introduce multi-agent, no cooperation |
| 5 | 2 | 3–4 | mixed | MAPPO / QMIX | Cooperative multi-agent |
| 6 | 2 | 6+ | mixed | MAPPO / QMIX | Stress test, difficulty analysis |

Each stage trains until convergence (plateau in rolling avg reward) before moving on.

### 3. Evaluation & Analysis

- **Success rate vs. car count:** For each stage, plot % of episodes where all frogs survive over N eval episodes (no exploration).
- **Average episode length:** Measures efficiency of learned policy.
- **Cooperation gap:** Compare Stage 5 (cooperative) vs. Stage 4 (independent) to quantify the value of communication.
- **Policy visualization:** For key board states, render a heatmap of chosen actions to reveal strategic structure.
- **Difficulty curve:** Fit success rate as a function of car count to characterize how difficulty scales.
