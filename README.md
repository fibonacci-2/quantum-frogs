## Project Idea
Quantum Frog is a 2d 8X8 game similar to the original frogger of a frog crossing
from one end of the stree to the other avoiding getting hit by a car. The difference is in way time moves. It is fixed, quantized, time moves 
only when the frog moves, for one time step then stops again. This adds layer of
strategy to the game where players have to stop and think about their
next action. Game is by design a 2 player game both working together
to cross to the other end. One player's action affect the other, in this 
way. So they have to communicate.

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

