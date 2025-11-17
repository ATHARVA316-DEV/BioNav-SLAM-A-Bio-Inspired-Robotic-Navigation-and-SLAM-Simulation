# BioNav-SLAM: A Bio-Inspired Robotic Navigation and SLAM Simulation

A Python-based simulation of an insect-inspired agent using a Central Complex model for navigation, exploration, and Simultaneous Localization and Mapping (SLAM).

This project is a real-time Python simulation of an autonomous agent that uses a bio-inspired navigation system modeled after an insect's brain, specifically the Central Complex (CX). The agent's goal is to navigate an unknown 2D environment containing obstacles to discover points of interest ("food"). As it explores, it builds an internal map and, once enough information is gathered, calculates the most efficient routes to visit all known locations. This process demonstrates a fundamental form of Simultaneous Localization and Mapping (SLAM), where an agent builds a map of an unknown environment while simultaneously keeping track of its own location within it.

The simulation showcases how complex, intelligent, and goal-oriented behaviors can emerge from a set of relatively simple rules and a simplified neural network.
<img width="1389" height="936" alt="image" src="https://github.com/user-attachments/assets/b9057db1-1d26-4af7-bb2f-6e2d1b14e5b8" />


<!-- <img width="1600" height="899" alt="image" src="https://github.com/user-attachments/assets/9090eb9d-a712-4c50-9773-02ea58e9b4bf" /> -->

*(You can replace this with a screenshot of your simulation running)*

## Simulation Workflow: From Chaos to Order

A typical run of the simulation follows a distinct, multi-stage process that mirrors how an animal might forage:

1.  **Initial Exploration**: The agent begins at the nest with no knowledge of its surroundings. It enters the `EXPLORING` state, using a guided random walk that encourages it to visit less-traveled areas. It leaves a pheromone trail, creating a temporary map of where it has been.

2.  **Discovery**: The agent eventually stumbles upon a food source. Upon the first discovery, it enters the `FOUND_FOOD_EXPLORING` state. It "remembers" the location of this food by storing its current "home vector" (the path back to the nest) in its Central Complex memory. It then continues exploring for a fixed duration to see if other food sources are nearby.

3.  **The Threshold for Intelligence**: The agent continues this process until it has discovered a minimum number of food sources (default is 2). This rule prevents it from prematurely optimizing a path for a single, lonely location.

4.  **Return to Base**: Once the threshold is met, the agent transitions to the `RETURNING_HOME` state. Using its internal path integrator (the home vector), it navigates back to the nest, ignoring all other distractions.

5.  **Path Optimization**: Upon arriving safely at the nest, the agent enters the `OPTIMAL_FORAGING` state. It accesses its memory of all known food locations and uses the A\* pathfinding algorithm to calculate the absolute shortest, collision-free path to the nearest unvisited food source.

6.  **The Foraging Tour**: The agent now follows this calculated optimal path. When it reaches the food source, it immediately calculates the shortest path back to the nest and follows it. Upon returning, it calculates the shortest path to the next unvisited food source.

7.  **Completion and Resumption**: This efficient, optimized tour continues until all known food locations have been visited. The agent then reverts to the `EXPLORING` state to search for any remaining undiscovered food.

## Key Features in Detail

### Finite State Machine (FSM)

The agent's decision-making is governed by a 4-state machine:

-   **`EXPLORING`**: Search for new areas and food.
-   **`FOUND_FOOD_EXPLORING`**: Continue exploring for a short period after the first food discovery.
-   **`RETURNING_HOME`**: Disengage from all other tasks and navigate directly to the nest.
-   **`OPTIMAL_FORAGING`**: Execute a pre-calculated, efficient tour of all known food sources.

### 5-Layer Central Complex (CX) Model

A simplified neural network that gives the agent its sense of direction and location.

-   **TB1 (Heading)**: An internal compass representing the agent's current direction.
-   **TL2 (Steering)**: Represents the immediate turning direction or goal.
-   **CL1a (Speed)**: Encodes the agent's current speed.
-   **CPU4 (Home Vector)**: The path integrator. It constantly maintains a vector pointing home, which is updated with every step the agent takes.
-   **CPU1 (Memory)**: A long-term memory that stores the home vectors associated with each discovered food location.

### Intelligent Exploration

To avoid getting stuck in loops, the agent's exploration is guided by a "visit grid." It actively tries to steer towards adjacent areas it has visited the least, ensuring more comprehensive coverage of the map.

### A\* Pathfinding

This classic algorithm finds the mathematically shortest path on a grid. It is used during the `OPTIMAL_FORAGING` state to ensure the agent moves with maximum efficiency, avoiding all obstacles.

### Two-Tier Dynamic Obstacle Avoidance

-   **Proactive "Whiskers"**: The agent projects virtual sensors to detect obstacles in its immediate path, allowing it to make smooth, early turns to avoid collisions.
-   **Reactive Collision Check**: A final check ensures the agent's next position is not inside an obstacle, preventing it from ever violating boundaries, even at high speeds.

### Interactive GUI

A comprehensive 3x2 live display built with Tkinter and Matplotlib:

-   **Main Environment**: Shows the agent, nest, obstacles, food, and all paths.
-   **Pheromone Map**: Visualizes the agent's recent travels.
-   **Speed Graph**: Plots the agent's effective speed over the last 200 steps.
-   **CPU1 Memory**: Displays the vectors to all known food locations from the nest's perspective.
-   **Polar Plots**: Two plots showing the real-time neural activity of the CX layers, giving insight into the agent's internal "thoughts."

### Customizable Simulation

All key parameters can be adjusted in real-time via the GUI.

## How to Run

### Prerequisites

You will need Python 3 and the following libraries installed:
-   `numpy`
-   `matplotlib`
-   `tkinter` (this is typically included with standard Python installations on Windows and macOS, but may need to be installed separately on Linux, e.g., `sudo apt-get install python3-tk`).

You can install the required libraries using pip:
```bash
pip install numpy matplotlib
