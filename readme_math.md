# The Math and Concepts Behind BioNav-SLAM

This document provides a deeper look into the core algorithms and mathematical models that power the BioNav-SLAM agent.

## 1. The Finite State Machine (FSM)

The agent's high-level behavior is controlled by a Finite State Machine. This is a simple model of computation that allows the agent to be in one of several defined "states" at any given time and transition between them based on specific rules.

* **`EXPLORING`**: The initial state. The agent performs a guided random walk to discover unknown areas.
* **`FOUND_FOOD_EXPLORING`**: Triggered upon finding the first food source. The agent continues exploring for a set duration to see if other food is nearby.
* **`RETURNING_HOME`**: Triggered after the exploration period is over and the agent has found at least two food sources. The agent uses its path integrator to navigate back to the nest.
* **`OPTIMAL_FORAGING`**: The main exploitation state. The agent uses A\* to travel between the nest and all known food sources in a systematic tour.

## 2. The Central Complex (CX) Model

The Central Complex is the agent's "brain" and navigation computer. It's a simplified neural network inspired by insect neurobiology. It uses arrays of neurons, where each neuron corresponds to a specific direction (angle). The activity level of a neuron represents the strength of a signal in that direction.

* **Layer 1: TB1 (Compass Heading)**
    * This layer represents the agent's current heading. It's modeled as a cosine function centered on the agent's current angle, $\theta_{heading}$.
    * The activation $A$ for a neuron at angle $\phi$ is:
        $A(\phi) = \cos(\phi - \theta_{heading})$

* **Layer 2: TL2 (Steering Direction)**
    * This layer represents the immediate steering goal. Its activity is centered on the direction the agent wants to turn towards.

* **Layer 3: CL1a (Speed Signal)**
    * This is a simple layer where the activation of all neurons is proportional to the agent's current effective speed.

* **Layer 4: CPU4 (Path Integration / Home Vector)**
    * This is the most critical layer for navigation. It constantly maintains a vector pointing from the agent's current location back to the nest (the "home vector").
    * With every step, it updates the home vector by subtracting the agent's movement vector.
    * Let the home vector be $\vec{H} = (H_x, H_y)$ and the movement vector be $\vec{M} = (M_x, M_y)$. The new home vector $\vec{H'}$ is:
        $\vec{H'} = \vec{H} - \vec{M}$
    * This vector is then encoded back into the neural activity pattern.

* **Layer 5: CPU1 (Memory)**
    * This layer acts as a long-term memory. When the agent discovers a food source, it takes a "snapshot" of its current CPU4 (home vector) activity and stores it. This stored vector is essentially the vector from the food source back to the nest.

## 3. A\* Pathfinding Algorithm

To find the shortest path while avoiding obstacles, the agent uses the A\* (A-star) algorithm. A\* is a graph traversal and path search algorithm that is widely used in robotics and games.

It works by exploring a grid-based map and prioritizing paths that are both short and directed towards the goal. For each grid cell (node) `n`, it calculates a cost function $f(n)$:

$f(n) = g(n) + h(n)$

* $g(n)$: The cost of the path from the start node to node `n`.
* $h(n)$: The "heuristic" – an estimated cost of the cheapest path from node `n` to the goal. In this simulation, we use the **Euclidean distance** as the heuristic, which is the straight-line distance between the two points.

A\* explores the node with the lowest $f(n)$ value first, guaranteeing that it finds the shortest path on the grid.

## 4. Obstacle Avoidance

The agent uses a proactive "whisker" system. This involves projecting two virtual lines forward from the agent's position at an angle.

* Two points, $P_{left}$ and $P_{right}$, are calculated at a set distance in front of the agent.
* The system checks if either of these points falls within the radius of any obstacle.
* If a whisker is "tripped," it triggers a steering response away from that whisker, allowing the agent to turn smoothly before it gets too close to an object.
