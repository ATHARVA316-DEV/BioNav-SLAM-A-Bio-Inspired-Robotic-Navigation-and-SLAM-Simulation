"""
BioNav-SLAM: Bio-Inspired Robotic Navigation and SLAM Simulation
A Python-based simulation of an insect-inspired agent using a Central Complex model
for navigation, exploration, and Simultaneous Localization and Mapping (SLAM).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from collections import deque
from enum import Enum
import heapq


class AgentState(Enum):
    """Agent behavior states"""
    EXPLORING = 1
    FOUND_FOOD_EXPLORING = 2
    RETURNING_HOME = 3
    OPTIMAL_FORAGING = 4


class LGMD:
    """
    Lobula Giant Movement Detector - collision avoidance neuron
    Models the insect visual system for detecting looming objects
    """
    def __init__(self, n_receptors=8):
        self.n_receptors = n_receptors
        self.receptors = np.zeros(n_receptors)  # Visual field receptors
        self.excitation = 0  # Excitation level
        self.inhibition = 0  # Inhibition level
        self.threshold = 0.6  # Collision threshold
        self.previous_receptors = np.zeros(n_receptors)
        
    def update(self, obstacle_distances):
        """
        Update LGMD based on obstacle distances
        Closer obstacles = higher activation
        """
        # Convert distances to activation (inverse relationship)
        max_dist = 20.0
        self.receptors = np.clip(1.0 - (obstacle_distances / max_dist), 0, 1)
        
        # Calculate rate of change (looming detection)
        expansion_rate = self.receptors - self.previous_receptors
        
        # Excitation: sum of receptors weighted by expansion rate
        self.excitation = np.sum(self.receptors * (1 + expansion_rate * 2))
        
        # Inhibition: lateral inhibition
        self.inhibition = np.mean(self.receptors) * 0.5
        
        # LGMD output
        lgmd_output = max(0, self.excitation - self.inhibition)
        
        self.previous_receptors = self.receptors.copy()
        
        return lgmd_output
    
    def collision_detected(self):
        """Check if collision threat detected"""
        lgmd_output = self.excitation - self.inhibition
        return lgmd_output > self.threshold


class CentralComplex:
    """
    Bio-inspired neural network mimicking insect Central Complex
    """
    def __init__(self, n_tb1=8):
        self.n_tb1 = n_tb1  # Heading neurons
        self.tb1 = np.zeros(n_tb1)  # Current heading
        self.tl2 = np.zeros(2)  # Steering signals [left, right]
        self.cl1a = 0  # Speed signal
        self.cpu4 = np.zeros(2)  # Home vector [x, y]
        self.cpu1_memory = []  # Memory of food locations
        
    def update_heading(self, angle):
        """Update TB1 heading representation"""
        angles = np.linspace(0, 2*np.pi, self.n_tb1, endpoint=False)
        self.tb1 = np.exp(-((angles - angle) ** 2) / (2 * 0.3 ** 2))
        self.tb1 /= np.sum(self.tb1)
        
    def update_home_vector(self, dx, dy):
        """Update CPU4 path integrator"""
        self.cpu4[0] += dx
        self.cpu4[1] += dy
        
    def store_food_location(self):
        """Store current home vector as food memory"""
        self.cpu1_memory.append(self.cpu4.copy())
        
    def reset_home_vector(self):
        """Reset path integrator when at nest"""
        self.cpu4 = np.zeros(2)


class Environment:
    """2D environment with obstacles and food sources"""
    def __init__(self, width=100, height=100):
        self.width = width
        self.height = height
        self.obstacles = []
        self.food_sources = []
        self.nest_pos = np.array([width // 2, height // 2])
        self.generate_random_environment()
        
    def generate_random_environment(self):
        """Generate random obstacles and food sources"""
        # Create border obstacles
        self.obstacles.append((0, 0, self.width, 2))  # Bottom
        self.obstacles.append((0, self.height-2, self.width, 2))  # Top
        self.obstacles.append((0, 0, 2, self.height))  # Left
        self.obstacles.append((self.width-2, 0, 2, self.height))  # Right
        
        # Random internal obstacles
        np.random.seed(42)
        for _ in range(8):
            x = np.random.randint(10, self.width - 20)
            y = np.random.randint(10, self.height - 20)
            w = np.random.randint(5, 15)
            h = np.random.randint(5, 15)
            self.obstacles.append((x, y, w, h))
        
        # Random food sources
        for _ in range(5):
            while True:
                fx = np.random.randint(10, self.width - 10)
                fy = np.random.randint(10, self.height - 10)
                if not self.is_in_obstacle(fx, fy):
                    self.food_sources.append((fx, fy))
                    break
    
    def is_in_obstacle(self, x, y):
        """Check if position is inside obstacle"""
        for ox, oy, ow, oh in self.obstacles:
            if ox <= x <= ox + ow and oy <= y <= oy + oh:
                return True
        return False


class AStar:
    """A* pathfinding algorithm"""
    @staticmethod
    def heuristic(a, b):
        return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
    
    @staticmethod
    def find_path(start, goal, env):
        """Find shortest path avoiding obstacles"""
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: AStar.heuristic(start, goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if AStar.heuristic(current, goal) < 3:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]
            
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if env.is_in_obstacle(neighbor[0], neighbor[1]):
                    continue
                
                tentative_g = g_score[current] + np.sqrt(dx**2 + dy**2)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + AStar.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return []


class Agent:
    """Bio-inspired autonomous agent with LGMD collision avoidance"""
    def __init__(self, env):
        self.env = env
        self.pos = env.nest_pos.copy().astype(float)
        self.angle = 0
        self.speed = 0
        self.max_speed = 1.5
        
        self.cx = CentralComplex()
        self.lgmd = LGMD(n_receptors=8)  # LGMD visual system
        self.state = AgentState.EXPLORING
        
        self.discovered_food = []
        self.visited_food = set()
        self.min_food_threshold = 2
        self.explore_duration = 0
        self.explore_max_duration = 200
        
        self.visit_grid = np.zeros((env.width // 5, env.height // 5))
        self.pheromone_trail = deque(maxlen=500)
        
        self.current_path = []
        self.target_food_idx = None
        
        self.speed_history = deque(maxlen=200)
        self.lgmd_history = deque(maxlen=200)  # Track LGMD activation
        
        # Stuck detection and recovery
        self.position_history = deque(maxlen=30)
        self.stuck_counter = 0
        self.recovery_mode = False
        self.recovery_timer = 0
        
    def sense_obstacles_lgmd(self):
        """
        LGMD-based collision avoidance - mimics insect visual system
        Returns LGMD activation and best escape direction
        """
        n_receptors = 8
        sensor_angles = np.linspace(-np.pi/2, np.pi/2, n_receptors)
        obstacle_distances = np.full(n_receptors, 100.0)  # Max distance
        
        # Sample obstacles in visual field
        for i, sensor_angle in enumerate(sensor_angles):
            check_angle = self.angle + sensor_angle
            
            # Check multiple distances for each receptor
            for dist in range(5, 25, 2):
                check_x = self.pos[0] + dist * np.cos(check_angle)
                check_y = self.pos[1] + dist * np.sin(check_angle)
                
                if self.env.is_in_obstacle(check_x, check_y):
                    obstacle_distances[i] = dist
                    break
        
        # Update LGMD with obstacle distances
        lgmd_activation = self.lgmd.update(obstacle_distances)
        self.lgmd_history.append(lgmd_activation)
        
        # Calculate escape vector based on LGMD receptors
        if self.lgmd.collision_detected():
            # Find direction with least threat
            escape_angles = sensor_angles[obstacle_distances > 15]
            if len(escape_angles) > 0:
                # Prefer angles away from obstacles
                best_escape = escape_angles[len(escape_angles)//2]
                return True, lgmd_activation, best_escape
            else:
                # Emergency: turn sharply
                return True, lgmd_activation, np.random.choice([-np.pi/2, np.pi/2])
        
        return False, lgmd_activation, 0
    
    def sense_obstacles(self):
        """Enhanced whisker-like obstacle detection with LGMD integration"""
        # Use LGMD for primary collision detection
        collision_threat, lgmd_value, escape_angle = self.sense_obstacles_lgmd()
        
        if collision_threat:
            # LGMD-driven avoidance
            self.angle += escape_angle * 0.4
            # Reduce speed when collision detected
            self.speed = max(0.5, self.max_speed * (1.0 - lgmd_value * 0.5))
            return True
        
        # Backup whisker system
        whisker_length = 15
        whisker_angles = [-0.7, -0.5, -0.25, 0, 0.25, 0.5, 0.7]
        
        obstacle_detected = False
        avoidance_turn = 0
        
        for w_angle in whisker_angles:
            check_angle = self.angle + w_angle
            # Check multiple distances
            for dist_mult in [0.5, 1.0, 1.5]:
                check_dist = whisker_length * dist_mult
                check_x = self.pos[0] + check_dist * np.cos(check_angle)
                check_y = self.pos[1] + check_dist * np.sin(check_angle)
                
                if self.env.is_in_obstacle(check_x, check_y):
                    # Stronger turn away from obstacles
                    avoidance_turn -= w_angle * (2.0 / dist_mult)
                    obstacle_detected = True
                    break
        
        if obstacle_detected:
            self.angle += avoidance_turn * 0.5
            # Add extra random turn to escape corners
            self.angle += np.random.uniform(-0.5, 0.5)
            
        return obstacle_detected
    
    def detect_stuck(self):
        """Detect if agent is stuck in one place"""
        self.position_history.append(self.pos.copy())
        
        if len(self.position_history) < 30:
            return False
        
        # Calculate movement variance
        positions = np.array(self.position_history)
        variance = np.var(positions, axis=0)
        total_variance = np.sum(variance)
        
        # If barely moving, we're stuck
        if total_variance < 5.0:
            self.stuck_counter += 1
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)
        
        # If stuck for multiple frames, enter recovery
        if self.stuck_counter > 15:
            return True
        
        return False
    
    def recovery_behavior(self):
        """Execute recovery behavior when stuck"""
        if not self.recovery_mode:
            self.recovery_mode = True
            self.recovery_timer = 0
            # Random recovery direction
            self.angle += np.random.uniform(-np.pi, np.pi)
        
        self.recovery_timer += 1
        
        # Move backwards initially
        if self.recovery_timer < 10:
            self.speed = -self.max_speed * 0.8
        # Then turn sharply
        elif self.recovery_timer < 20:
            self.speed = 0
            self.angle += np.random.uniform(-0.8, 0.8)
        # Move forward with random walk
        elif self.recovery_timer < 50:
            self.speed = self.max_speed
            self.angle += np.random.uniform(-0.3, 0.3)
        else:
            # Exit recovery mode
            self.recovery_mode = False
            self.stuck_counter = 0
            self.position_history.clear()
    
    def update_visit_grid(self):
        """Update exploration map"""
        gx = int(self.pos[0] // 5)
        gy = int(self.pos[1] // 5)
        if 0 <= gx < self.visit_grid.shape[0] and 0 <= gy < self.visit_grid.shape[1]:
            self.visit_grid[gx, gy] += 1
    
    def explore_steering(self):
        """Steer towards less-visited areas"""
        gx = int(self.pos[0] // 5)
        gy = int(self.pos[1] // 5)
        
        if 0 <= gx < self.visit_grid.shape[0] and 0 <= gy < self.visit_grid.shape[1]:
            neighbors = []
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < self.visit_grid.shape[0] and 0 <= ny < self.visit_grid.shape[1]:
                    neighbors.append((self.visit_grid[nx, ny], nx*5, ny*5))
            
            if neighbors:
                neighbors.sort()
                target_x, target_y = neighbors[0][1], neighbors[0][2]
                target_angle = np.arctan2(target_y - self.pos[1], target_x - self.pos[0])
                angle_diff = np.arctan2(np.sin(target_angle - self.angle), 
                                       np.cos(target_angle - self.angle))
                self.angle += angle_diff * 0.1
        
        # Add random exploration
        self.angle += np.random.uniform(-0.2, 0.2)
    
    def check_food_discovery(self):
        """Check if agent discovered food"""
        for i, (fx, fy) in enumerate(self.env.food_sources):
            if i not in self.discovered_food:
                dist = np.sqrt((self.pos[0] - fx)**2 + (self.pos[1] - fy)**2)
                if dist < 3:
                    self.discovered_food.append(i)
                    self.cx.store_food_location()
                    
                    if self.state == AgentState.EXPLORING:
                        self.state = AgentState.FOUND_FOOD_EXPLORING
                        self.explore_duration = 0
                    return True
        return False
    
    def update(self):
        """Main update loop with LGMD-based collision avoidance"""
        self.speed_history.append(self.speed)
        
        # Check if stuck and handle recovery
        if self.detect_stuck():
            self.recovery_behavior()
            self.move_agent()
            return
        
        # Reset recovery mode if active
        if self.recovery_mode:
            self.recovery_behavior()
            self.move_agent()
            return
        
        # State machine
        if self.state == AgentState.EXPLORING:
            self.speed = self.max_speed
            self.explore_steering()
            self.sense_obstacles()  # Now uses LGMD
            
            if self.check_food_discovery():
                pass  # State already changed in check_food_discovery
                
        elif self.state == AgentState.FOUND_FOOD_EXPLORING:
            self.explore_duration += 1
            self.speed = self.max_speed
            self.explore_steering()
            self.sense_obstacles()  # Now uses LGMD
            self.check_food_discovery()
            
            if self.explore_duration > self.explore_max_duration:
                if len(self.discovered_food) >= self.min_food_threshold:
                    self.state = AgentState.RETURNING_HOME
                else:
                    self.state = AgentState.EXPLORING
                    
        elif self.state == AgentState.RETURNING_HOME:
            # Follow home vector
            target_angle = np.arctan2(-self.cx.cpu4[1], -self.cx.cpu4[0])
            angle_diff = np.arctan2(np.sin(target_angle - self.angle), 
                                   np.cos(target_angle - self.angle))
            self.angle += angle_diff * 0.2
            self.speed = self.max_speed
            
            # LGMD-enhanced obstacle avoidance during return
            if self.sense_obstacles():
                # If hitting obstacles, add lateral movement
                self.angle += np.random.uniform(-0.4, 0.4)
            
            # Check if at nest
            dist_to_nest = np.linalg.norm(self.pos - self.env.nest_pos)
            if dist_to_nest < 5:
                self.cx.reset_home_vector()
                self.state = AgentState.OPTIMAL_FORAGING
                self.visited_food.clear()
                
        elif self.state == AgentState.OPTIMAL_FORAGING:
            if not self.current_path:
                # Find next unvisited food
                unvisited = [i for i in self.discovered_food if i not in self.visited_food]
                
                if not unvisited:
                    self.state = AgentState.EXPLORING
                    return
                
                # Find nearest unvisited food
                nearest_idx = None
                min_dist = float('inf')
                for idx in unvisited:
                    fx, fy = self.env.food_sources[idx]
                    dist = np.linalg.norm(self.pos - np.array([fx, fy]))
                    if dist < min_dist:
                        min_dist = dist
                        nearest_idx = idx
                
                # Calculate optimal path
                target = self.env.food_sources[nearest_idx]
                self.current_path = AStar.find_path(self.pos, target, self.env)
                self.target_food_idx = nearest_idx
                
                # If path not found, skip this food and try next
                if not self.current_path:
                    self.visited_food.add(nearest_idx)
                    return
            
            # Follow path with LGMD-based obstacle avoidance
            if self.current_path:
                target = self.current_path[0]
                target_angle = np.arctan2(target[1] - self.pos[1], target[0] - self.pos[0])
                angle_diff = np.arctan2(np.sin(target_angle - self.angle), 
                                       np.cos(target_angle - self.angle))
                self.angle += angle_diff * 0.3
                self.speed = self.max_speed
                
                # LGMD checks for obstacles and recalculates path if needed
                if self.sense_obstacles():
                    # Recalculate path if obstacle encountered
                    if len(self.current_path) > 5:
                        remaining_target = self.current_path[-1]
                        self.current_path = AStar.find_path(self.pos, remaining_target, self.env)
                        if not self.current_path and self.target_food_idx is not None:
                            # Can't reach target, mark as visited and move on
                            self.visited_food.add(self.target_food_idx)
                            self.target_food_idx = None
                
                if np.linalg.norm(self.pos - np.array(target)) < 2:
                    self.current_path.pop(0)
                    
                    if not self.current_path and self.target_food_idx is not None:
                        self.visited_food.add(self.target_food_idx)
                        # Return to nest
                        self.current_path = AStar.find_path(self.pos, self.env.nest_pos, self.env)
                        self.target_food_idx = None
        
        self.move_agent()
    
    def move_agent(self):
        """Move agent with collision detection"""
        dx = self.speed * np.cos(self.angle)
        dy = self.speed * np.sin(self.angle)
        
        new_x = self.pos[0] + dx
        new_y = self.pos[1] + dy
        
        # Multiple collision checks with fallback
        if not self.env.is_in_obstacle(new_x, new_y):
            self.pos[0] = new_x
            self.pos[1] = new_y
            self.cx.update_home_vector(dx, dy)
            self.update_visit_grid()
            self.pheromone_trail.append(self.pos.copy())
        else:
            # Try sliding along obstacle
            if not self.env.is_in_obstacle(new_x, self.pos[1]):
                self.pos[0] = new_x
                self.cx.update_home_vector(dx, 0)
            elif not self.env.is_in_obstacle(self.pos[0], new_y):
                self.pos[1] = new_y
                self.cx.update_home_vector(0, dy)
            else:
                # Completely blocked, turn around
                self.angle += np.random.uniform(np.pi/2, np.pi)
                self.stuck_counter += 2
        
        self.cx.update_heading(self.angle)


class SimulationGUI:
    """Main GUI for visualization"""
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("BioNav-SLAM Simulation")
        self.root.geometry("1400x900")
        
        self.env = Environment(100, 100)
        self.agent = Agent(self.env)
        
        self.running = False
        self.setup_ui()
        
    def setup_ui(self):
        """Setup GUI layout"""
        # Control panel
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        self.start_btn = ttk.Button(control_frame, text="Start", command=self.toggle_simulation)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Reset", command=self.reset_simulation).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(control_frame, text="Speed:").pack(side=tk.LEFT, padx=5)
        self.speed_scale = ttk.Scale(control_frame, from_=0.5, to=3.0, orient=tk.HORIZONTAL, 
                                     command=self.update_speed)
        self.speed_scale.set(1.5)
        self.speed_scale.pack(side=tk.LEFT, padx=5)
        
        self.state_label = ttk.Label(control_frame, text="State: EXPLORING", font=("Arial", 12, "bold"))
        self.state_label.pack(side=tk.LEFT, padx=20)
        
        # Matplotlib figures
        self.fig = plt.figure(figsize=(14, 8))
        gs = self.fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        self.ax_main = self.fig.add_subplot(gs[0:2, 0:2])
        self.ax_pheromone = self.fig.add_subplot(gs[2, 0])
        self.ax_lgmd = self.fig.add_subplot(gs[2, 1])  # LGMD activation plot
        self.ax_speed = self.fig.add_subplot(gs[0, 2])
        self.ax_memory = self.fig.add_subplot(gs[1, 2])
        self.ax_tb1 = self.fig.add_subplot(gs[2, 2], projection='polar')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def update_speed(self, val):
        """Update agent speed"""
        self.agent.max_speed = float(val)
        
    def toggle_simulation(self):
        """Start/stop simulation"""
        self.running = not self.running
        self.start_btn.config(text="Pause" if self.running else "Start")
        if self.running:
            self.run_simulation()
            
    def reset_simulation(self):
        """Reset simulation"""
        self.running = False
        self.env = Environment(100, 100)
        self.agent = Agent(self.env)
        self.start_btn.config(text="Start")
        self.update_plots()
        
    def update_plots(self):
        """Update all visualizations"""
        # Main environment
        self.ax_main.clear()
        self.ax_main.set_xlim(0, self.env.width)
        self.ax_main.set_ylim(0, self.env.height)
        self.ax_main.set_aspect('equal')
        self.ax_main.set_title('Environment', fontsize=12, fontweight='bold')
        
        # Draw obstacles
        for ox, oy, ow, oh in self.env.obstacles:
            self.ax_main.add_patch(plt.Rectangle((ox, oy), ow, oh, color='gray'))
        
        # Draw nest
        self.ax_main.plot(self.env.nest_pos[0], self.env.nest_pos[1], 'g^', markersize=15, label='Nest')
        
        # Draw food
        for i, (fx, fy) in enumerate(self.env.food_sources):
            color = 'red' if i in self.agent.discovered_food else 'orange'
            marker = 'x' if i in self.agent.visited_food else 'o'
            self.ax_main.plot(fx, fy, marker, color=color, markersize=10)
        
        # Draw pheromone trail
        if len(self.agent.pheromone_trail) > 1:
            trail = np.array(self.agent.pheromone_trail)
            self.ax_main.plot(trail[:, 0], trail[:, 1], 'b-', alpha=0.3, linewidth=1)
        
        # Draw agent
        arrow_len = 5
        dx = arrow_len * np.cos(self.agent.angle)
        dy = arrow_len * np.sin(self.agent.angle)
        self.ax_main.arrow(self.agent.pos[0], self.agent.pos[1], dx, dy, 
                          head_width=2, head_length=2, fc='blue', ec='blue')
        
        # Draw current path
        if self.agent.current_path:
            path = np.array(self.agent.current_path)
            self.ax_main.plot(path[:, 0], path[:, 1], 'g--', linewidth=2, alpha=0.7)
        
        # Pheromone map
        self.ax_pheromone.clear()
        self.ax_pheromone.set_title('Visit Map', fontsize=10)
        self.ax_pheromone.imshow(self.agent.visit_grid.T, cmap='hot', origin='lower', 
                                 interpolation='bilinear')
        
        # LGMD activation history
        self.ax_lgmd.clear()
        self.ax_lgmd.set_title('LGMD Collision Detection', fontsize=10)
        self.ax_lgmd.set_ylim(0, 2)
        if self.agent.lgmd_history:
            self.ax_lgmd.plot(list(self.agent.lgmd_history), 'r-', linewidth=2)
            self.ax_lgmd.axhline(y=self.agent.lgmd.threshold, color='orange', 
                                linestyle='--', label='Threshold')
            self.ax_lgmd.fill_between(range(len(self.agent.lgmd_history)), 
                                     list(self.agent.lgmd_history), 
                                     alpha=0.3, color='red')
        self.ax_lgmd.set_xlabel('Time')
        self.ax_lgmd.set_ylabel('LGMD Activation')
        self.ax_lgmd.legend(fontsize=8)
        
        # Speed graph
        self.ax_speed.clear()
        self.ax_speed.set_title('Speed History', fontsize=10)
        self.ax_speed.set_ylim(0, 3)
        if self.agent.speed_history:
            self.ax_speed.plot(list(self.agent.speed_history), 'b-')
        self.ax_speed.set_xlabel('Time')
        self.ax_speed.set_ylabel('Speed')
        
        # CPU1 Memory
        self.ax_memory.clear()
        self.ax_memory.set_title('CPU1 Food Memory', fontsize=10)
        self.ax_memory.set_xlim(-60, 60)
        self.ax_memory.set_ylim(-60, 60)
        self.ax_memory.axhline(0, color='k', linewidth=0.5)
        self.ax_memory.axvline(0, color='k', linewidth=0.5)
        for mem in self.agent.cx.cpu1_memory:
            self.ax_memory.arrow(0, 0, -mem[0], -mem[1], head_width=2, head_length=2, 
                               fc='red', ec='red', alpha=0.7)
        
        # TB1 polar plot
        self.ax_tb1.clear()
        self.ax_tb1.set_title('TB1 Heading', fontsize=10)
        angles = np.linspace(0, 2*np.pi, len(self.agent.cx.tb1), endpoint=False)
        self.ax_tb1.plot(angles, self.agent.cx.tb1, 'o-')
        self.ax_tb1.fill(angles, self.agent.cx.tb1, alpha=0.3)
        
        # Update state label
        status = f"State: {self.agent.state.name} | Food: {len(self.agent.discovered_food)}"
        if self.agent.recovery_mode:
            status += " | RECOVERING"
        if self.agent.lgmd.collision_detected():
            status += " | ⚠️ COLLISION THREAT"
        self.state_label.config(text=status)
        
        self.canvas.draw()
        
    def run_simulation(self):
        """Main simulation loop"""
        if self.running:
            self.agent.update()
            self.update_plots()
            self.root.after(50, self.run_simulation)
            
    def start(self):
        """Start GUI"""
        self.update_plots()
        self.root.mainloop()


if __name__ == "__main__":
    sim = SimulationGUI()
    sim.start()
