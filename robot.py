import numpy as np
import matplotlib.pyplot as plt
import math
from heapq import heappush, heappop
from node import Node

class Robot:
    def __init__(self, map_size=(15, 15), sensor_range=5.0):
        self.map_size = map_size
        self.sensor_range = sensor_range
        self.obstacle_map = np.zeros(map_size, dtype=bool)
        self.movements = [(0, 1), (1, 0), (0, -1), (-1, 0),
                         (1, 1), (-1, 1), (1, -1), (-1, -1)]
        
        # Unicycle model parameters
        self.x = 0.0  # x position
        self.y = 0.0  # y position
        self.theta = 0.0  # orientation in radians
        self.dt = 0.1  # time step
        self.wheel_radius = 0.1  # meters
        self.wheel_base = 0.3  # meters between wheels
        
    def set_pose(self, x, y, theta):
        """Set robot's initial pose"""
        self.x = x
        self.y = y
        self.theta = theta
    
    def move(self, v, omega):
        """
        Move robot using unicycle model
        v: linear velocity (m/s)
        omega: angular velocity (rad/s)
        """
        # Update pose using unicycle model equations
        self.x += v * math.cos(self.theta) * self.dt
        self.y += v * math.sin(self.theta) * self.dt
        self.theta += omega * self.dt
        
        # Normalize theta to [-pi, pi]
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
        
        return (self.x, self.y, self.theta)
    
    def get_current_pose(self):
        """Return current pose"""
        return (self.x, self.y, self.theta)
    
    def set_obstacles(self, obstacles):
        """Set obstacles in the map"""
        for obs in obstacles:
            x, y = int(obs[0]), int(obs[1])
            if 0 <= x < self.map_size[0] and 0 <= y < self.map_size[1]:
                self.obstacle_map[x, y] = True
    
    def plan_path_astar(self, start, goal):
        """A* path planning algorithm"""
        start_node = Node(start)
        goal_node = Node(goal)
        
        open_list = []
        closed_set = set()
        
        heappush(open_list, (start_node.f, start_node))
        
        while open_list:
            current_node = heappop(open_list)[1]
            
            if self.goal_reached(current_node.position, goal):
                path = []
                while current_node is not None:
                    path.append(current_node.position)
                    current_node = current_node.parent
                return path[::-1]
            
            closed_set.add(current_node.position)
            
            for dx, dy in self.movements:
                next_x = current_node.position[0] + dx
                next_y = current_node.position[1] + dy
                
                if not (0 <= next_x < self.map_size[0] and 0 <= next_y < self.map_size[1]):
                    continue
                    
                if self.obstacle_map[int(next_x), int(next_y)]:
                    continue
                    
                neighbor = Node((next_x, next_y), current_node)
                
                if (next_x, next_y) in closed_set:
                    continue
                
                neighbor.g = current_node.g + math.sqrt(dx**2 + dy**2)
                neighbor.h = math.sqrt((goal[0] - next_x)**2 + (goal[1] - next_y)**2)
                neighbor.f = neighbor.g + neighbor.h
                
                heappush(open_list, (neighbor.f, neighbor))
        
        return None
    
    def path_blocked(self, path):
        """Check if the current path is blocked by obstacles"""
        if not path:
            return True
            
        for point in path:
            x, y = int(point[0]), int(point[1])
            if self.obstacle_map[x, y]:
                return True
        return False
    
    def goal_reached(self, current_pos, goal_pos, threshold=0.5):
        """Check if the goal has been reached within a threshold"""
        return math.sqrt((current_pos[0] - goal_pos[0])**2 + 
                        (current_pos[1] - goal_pos[1])**2) < threshold
    
    def visualize_state(self, goal_pos, path, obstacles):
        """Visualize the current state with unicycle robot"""
        plt.clf()
        plt.grid(True)
        
        # Plot obstacles
        obstacle_x = [obs[0] for obs in obstacles]
        obstacle_y = [obs[1] for obs in obstacles]
        plt.plot(obstacle_x, obstacle_y, 'ks', label='Obstacles')
        
        # Plot path
        if path:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            plt.plot(path_x, path_y, 'b-', label='Planned Path')
        
        # Plot robot as an arrow to show orientation
        robot_arrow_length = 0.5
        dx = robot_arrow_length * math.cos(self.theta)
        dy = robot_arrow_length * math.sin(self.theta)
        plt.arrow(self.x, self.y, dx, dy, 
                 head_width=0.3, head_length=0.2, fc='g', ec='g', label='Robot')
        
        # Plot goal
        plt.plot(goal_pos[0], goal_pos[1], 'ro', label='Goal')
        
        plt.xlim(-1, self.map_size[0])
        plt.ylim(-1, self.map_size[1])
        plt.legend()
        plt.draw()
        plt.pause(0.1)