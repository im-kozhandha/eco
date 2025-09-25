"""Custom Gymnasium environment for EcoRouteRL."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
from utils import create_road_network, calculate_shortest_path, calculate_route_metrics, normalize_state
from config import ENV_CONFIG, REWARD_CONFIG, VEHICLE_CONFIG, NETWORK_CONFIG

class EcoRouteEnv(gym.Env):
    """
    Custom environment for eco-friendly route optimization.
    
    State: [current_node_onehot, unserved_customers_mask, cum_distance, cum_time, cum_emission, time_of_day]
    Action: Select next unserved customer node
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 1}
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        
        self.config = config or ENV_CONFIG
        self.reward_config = REWARD_CONFIG
        self.vehicle_config = VEHICLE_CONFIG['petrol_4w']
        
        # Create road network
        self.num_nodes = self.config['num_customers'] + 1  # +1 for warehouse
        self.graph = create_road_network(self.num_nodes, seed=self.config['seed'])
        self.warehouse = 0
        self.customers = list(range(1, self.num_nodes))
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(self.customers))
        
        # Observation space: [current_node_onehot, unserved_mask, cum_distance, cum_time, cum_emission, time_of_day]
        obs_dim = self.num_nodes + len(self.customers) + 4  # +4 for cumulative metrics + time
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.current_node = self.warehouse
        self.unserved_customers = np.ones(len(self.customers), dtype=bool)
        self.cumulative_distance = 0.0
        self.cumulative_time = 0.0
        self.cumulative_emission = 0.0
        self.time_of_day = 8.0  # Start at 8 AM
        self.step_count = 0
        self.route_history = [self.warehouse]
        
        state = self._get_state()
        info = self._get_info()
        
        return normalize_state(state, self.reward_config), info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        self.step_count += 1
        
        # Map action to customer node
        customer_idx = action
        if customer_idx >= len(self.customers) or not self.unserved_customers[customer_idx]:
            # Invalid action - penalize
            reward = -100
            state = self._get_state()
            info = self._get_info()
            info['invalid_action'] = True
            return normalize_state(state, self.reward_config), reward, False, False, info
        
        target_customer = self.customers[customer_idx]
        
        # Move to customer
        path, path_distance = calculate_shortest_path(self.graph, self.current_node, target_customer, 'distance')
        
        if not path:
            # No path available - penalize and terminate
            reward = -500
            state = self._get_state()
            info = self._get_info()
            return normalize_state(state, self.reward_config), reward, True, False, info
        
        # Calculate step metrics
        step_metrics = self._calculate_step_metrics(self.current_node, target_customer)
        
        # Update cumulative metrics
        self.cumulative_distance += step_metrics['distance']
        self.cumulative_time += step_metrics['time']
        self.cumulative_emission += step_metrics['emission']
        self.time_of_day += step_metrics['time'] / 60  # Convert minutes to hours
        
        # Update state
        self.current_node = target_customer
        self.unserved_customers[customer_idx] = False
        self.route_history.append(target_customer)
        
        # Calculate reward
        reward = self._calculate_step_reward(step_metrics)
        
        # Check termination conditions
        terminated = np.sum(self.unserved_customers) == 0  # All customers served
        truncated = self.step_count >= self.config['max_steps']
        
        # Add terminal reward
        if terminated:
            terminal_reward = self._calculate_terminal_reward(success=True)
            reward += terminal_reward
        elif truncated:
            terminal_reward = self._calculate_terminal_reward(success=False)
            reward += terminal_reward
        
        state = self._get_state()
        info = self._get_info()
        
        return normalize_state(state, self.reward_config), reward, terminated, truncated, info
    
    def _calculate_step_metrics(self, from_node: int, to_node: int) -> Dict[str, float]:
        """Calculate distance, time, and emission for a single step."""
        if self.graph.has_edge(from_node, to_node):
            edge_data = self.graph[from_node][to_node]
            distance = edge_data['distance']
            traffic_mult = edge_data['traffic_multiplier']
        else:
            # Use shortest path
            path, distance = calculate_shortest_path(self.graph, from_node, to_node, 'distance')
            traffic_mult = 1.0  # Default traffic multiplier
        
        # Calculate metrics
        time = (distance / self.vehicle_config['avg_speed']) * traffic_mult * 60  # minutes
        emission = distance * self.vehicle_config['emission_factor'] * traffic_mult / 1000  # gCO2
        
        return {'distance': distance, 'time': time, 'emission': emission}
    
    def _calculate_step_reward(self, metrics: Dict[str, float]) -> float:
        """Calculate reward for a single step."""
        # Normalize metrics
        norm_distance = metrics['distance'] / self.reward_config['distance_ref']
        norm_time = metrics['time'] / self.reward_config['time_ref']
        norm_emission = metrics['emission'] / self.reward_config['max_emission']
        
        # Calculate step reward (negative because we want to minimize)
        step_reward = -(
            self.reward_config['w_distance'] * norm_distance +
            self.reward_config['w_time'] * norm_time +
            self.reward_config['w_emission'] * norm_emission
        )
        
        return step_reward
    
    def _calculate_terminal_reward(self, success: bool) -> float:
        """Calculate terminal reward based on episode outcome."""
        if success:
            reward = self.reward_config['success_reward']
            
            # On-time bonus (if total time < 4 hours)
            if self.cumulative_time < 240:  # 4 hours in minutes
                reward += self.reward_config['ontime_bonus']
        else:
            reward = self.reward_config['failure_penalty']
        
        # Penalty for total emissions and time
        emission_penalty = self.reward_config['beta_emission'] * self.cumulative_emission
        time_penalty = self.reward_config['gamma_time'] * (self.cumulative_time / 60)  # Convert to hours
        
        reward -= emission_penalty + time_penalty
        
        return reward
    
    def _get_state(self) -> Dict[str, Any]:
        """Get current state dictionary."""
        return {
            'current_node': self.current_node,
            'unserved_customers': self.unserved_customers.copy(),
            'cumulative_distance': self.cumulative_distance,
            'cumulative_time': self.cumulative_time,
            'cumulative_emission': self.cumulative_emission,
            'time_of_day': self.time_of_day
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information."""
        return {
            'route_history': self.route_history.copy(),
            'customers_served': len(self.customers) - np.sum(self.unserved_customers),
            'total_customers': len(self.customers),
            'step_count': self.step_count,
            'cumulative_metrics': {
                'distance': self.cumulative_distance,
                'time': self.cumulative_time,
                'emission': self.cumulative_emission
            }
        }
    
    def render(self, mode: str = 'human'):
        """Render the current state."""
        if mode == 'human':
            print(f"Step: {self.step_count}")
            print(f"Current node: {self.current_node}")
            print(f"Unserved customers: {np.where(self.unserved_customers)[0]}")
            print(f"Cumulative - Distance: {self.cumulative_distance:.2f}km, "
                  f"Time: {self.cumulative_time:.2f}min, "
                  f"Emission: {self.cumulative_emission:.2f}gCO2")
            print(f"Route: {' -> '.join(map(str, self.route_history))}")
            print("-" * 50)
    
    def close(self):
        """Close the environment."""
        pass