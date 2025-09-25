"""Utility functions for EcoRouteRL."""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt

def create_road_network(num_nodes: int, seed: int = 42) -> nx.Graph:
    """Create a road network with warehouse and customer nodes."""
    np.random.seed(seed)
    
    # Create a connected graph
    G = nx.Graph()
    
    # Add nodes with positions
    positions = {}
    for i in range(num_nodes):
        positions[i] = (np.random.uniform(0, 100), np.random.uniform(0, 100))
        node_type = 'warehouse' if i == 0 else 'customer'
        G.add_node(i, pos=positions[i], type=node_type)
    
    # Ensure connectivity by creating a minimum spanning tree
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))
            edges.append((i, j, dist))
    
    # Sort by distance and add edges to ensure connectivity
    edges.sort(key=lambda x: x[2])
    
    # Add edges with random selection to create realistic road network
    added_edges = set()
    
    # First, ensure connectivity with MST
    temp_graph = nx.Graph()
    temp_graph.add_nodes_from(range(num_nodes))
    
    for i, j, dist in edges:
        temp_graph.add_edge(i, j, weight=dist)
        if nx.is_connected(temp_graph):
            break
    
    # Add additional edges for realistic road network (about 1.5x nodes)
    target_edges = int(num_nodes * 1.5)
    edge_count = 0
    
    for i, j, base_dist in edges:
        if edge_count >= target_edges:
            break
            
        if (i, j) not in added_edges and np.random.random() < 0.4:
            # Add some randomness to distance
            distance = base_dist * np.random.uniform(0.9, 1.1)
            
            # Calculate edge properties
            traffic_multiplier = np.random.uniform(0.8, 1.5)
            emission_factor = 180  # gCO2/km for petrol vehicle
            
            G.add_edge(i, j, 
                      distance=distance,
                      traffic_multiplier=traffic_multiplier,
                      emission_factor=emission_factor)
            
            added_edges.add((i, j))
            edge_count += 1
    
    # Ensure warehouse (node 0) is well connected
    warehouse_edges = list(G.edges(0))
    if len(warehouse_edges) < 3:
        # Add more connections to warehouse
        for node in range(1, min(4, num_nodes)):
            if not G.has_edge(0, node):
                dist = np.linalg.norm(np.array(positions[0]) - np.array(positions[node]))
                G.add_edge(0, node,
                          distance=dist,
                          traffic_multiplier=np.random.uniform(0.8, 1.5),
                          emission_factor=180)
    
    return G

def calculate_shortest_path(graph: nx.Graph, start: int, end: int, weight: str = 'distance') -> Tuple[List[int], float]:
    """Calculate shortest path between two nodes."""
    try:
        path = nx.shortest_path(graph, start, end, weight=weight)
        path_length = nx.shortest_path_length(graph, start, end, weight=weight)
        return path, path_length
    except nx.NetworkXNoPath:
        return [], float('inf')

def calculate_route_metrics(graph: nx.Graph, route: List[int], vehicle_speed: float = 40) -> Dict[str, float]:
    """Calculate total distance, time, and emissions for a route."""
    if len(route) < 2:
        return {'distance': 0, 'time': 0, 'emission': 0}
    
    total_distance = 0
    total_time = 0
    total_emission = 0
    
    for i in range(len(route) - 1):
        current = route[i]
        next_node = route[i + 1]
        
        if graph.has_edge(current, next_node):
            edge_data = graph[current][next_node]
            distance = edge_data['distance']
            traffic_mult = edge_data['traffic_multiplier']
            emission_factor = edge_data['emission_factor']
            
            # Calculate metrics
            segment_distance = distance
            segment_time = (distance / vehicle_speed) * traffic_mult * 60  # minutes
            segment_emission = distance * emission_factor * traffic_mult / 1000  # gCO2
            
            total_distance += segment_distance
            total_time += segment_time
            total_emission += segment_emission
        else:
            # Use shortest path if direct edge doesn't exist
            path, path_length = calculate_shortest_path(graph, current, next_node, 'distance')
            if path:
                # Approximate metrics for path
                total_distance += path_length
                total_time += (path_length / vehicle_speed) * 60
                total_emission += path_length * 180 / 1000  # Default emission factor
    
    return {
        'distance': total_distance,
        'time': total_time,
        'emission': total_emission
    }

def normalize_state(state: Dict[str, Any], config: Dict[str, Any]) -> np.ndarray:
    """Normalize state for RL agent input."""
    current_node = state['current_node']
    unserved_mask = state['unserved_customers']
    distance = state['cumulative_distance'] / config['distance_ref']
    time = state['cumulative_time'] / config['time_ref']
    emission = state['cumulative_emission'] / config['max_emission']
    time_of_day = state['time_of_day'] / 24.0  # Normalize to [0,1]
    
    # One-hot encode current node
    num_nodes = len(unserved_mask) + 1  # +1 for warehouse
    current_node_onehot = np.zeros(num_nodes)
    current_node_onehot[current_node] = 1
    
    # Combine all features
    normalized_state = np.concatenate([
        current_node_onehot,
        unserved_mask.astype(np.float32),
        [distance, time, emission, time_of_day]
    ])
    
    return normalized_state.astype(np.float32)

def plot_network(graph: nx.Graph, route: List[int] = None, title: str = "Road Network"):
    """Plot the road network with optional route highlighting."""
    plt.figure(figsize=(12, 8))
    
    # Get positions
    pos = nx.get_node_attributes(graph, 'pos')
    if not pos:
        pos = nx.spring_layout(graph, seed=42)
    
    # Draw network
    nx.draw_networkx_edges(graph, pos, alpha=0.5, color='gray', width=1)
    
    # Draw nodes
    warehouse_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'warehouse']
    customer_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'customer']
    
    nx.draw_networkx_nodes(graph, pos, nodelist=warehouse_nodes, 
                          node_color='red', node_size=500, label='Warehouse')
    nx.draw_networkx_nodes(graph, pos, nodelist=customer_nodes, 
                          node_color='lightblue', node_size=300, label='Customers')
    
    # Draw route if provided
    if route and len(route) > 1:
        route_edges = [(route[i], route[i+1]) for i in range(len(route)-1)]
        nx.draw_networkx_edges(graph, pos, edgelist=route_edges, 
                              edge_color='blue', width=3, alpha=0.8)
    
    # Labels
    nx.draw_networkx_labels(graph, pos, font_size=10)
    
    plt.title(title)
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def save_results(results: Dict[str, Any], filename: str = 'results.json'):
    """Save evaluation results to file."""
    import json
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")

def load_results(filename: str = 'results.json') -> Dict[str, Any]:
    """Load evaluation results from file."""
    import json
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return {}