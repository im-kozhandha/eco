"""Baseline algorithms for route optimization."""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Any
from utils import calculate_shortest_path, calculate_route_metrics
import heapq

def dijkstra_min_distance(graph: nx.Graph, warehouse: int, customers: List[int]) -> Tuple[List[int], Dict[str, float]]:
    """
    Solve VRP using Dijkstra algorithm optimizing for minimum distance.
    Returns route and metrics.
    """
    route = [warehouse]
    unvisited = set(customers)
    current = warehouse
    
    while unvisited:
        # Find nearest unvisited customer
        min_distance = float('inf')
        next_customer = None
        
        for customer in unvisited:
            _, distance = calculate_shortest_path(graph, current, customer, 'distance')
            if distance < min_distance:
                min_distance = distance
                next_customer = customer
        
        if next_customer is not None:
            route.append(next_customer)
            unvisited.remove(next_customer)
            current = next_customer
        else:
            break  # No reachable customers
    
    # Return to warehouse
    route.append(warehouse)
    
    # Calculate metrics
    metrics = calculate_route_metrics(graph, route)
    
    return route, metrics

def dijkstra_min_time(graph: nx.Graph, warehouse: int, customers: List[int]) -> Tuple[List[int], Dict[str, float]]:
    """
    Solve VRP using Dijkstra algorithm optimizing for minimum time.
    """
    # Add time weights to graph
    for u, v, data in graph.edges(data=True):
        distance = data['distance']
        traffic_mult = data['traffic_multiplier']
        avg_speed = 40  # km/h
        time_weight = (distance / avg_speed) * traffic_mult * 60  # minutes
        graph[u][v]['time_weight'] = time_weight
    
    route = [warehouse]
    unvisited = set(customers)
    current = warehouse
    
    while unvisited:
        # Find customer with minimum time
        min_time = float('inf')
        next_customer = None
        
        for customer in unvisited:
            try:
                time = nx.shortest_path_length(graph, current, customer, weight='time_weight')
                if time < min_time:
                    min_time = time
                    next_customer = customer
            except nx.NetworkXNoPath:
                continue
        
        if next_customer is not None:
            route.append(next_customer)
            unvisited.remove(next_customer)
            current = next_customer
        else:
            break
    
    # Return to warehouse
    route.append(warehouse)
    
    # Calculate metrics
    metrics = calculate_route_metrics(graph, route)
    
    return route, metrics

def greedy_nearest_neighbor(graph: nx.Graph, warehouse: int, customers: List[int]) -> Tuple[List[int], Dict[str, float]]:
    """
    Greedy nearest neighbor algorithm.
    """
    route = [warehouse]
    unvisited = set(customers)
    current = warehouse
    
    while unvisited:
        # Find nearest neighbor
        min_distance = float('inf')
        nearest = None
        
        for customer in unvisited:
            if graph.has_edge(current, customer):
                distance = graph[current][customer]['distance']
            else:
                _, distance = calculate_shortest_path(graph, current, customer, 'distance')
            
            if distance < min_distance:
                min_distance = distance
                nearest = customer
        
        if nearest is not None:
            route.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        else:
            break
    
    # Return to warehouse
    route.append(warehouse)
    
    # Calculate metrics
    metrics = calculate_route_metrics(graph, route)
    
    return route, metrics

def or_tools_vrp(graph: nx.Graph, warehouse: int, customers: List[int]) -> Tuple[List[int], Dict[str, float]]:
    """
    OR-Tools Vehicle Routing Problem solver.
    """
    try:
        from ortools.constraint_solver import pywrapcp
        from ortools.constraint_solver import routing_enums_pb2
        
        # Create distance matrix
        all_nodes = [warehouse] + customers
        num_nodes = len(all_nodes)
        distance_matrix = np.zeros((num_nodes, num_nodes))
        
        for i, node_i in enumerate(all_nodes):
            for j, node_j in enumerate(all_nodes):
                if i != j:
                    _, distance = calculate_shortest_path(graph, node_i, node_j, 'distance')
                    distance_matrix[i][j] = int(distance * 1000)  # Convert to meters
        
        # Create the routing index manager
        manager = pywrapcp.RoutingIndexManager(num_nodes, 1, 0)
        
        # Create routing model
        routing = pywrapcp.RoutingModel(manager)
        
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Setting first solution heuristic
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        
        # Solve the problem
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            # Extract route
            route = []
            index = routing.Start(0)
            
            while not routing.IsEnd(index):
                route.append(all_nodes[manager.IndexToNode(index)])
                index = solution.Value(routing.NextVar(index))
            
            route.append(all_nodes[manager.IndexToNode(index)])  # Add end depot
            
            # Calculate metrics
            metrics = calculate_route_metrics(graph, route)
            
            return route, metrics
        else:
            # Fallback to greedy if OR-Tools fails
            return greedy_nearest_neighbor(graph, warehouse, customers)
    
    except ImportError:
        # OR-Tools not available, fallback to greedy
        print("OR-Tools not available, using greedy nearest neighbor instead")
        return greedy_nearest_neighbor(graph, warehouse, customers)

# Dictionary of all baseline methods
BASELINE_METHODS = {
    'dijkstra_distance': dijkstra_min_distance,
    'dijkstra_time': dijkstra_min_time,
    'greedy_nearest': greedy_nearest_neighbor,
    'ortools_vrp': or_tools_vrp,
}

def run_all_baselines(graph: nx.Graph, warehouse: int, customers: List[int]) -> Dict[str, Dict[str, Any]]:
    """Run all baseline methods and return results."""
    results = {}
    
    for method_name, method_func in BASELINE_METHODS.items():
        try:
            route, metrics = method_func(graph, warehouse, customers)
            results[method_name] = {
                'route': route,
                'metrics': metrics,
                'success': len(set(route) & set(customers)) == len(customers)
            }
            print(f"✓ {method_name}: Distance={metrics['distance']:.2f}km, "
                  f"Time={metrics['time']:.2f}min, Emission={metrics['emission']:.2f}gCO2")
        except Exception as e:
            print(f"✗ {method_name} failed: {e}")
            results[method_name] = {
                'route': [],
                'metrics': {'distance': float('inf'), 'time': float('inf'), 'emission': float('inf')},
                'success': False,
                'error': str(e)
            }
    
    return results