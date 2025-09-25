"""Visualization tools for EcoRouteRL."""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Any, Optional
from utils import plot_network
from config import ENV_CONFIG
import pickle
import os

def visualize_routes(graph: nx.Graph, routes_dict: Dict[str, List[int]], title: str = "Route Comparison"):
    """Visualize multiple routes on the same network."""
    plt.figure(figsize=(15, 10))
    
    # Get positions
    pos = nx.get_node_attributes(graph, 'pos')
    if not pos:
        pos = nx.spring_layout(graph, seed=42, k=3, iterations=50)
    
    # Draw base network
    nx.draw_networkx_edges(graph, pos, alpha=0.3, color='lightgray', width=1)
    
    # Draw nodes
    warehouse_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'warehouse']
    customer_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'customer']
    
    nx.draw_networkx_nodes(graph, pos, nodelist=warehouse_nodes, 
                          node_color='red', node_size=800, label='Warehouse', 
                          edgecolors='darkred', linewidths=2)
    nx.draw_networkx_nodes(graph, pos, nodelist=customer_nodes, 
                          node_color='lightblue', node_size=400, label='Customers',
                          edgecolors='navy', linewidths=1)
    
    # Color scheme for different methods
    colors = ['blue', 'green', 'orange', 'purple', 'brown']
    linestyles = ['-', '--', '-.', ':', '-']
    
    # Draw routes
    for i, (method_name, route) in enumerate(routes_dict.items()):
        if len(route) > 1:
            color = colors[i % len(colors)]
            linestyle = linestyles[i % len(linestyles)]
            
            # Create path for route
            for j in range(len(route) - 1):
                if pos[route[j]] and pos[route[j+1]]:
                    x_coords = [pos[route[j]][0], pos[route[j+1]][0]]
                    y_coords = [pos[route[j]][1], pos[route[j+1]][1]]
                    plt.plot(x_coords, y_coords, color=color, linewidth=3, 
                            linestyle=linestyle, alpha=0.8, label=method_name if j == 0 else "")
            
            # Add arrows to show direction
            for j in range(len(route) - 1):
                start_pos = pos[route[j]]
                end_pos = pos[route[j+1]]
                
                # Calculate arrow position (midpoint)
                arrow_x = (start_pos[0] + end_pos[0]) / 2
                arrow_y = (start_pos[1] + end_pos[1]) / 2
                
                # Calculate direction
                dx = end_pos[0] - start_pos[0]
                dy = end_pos[1] - start_pos[1]
                length = np.sqrt(dx**2 + dy**2)
                
                if length > 0:
                    dx_norm = dx / length * 2
                    dy_norm = dy / length * 2
                    
                    plt.arrow(arrow_x, arrow_y, dx_norm, dy_norm, 
                             head_width=1.5, head_length=1, fc=color, ec=color, alpha=0.7)
    
    # Draw node labels
    nx.draw_networkx_labels(graph, pos, font_size=10, font_weight='bold')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def plot_training_progress(training_file: str = 'models/training_metrics.pkl'):
    """Plot training progress from saved metrics."""
    if not os.path.exists(training_file):
        print(f"❌ Training metrics file not found: {training_file}")
        return
    
    with open(training_file, 'rb') as f:
        training_data = pickle.load(f)
    
    if not training_data['episode_rewards']:
        print("No training data available to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('EcoRouteRL Training Progress', fontsize=16, fontweight='bold')
    
    # Smooth the curves using rolling average
    window = min(50, len(training_data['episode_rewards']) // 10)
    
    def smooth_curve(data, window_size):
        if len(data) < window_size:
            return data
        return pd.Series(data).rolling(window=window_size, min_periods=1).mean().values
    
    # Plot rewards
    rewards = training_data['episode_rewards']
    smooth_rewards = smooth_curve(rewards, window)
    axes[0, 0].plot(rewards, alpha=0.3, color='blue', label='Raw')
    axes[0, 0].plot(smooth_rewards, color='blue', linewidth=2, label='Smoothed')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot emissions
    if training_data['episode_emissions']:
        emissions = training_data['episode_emissions']
        smooth_emissions = smooth_curve(emissions, window)
        axes[0, 1].plot(emissions, alpha=0.3, color='green', label='Raw')
        axes[0, 1].plot(smooth_emissions, color='green', linewidth=2, label='Smoothed')
        axes[0, 1].set_title('Episode CO₂ Emissions')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Total CO₂ (g)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
    
    # Plot times
    if training_data['episode_times']:
        times = training_data['episode_times']
        smooth_times = smooth_curve(times, window)
        axes[1, 0].plot(times, alpha=0.3, color='orange', label='Raw')
        axes[1, 0].plot(smooth_times, color='orange', linewidth=2, label='Smoothed')
        axes[1, 0].set_title('Episode Times')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Total Time (min)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
    
    # Plot distances
    if training_data['episode_distances']:
        distances = training_data['episode_distances']
        smooth_distances = smooth_curve(distances, window)
        axes[1, 1].plot(distances, alpha=0.3, color='red', label='Raw')
        axes[1, 1].plot(smooth_distances, color='red', linewidth=2, label='Smoothed')
        axes[1, 1].set_title('Episode Distances')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Total Distance (km)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('training_progress_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("📊 Detailed training progress saved to training_progress_detailed.png")

def create_metrics_dashboard(results_file: str = 'evaluation_results.json'):
    """Create a comprehensive metrics dashboard."""
    try:
        with open(results_file, 'r') as f:
            import json
            data = json.load(f)
        
        results = data['results']
        improvements = data.get('improvements', {})
    except FileNotFoundError:
        print(f"❌ Results file not found: {results_file}")
        return
    
    # Create dashboard
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Method comparison metrics
    methods = [m for m in results.keys() if results[m]['success_rate'] > 0]
    method_names = []
    distances = []
    times = []
    emissions = []
    success_rates = []
    
    name_mapping = {
        'dijkstra_distance': 'Dijkstra\n(Distance)',
        'dijkstra_time': 'Dijkstra\n(Time)',
        'greedy_nearest': 'Greedy\nNearest',
        'ortools_vrp': 'OR-Tools\nVRP',
        'rl_agent': 'EcoRouteRL\n(PPO)'
    }
    
    for method in methods:
        method_names.append(name_mapping.get(method, method))
        distances.append(results[method]['avg_distance'])
        times.append(results[method]['avg_time'])
        emissions.append(results[method]['avg_emission'])
        success_rates.append(results[method]['success_rate'] * 100)
    
    # Colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(method_names)))
    rl_color = 'darkred' if 'EcoRouteRL\n(PPO)' in method_names else colors[0]
    plot_colors = [rl_color if name == 'EcoRouteRL\n(PPO)' else colors[i] 
                   for i, name in enumerate(method_names)]
    
    # Distance comparison
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.bar(method_names, distances, color=plot_colors, alpha=0.8, edgecolor='black')
    ax1.set_title('Total Distance (km)', fontweight='bold')
    ax1.set_ylabel('Distance (km)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    for bar, val in zip(bars1, distances):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(distances)*0.02,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Time comparison
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(method_names, times, color=plot_colors, alpha=0.8, edgecolor='black')
    ax2.set_title('Total Time (minutes)', fontweight='bold')
    ax2.set_ylabel('Time (min)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    for bar, val in zip(bars2, times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.02,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Emission comparison (PRIMARY METRIC)
    ax3 = fig.add_subplot(gs[0, 2])
    bars3 = ax3.bar(method_names, emissions, color=plot_colors, alpha=0.8, edgecolor='black')
    ax3.set_title('CO₂ Emissions (g)', fontweight='bold', fontsize=14, color='darkgreen')
    ax3.set_ylabel('CO₂ Emissions (g)', fontweight='bold', color='darkgreen')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    for bar, val in zip(bars3, emissions):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(emissions)*0.02,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold', color='darkgreen')
    
    # Success rate comparison
    ax4 = fig.add_subplot(gs[0, 3])
    bars4 = ax4.bar(method_names, success_rates, color=plot_colors, alpha=0.8, edgecolor='black')
    ax4.set_title('Success Rate (%)', fontweight='bold')
    ax4.set_ylabel('Success Rate (%)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.set_ylim(0, 105)
    ax4.grid(True, alpha=0.3)
    for bar, val in zip(bars4, success_rates):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}%', ha='center', va='bottom', fontweight='bold')
    
    # Improvements radar chart
    if improvements:
        ax5 = fig.add_subplot(gs[1, :2], projection='polar')
        
        baseline_methods = list(improvements.keys())
        metrics = ['CO₂ Reduction', 'Time Improvement', 'Distance Improvement']
        
        # Prepare data for radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for i, method in enumerate(baseline_methods):
            values = [
                improvements[method]['co2_reduction_percent'],
                improvements[method]['time_improvement'],
                improvements[method]['distance_improvement']
            ]
            values += values[:1]  # Complete the circle
            
            ax5.plot(angles, values, 'o-', linewidth=2, 
                    label=name_mapping.get(method, method).replace('\n', ' '),
                    alpha=0.8)
            ax5.fill(angles, values, alpha=0.15)
        
        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(metrics)
        ax5.set_title('EcoRouteRL Improvements vs Baselines (%)', 
                     fontweight='bold', pad=20)
        ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax5.grid(True)
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Efficiency comparison (emissions per km)
    ax6 = fig.add_subplot(gs[1, 2:])
    efficiency = [e/d if d > 0 else 0 for e, d in zip(emissions, distances)]
    bars6 = ax6.bar(method_names, efficiency, color=plot_colors, alpha=0.8, edgecolor='black')
    ax6.set_title('Emission Efficiency (gCO₂/km)', fontweight='bold')
    ax6.set_ylabel('gCO₂/km')
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(True, alpha=0.3)
    for bar, val in zip(bars6, efficiency):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(efficiency)*0.02,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Summary statistics table
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('tight')
    ax7.axis('off')
    
    # Create summary table
    table_data = []
    for i, method in enumerate(methods):
        method_data = results[method]
        table_data.append([
            name_mapping.get(method, method).replace('\n', ' '),
            f"{method_data['avg_distance']:.1f} ± {method_data['std_distance']:.1f}",
            f"{method_data['avg_time']:.1f} ± {method_data['std_time']:.1f}",
            f"{method_data['avg_emission']:.1f} ± {method_data['std_emission']:.1f}",
            f"{method_data['success_rate']*100:.0f}%",
            f"{efficiency[i]:.2f}"
        ])
    
    # Highlight RL agent row
    row_colors = ['lightcoral' if 'EcoRouteRL' in row[0] else 'white' for row in table_data]
    
    table = ax7.table(cellText=table_data,
                     colLabels=['Method', 'Distance (km)', 'Time (min)', 'CO₂ (g)', 'Success', 'gCO₂/km'],
                     cellLoc='center',
                     loc='center',
                     rowColours=row_colors)
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(table_data) + 1):
        for j in range(6):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            elif 'EcoRouteRL' in table_data[i-1][0]:  # RL agent row
                cell.set_text_props(weight='bold')
    
    ax7.set_title('Detailed Performance Summary', fontweight='bold', fontsize=14, pad=20)
    
    plt.suptitle('🌱 EcoRouteRL: Comprehensive Performance Dashboard', 
                fontsize=20, fontweight='bold', y=0.98)
    
    plt.savefig('ecorouterl_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("📊 Comprehensive dashboard saved to ecorouterl_dashboard.png")

def visualize_network_with_best_routes(results_file: str = 'evaluation_results.json'):
    """Visualize the network with routes from all methods."""
    from eco_route_env import EcoRouteEnv
    
    # Create environment to get the graph
    env = EcoRouteEnv(ENV_CONFIG)
    graph = env.graph
    
    try:
        with open(results_file, 'r') as f:
            import json
            data = json.load(f)
        results = data['results']
    except FileNotFoundError:
        print(f"❌ Results file not found: {results_file}")
        print("Run evaluate.py first to generate results")
        return
    
    # Extract routes from successful methods
    routes_dict = {}
    
    method_names = {
        'dijkstra_distance': 'Dijkstra (Distance)',
        'dijkstra_time': 'Dijkstra (Time)',
        'greedy_nearest': 'Greedy Nearest',
        'ortools_vrp': 'OR-Tools VRP',
        'rl_agent': 'EcoRouteRL (PPO)'
    }
    
    # For this visualization, we need to run the methods once to get actual routes
    # Since we only have aggregated results, let's run a single instance
    warehouse = 0
    customers = list(range(1, env.num_nodes))
    
    from baselines import run_all_baselines
    baseline_results = run_all_baselines(graph, warehouse, customers)
    
    for method, result in baseline_results.items():
        if result['success'] and result['route']:
            routes_dict[method_names[method]] = result['route']
    
    # Add RL route if available (this would need to be generated)
    # For demo purposes, we'll skip the RL route in this visualization
    
    if routes_dict:
        fig = visualize_routes(graph, routes_dict, 
                             "EcoRouteRL: Route Comparison Across Methods")
        plt.savefig('route_comparison.png', dpi=300, bbox_inches='tight')
        print("🗺️ Route comparison saved to route_comparison.png")
    else:
        print("❌ No successful routes to visualize")

def create_sustainability_report():
    """Generate a sustainability impact report."""
    try:
        with open('evaluation_results.json', 'r') as f:
            import json
            data = json.load(f)
        results = data['results']
        improvements = data.get('improvements', {})
    except FileNotFoundError:
        print("❌ No evaluation results found. Run evaluate.py first.")
        return
    
    # Generate report
    report = []
    report.append("🌱 ECOROUTERL SUSTAINABILITY IMPACT REPORT")
    report.append("=" * 60)
    report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Executive Summary
    report.append("📋 EXECUTIVE SUMMARY")
    report.append("-" * 30)
    
    if 'rl_agent' in results and results['rl_agent']['success_rate'] > 0:
        rl_emission = results['rl_agent']['avg_emission']
        
        # Find best baseline
        baseline_emissions = {}
        for method in ['dijkstra_distance', 'dijkstra_time', 'greedy_nearest', 'ortools_vrp']:
            if method in results and results[method]['success_rate'] > 0:
                baseline_emissions[method] = results[method]['avg_emission']
        
        if baseline_emissions:
            best_baseline = min(baseline_emissions.items(), key=lambda x: x[1])
            best_baseline_name = best_baseline[0]
            best_baseline_emission = best_baseline[1]
            
            improvement = ((best_baseline_emission - rl_emission) / best_baseline_emission) * 100
            
            report.append(f"• EcoRouteRL achieves {improvement:.1f}% CO₂ reduction vs best baseline")
            report.append(f"• Average CO₂ per delivery: {rl_emission:.2f}g (vs {best_baseline_emission:.2f}g)")
            report.append(f"• Success rate: {results['rl_agent']['success_rate']*100:.1f}%")
        
        report.append("")
    
    # Environmental Impact
    report.append("🌍 ENVIRONMENTAL IMPACT")
    report.append("-" * 30)
    
    if improvements:
        total_reduction = 0
        count = 0
        
        for method, impr in improvements.items():
            co2_red = impr['co2_reduction_percent']
            if co2_red > 0:
                total_reduction += co2_red
                count += 1
                
                method_name = {
                    'dijkstra_distance': 'Dijkstra (Distance)',
                    'dijkstra_time': 'Dijkstra (Time)',
                    'greedy_nearest': 'Greedy Nearest',
                    'ortools_vrp': 'OR-Tools VRP'
                }.get(method, method)
                
                report.append(f"• vs {method_name}: {co2_red:.1f}% CO₂ reduction")
        
        if count > 0:
            avg_reduction = total_reduction / count
            report.append(f"• Average CO₂ reduction: {avg_reduction:.1f}%")
            
            # Estimate annual impact (hypothetical)
            daily_deliveries = 100
            annual_deliveries = daily_deliveries * 365
            avg_emission_per_delivery = rl_emission if 'rl_agent' in results else 100
            annual_savings_kg = (avg_reduction / 100) * avg_emission_per_delivery * annual_deliveries / 1000
            
            report.append(f"• Estimated annual CO₂ savings for 100 daily deliveries: {annual_savings_kg:.1f} kg")
        
        report.append("")
    
    # Performance Metrics
    report.append("📊 PERFORMANCE METRICS")
    report.append("-" * 30)
    
    for method, data in results.items():
        method_name = {
            'dijkstra_distance': 'Dijkstra (Min Distance)',
            'dijkstra_time': 'Dijkstra (Min Time)',
            'greedy_nearest': 'Greedy Nearest Neighbor',
            'ortools_vrp': 'OR-Tools VRP',
            'rl_agent': 'EcoRouteRL (PPO)'
        }.get(method, method)
        
        if data['success_rate'] > 0:
            report.append(f"{method_name}:")
            report.append(f"  Distance: {data['avg_distance']:.2f} ± {data['std_distance']:.2f} km")
            report.append(f"  Time: {data['avg_time']:.2f} ± {data['std_time']:.2f} min")
            report.append(f"  CO₂: {data['avg_emission']:.2f} ± {data['std_emission']:.2f} g")
            report.append(f"  Success Rate: {data['success_rate']*100:.1f}%")
            report.append("")
    
    # Recommendations
    report.append("💡 RECOMMENDATIONS")
    report.append("-" * 30)
    report.append("• Deploy EcoRouteRL for CO₂-optimized delivery routing")
    report.append("• Consider hybrid approaches combining RL with domain heuristics")
    report.append("• Extend to multi-vehicle and dynamic routing scenarios")
    report.append("• Integrate real-time traffic and emission data")
    report.append("• Implement continuous learning from operational data")
    report.append("")
    
    report.append("=" * 60)
    report.append("📧 For questions contact: EcoRouteRL Team")
    
    # Save report
    with open('sustainability_report.txt', 'w') as f:
        f.write('\n'.join(report))
    
    # Print report
    print('\n'.join(report))
    print(f"\n📄 Full report saved to sustainability_report.txt")

def main():
    """Main visualization function."""
    print("📊 Creating EcoRouteRL visualizations...")
    
    # 1. Plot training progress if available
    if os.path.exists('models/training_metrics.pkl'):
        plot_training_progress()
    else:
        print("⚠️ No training metrics found. Run train_agent.py first.")
    
    # 2. Create comprehensive dashboard
    if os.path.exists('evaluation_results.json'):
        create_metrics_dashboard()
        visualize_network_with_best_routes()
        create_sustainability_report()
    else:
        print("⚠️ No evaluation results found. Run evaluate.py first.")
    
    print("✅ Visualizations completed!")

if __name__ == "__main__":
    main()