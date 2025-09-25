
"""Evaluation script to compare RL agent with baselines."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from eco_route_env import EcoRouteEnv
from baselines import run_all_baselines
from config import ENV_CONFIG
from utils import save_results
import os

def load_trained_agent():
    """Load the trained RL agent."""
    try:
        # Load the model
        model = PPO.load("models/ecorouterl_ppo")
        
        # Load normalization parameters
        env = EcoRouteEnv(ENV_CONFIG)
        if os.path.exists("models/vec_normalize.pkl"):
            vec_env = VecNormalize.load("models/vec_normalize.pkl", env)
        else:
            vec_env = None
            print("Warning: VecNormalize parameters not found, using raw observations")
        
        return model, vec_env
    except FileNotFoundError:
        print("❌ Trained model not found. Please run train_agent.py first.")
        return None, None

def evaluate_rl_agent(model, vec_env, num_episodes=10):
    """Evaluate RL agent performance."""
    if model is None:
        return None
    
    env = EcoRouteEnv(ENV_CONFIG)
    results = []
    
    print("🤖 Evaluating RL agent...")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        
        if vec_env is not None:
            obs = vec_env.normalize_obs(obs.reshape(1, -1)).flatten()
        
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            if vec_env is not None:
                obs = vec_env.normalize_obs(obs.reshape(1, -1)).flatten()
            
            total_reward += reward
        
        # Extract final metrics
        final_metrics = info['cumulative_metrics']
        route = info['route_history']
        success = done and (info['customers_served'] == info['total_customers'])
        
        results.append({
            'route': route,
            'metrics': final_metrics,
            'success': success,
            'total_reward': total_reward
        })
    
    # Aggregate results
    successful_runs = [r for r in results if r['success']]
    
    if successful_runs:
        avg_metrics = {
            'distance': np.mean([r['metrics']['distance'] for r in successful_runs]),
            'time': np.mean([r['metrics']['time'] for r in successful_runs]),
            'emission': np.mean([r['metrics']['emission'] for r in successful_runs])
        }
        success_rate = len(successful_runs) / len(results)
    else:
        avg_metrics = {'distance': float('inf'), 'time': float('inf'), 'emission': float('inf')}
        success_rate = 0.0
    
    return {
        'rl_agent': {
            'route': successful_runs[0]['route'] if successful_runs else [],
            'metrics': avg_metrics,
            'success': success_rate > 0,
            'success_rate': success_rate,
            'all_results': results
        }
    }

def run_comprehensive_evaluation(num_trials=5):
    """Run comprehensive evaluation comparing all methods."""
    print("🔬 Running comprehensive evaluation...")
    
    # Load trained RL agent
    model, vec_env = load_trained_agent()
    
    all_results = []
    
    for trial in range(num_trials):
        print(f"\n--- Trial {trial + 1}/{num_trials} ---")
        
        # Create fresh environment for each trial
        env = EcoRouteEnv(ENV_CONFIG)
        warehouse = 0
        customers = list(range(1, env.num_nodes))
        
        trial_results = {'trial': trial}
        
        # Run baselines
        baseline_results = run_all_baselines(env.graph, warehouse, customers)
        trial_results.update(baseline_results)
        
        # Run RL agent
        if model is not None:
            rl_results = evaluate_rl_agent(model, vec_env, num_episodes=1)
            if rl_results:
                trial_results.update(rl_results)
        
        all_results.append(trial_results)
    
    return aggregate_results(all_results)

def aggregate_results(all_results):
    """Aggregate results across multiple trials."""
    methods = ['dijkstra_distance', 'dijkstra_time', 'greedy_nearest', 'ortools_vrp']
    if 'rl_agent' in all_results[0]:
        methods.append('rl_agent')
    
    aggregated = {}
    
    for method in methods:
        distances = []
        times = []
        emissions = []
        successes = []
        
        for trial_result in all_results:
            if method in trial_result and trial_result[method]['success']:
                distances.append(trial_result[method]['metrics']['distance'])
                times.append(trial_result[method]['metrics']['time'])
                emissions.append(trial_result[method]['metrics']['emission'])
                successes.append(1)
            else:
                successes.append(0)
        
        if distances:  # If there are successful runs
            aggregated[method] = {
                'avg_distance': np.mean(distances),
                'std_distance': np.std(distances),
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'avg_emission': np.mean(emissions),
                'std_emission': np.std(emissions),
                'success_rate': np.mean(successes),
                'num_trials': len(all_results)
            }
        else:
            aggregated[method] = {
                'avg_distance': float('inf'),
                'std_distance': 0,
                'avg_time': float('inf'),
                'std_time': 0,
                'avg_emission': float('inf'),
                'std_emission': 0,
                'success_rate': 0,
                'num_trials': len(all_results)
            }
    
    return aggregated

def calculate_improvements(results):
    """Calculate percentage improvements of RL agent over baselines."""
    if 'rl_agent' not in results:
        print("❌ RL agent results not available for comparison")
        return {}
    
    rl_metrics = results['rl_agent']
    improvements = {}
    
    baseline_methods = ['dijkstra_distance', 'dijkstra_time', 'greedy_nearest', 'ortools_vrp']
    
    for method in baseline_methods:
        if method in results and results[method]['success_rate'] > 0:
            baseline_metrics = results[method]
            
            # Calculate percentage improvements (negative means RL is better)
            distance_imp = ((rl_metrics['avg_distance'] - baseline_metrics['avg_distance']) 
                           / baseline_metrics['avg_distance']) * 100
            time_imp = ((rl_metrics['avg_time'] - baseline_metrics['avg_time']) 
                       / baseline_metrics['avg_time']) * 100
            emission_imp = ((rl_metrics['avg_emission'] - baseline_metrics['avg_emission']) 
                           / baseline_metrics['avg_emission']) * 100
            
            improvements[method] = {
                'distance_improvement': -distance_imp,  # Negative for reduction
                'time_improvement': -time_imp,
                'emission_improvement': -emission_imp,
                'co2_reduction_percent': -emission_imp  # Primary metric
            }
    
    return improvements

def create_comparison_plots(results, improvements):
    """Create visualization plots comparing all methods."""
    methods = [m for m in results.keys() if results[m]['success_rate'] > 0]
    
    if not methods:
        print("❌ No successful methods to plot")
        return
    
    # Prepare data for plotting
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
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('EcoRouteRL: Method Comparison', fontsize=16, fontweight='bold')
    
    # Colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(method_names)))
    rl_color = 'red' if 'EcoRouteRL\n(PPO)' in method_names else colors[0]
    plot_colors = [rl_color if name == 'EcoRouteRL\n(PPO)' else colors[i] 
                   for i, name in enumerate(method_names)]
    
    # Distance comparison
    bars1 = axes[0, 0].bar(method_names, distances, color=plot_colors, alpha=0.8)
    axes[0, 0].set_title('Total Distance (km)')
    axes[0, 0].set_ylabel('Distance (km)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars1, distances):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(distances)*0.01,
                       f'{val:.1f}', ha='center', va='bottom')
    
    # Time comparison
    bars2 = axes[0, 1].bar(method_names, times, color=plot_colors, alpha=0.8)
    axes[0, 1].set_title('Total Time (minutes)')
    axes[0, 1].set_ylabel('Time (min)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    for bar, val in zip(bars2, times):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01,
                       f'{val:.1f}', ha='center', va='bottom')
    
    # Emission comparison (PRIMARY METRIC)
    bars3 = axes[1, 0].bar(method_names, emissions, color=plot_colors, alpha=0.8)
    axes[1, 0].set_title('Total CO₂ Emissions (g)', fontweight='bold', color='darkgreen')
    axes[1, 0].set_ylabel('CO₂ Emissions (g)', fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    for bar, val in zip(bars3, emissions):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(emissions)*0.01,
                       f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Success rate comparison
    bars4 = axes[1, 1].bar(method_names, success_rates, color=plot_colors, alpha=0.8)
    axes[1, 1].set_title('Success Rate (%)')
    axes[1, 1].set_ylabel('Success Rate (%)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].set_ylim(0, 105)
    axes[1, 1].grid(True, alpha=0.3)
    
    for bar, val in zip(bars4, success_rates):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                       f'{val:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('evaluation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create improvements plot if RL agent is available
    if improvements and 'rl_agent' in results:
        create_improvements_plot(improvements)

def create_improvements_plot(improvements):
    """Create plot showing RL improvements over baselines."""
    if not improvements:
        return
    
    methods = list(improvements.keys())
    co2_improvements = [improvements[m]['co2_reduction_percent'] for m in methods]
    time_improvements = [improvements[m]['time_improvement'] for m in methods]
    distance_improvements = [improvements[m]['distance_improvement'] for m in methods]
    
    name_mapping = {
        'dijkstra_distance': 'vs Dijkstra\n(Distance)',
        'dijkstra_time': 'vs Dijkstra\n(Time)',
        'greedy_nearest': 'vs Greedy\nNearest',
        'ortools_vrp': 'vs OR-Tools\nVRP'
    }
    
    method_labels = [name_mapping.get(m, m) for m in methods]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    x = np.arange(len(methods))
    width = 0.25
    
    bars1 = ax.bar(x - width, distance_improvements, width, label='Distance Reduction %', alpha=0.8)
    bars2 = ax.bar(x, time_improvements, width, label='Time Reduction %', alpha=0.8)
    bars3 = ax.bar(x + width, co2_improvements, width, label='CO₂ Reduction %', alpha=0.8, color='green')
    
    ax.set_xlabel('Baseline Methods')
    ax.set_ylabel('Improvement (%)')
    ax.set_title('EcoRouteRL Performance vs Baselines\n(Positive = Better)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                   f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
    
    plt.tight_layout()
    plt.savefig('rl_improvements.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_detailed_results(results, improvements):
    """Print detailed evaluation results."""
    print("\n" + "="*80)
    print("🏆 ECOROUTERL EVALUATION RESULTS")
    print("="*80)
    
    # Print individual method results
    for method, data in results.items():
        method_name = {
            'dijkstra_distance': 'Dijkstra (Min Distance)',
            'dijkstra_time': 'Dijkstra (Min Time)', 
            'greedy_nearest': 'Greedy Nearest Neighbor',
            'ortools_vrp': 'OR-Tools VRP',
            'rl_agent': 'EcoRouteRL (PPO Agent)'
        }.get(method, method)
        
        print(f"\n📊 {method_name}:")
        print(f"   Success Rate: {data['success_rate']*100:.1f}%")
        if data['success_rate'] > 0:
            print(f"   Avg Distance: {data['avg_distance']:.2f} ± {data['std_distance']:.2f} km")
            print(f"   Avg Time: {data['avg_time']:.2f} ± {data['std_time']:.2f} min")
            print(f"   Avg CO₂: {data['avg_emission']:.2f} ± {data['std_emission']:.2f} g")
        else:
            print("   ❌ Method failed in all trials")
    
    # Print improvements
    if improvements:
        print(f"\n🌱 ECOROUTERL IMPROVEMENTS OVER BASELINES:")
        print("-" * 50)
        
        best_co2_reduction = 0
        best_method = ""
        
        for method, impr in improvements.items():
            method_name = {
                'dijkstra_distance': 'Dijkstra (Distance)',
                'dijkstra_time': 'Dijkstra (Time)',
                'greedy_nearest': 'Greedy Nearest',
                'ortools_vrp': 'OR-Tools VRP'
            }.get(method, method)
            
            co2_red = impr['co2_reduction_percent']
            if co2_red > best_co2_reduction:
                best_co2_reduction = co2_red
                best_method = method_name
            
            print(f"vs {method_name}:")
            print(f"   🌍 CO₂ Reduction: {co2_red:+.1f}%")
            print(f"   ⏱️ Time Change: {impr['time_improvement']:+.1f}%")
            print(f"   📏 Distance Change: {impr['distance_improvement']:+.1f}%")
            print()
        
        print("=" * 80)
        if best_co2_reduction > 0:
            print(f"🎉 BEST RESULT: {best_co2_reduction:.1f}% CO₂ reduction vs {best_method}")
        else:
            print("📈 RL agent shows competitive performance with baselines")
        print("=" * 80)

def main():
    """Main evaluation function."""
    print("🔍 Starting EcoRouteRL evaluation...")
    
    # Run comprehensive evaluation
    results = run_comprehensive_evaluation(num_trials=3)
    
    # Calculate improvements
    improvements = calculate_improvements(results)
    
    # Print results
    print_detailed_results(results, improvements)
    
    # Create visualizations
    create_comparison_plots(results, improvements)
    
    # Save results
    evaluation_data = {
        'results': results,
        'improvements': improvements,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    save_results(evaluation_data, 'evaluation_results.json')
    
    print(f"\n✅ Evaluation completed!")
    print(f"📁 Results saved to evaluation_results.json")
    print(f"📊 Plots saved to evaluation_comparison.png and rl_improvements.png")

if __name__ == "__main__":
    main()