"""Quick demo script for EcoRouteRL - Fixed Version."""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import sys

# Fix matplotlib backend issues
try:
    # Try to use a non-interactive backend if GUI not available
    import tkinter
    matplotlib.use('TkAgg')
except ImportError:
    # Fallback to non-interactive backend
    matplotlib.use('Agg')
    print("ℹ️ Using non-interactive matplotlib backend (plots saved to files)")

from eco_route_env import EcoRouteEnv
from baselines import run_all_baselines
from utils import plot_network
from config import ENV_CONFIG

def run_quick_demo():
    """Run a quick demonstration of EcoRouteRL."""
    print("🚀 EcoRouteRL Quick Demo")
    print("=" * 50)
    
    try:
        # Create environment
        print("🏗️ Creating delivery network...")
        env = EcoRouteEnv(ENV_CONFIG)
        
        print(f"📍 Network: 1 warehouse + {len(env.customers)} customers")
        print(f"🛣️ Roads: {len(env.graph.edges)} connections")
        
        # Visualize network
        print("\n🗺️ Visualizing road network...")
        try:
            fig = plot_network(env.graph, title="EcoRouteRL Demo Network")
            plt.savefig('demo_network.png', dpi=300, bbox_inches='tight')
            plt.close()  # Close instead of show to avoid display issues
            print("✅ Network visualization saved to demo_network.png")
        except Exception as e:
            print(f"⚠️ Visualization failed: {e}")
            print("   (Network still created successfully)")
        
        # Run baselines
        print("\n🔄 Running baseline algorithms...")
        try:
            baseline_results = run_all_baselines(env.graph, env.warehouse, env.customers)
        except Exception as e:
            print(f"❌ Baseline algorithms failed: {e}")
            return False
        
        # Display results
        print("\n📊 BASELINE RESULTS:")
        print("-" * 40)
        
        best_emission = float('inf')
        best_method = ""
        
        method_names = {
            'dijkstra_distance': 'Dijkstra (Distance)',
            'dijkstra_time': 'Dijkstra (Time)',
            'greedy_nearest': 'Greedy Nearest',
            'ortools_vrp': 'OR-Tools VRP'
        }
        
        for method, result in baseline_results.items():
            display_name = method_names.get(method, method)
            
            if result['success']:
                metrics = result['metrics']
                print(f"{display_name}:")
                print(f"  🛣️  Distance: {metrics['distance']:.2f} km")
                print(f"  ⏱️  Time: {metrics['time']:.2f} min")
                print(f"  🌍 CO₂: {metrics['emission']:.2f} g")
                print(f"  🗺️  Route: {' → '.join(map(str, result['route']))}")
                print()
                
                if metrics['emission'] < best_emission:
                    best_emission = metrics['emission']
                    best_method = display_name
            else:
                print(f"{display_name}: ❌ FAILED")
                if 'error' in result:
                    print(f"    Error: {result['error']}")
                print()
        
        if best_method:
            print(f"🏆 Best baseline: {best_method} ({best_emission:.2f}g CO₂)")
        
        # Test RL environment
        print("\n🤖 Testing RL Environment...")
        try:
            obs, info = env.reset()
            print(f"📊 Observation space: {env.observation_space.shape}")
            print(f"🎯 Action space: {env.action_space.n} actions")
            
            # Run a few random steps
            print("\n🎲 Random agent test:")
            total_reward = 0
            step = 0
            
            while step < 5 and not (info.get('customers_served', 0) == len(env.customers)):
                # Choose random valid action
                valid_actions = np.where(env.unserved_customers)[0]
                if len(valid_actions) == 0:
                    break
                    
                action = np.random.choice(valid_actions)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                step += 1
                
                print(f"  Step {step}: Action={action} → Customer {env.customers[action]}, Reward={reward:.2f}")
                print(f"    Served: {info['customers_served']}/{info['total_customers']}")
                
                if done:
                    print("  ✅ All customers served!")
                    break
            
            print(f"🎯 Total reward: {total_reward:.2f}")
            print(f"✅ Final metrics: {info['cumulative_metrics']}")
            
        except Exception as e:
            print(f"❌ RL environment test failed: {e}")
            return False
        
        # Summary
        print("\n" + "=" * 50)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("📋 What you've seen:")
        print("  ✓ Road network generation")
        print("  ✓ Baseline algorithm comparison") 
        print("  ✓ RL environment functionality")
        print("  ✓ Multi-objective reward system")
        print()
        print("🚀 Next steps:")
        print("  1. Run 'python train_agent.py' to train RL agent")
        print("  2. Run 'python evaluate.py' to compare RL vs baselines")
        print("  3. Run 'python visualize.py' for detailed analysis")
        print()
        print("📊 Network visualization saved to demo_network.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs('models', exist_ok=True)
    
    # Run demo
    success = run_quick_demo()
    sys.exit(0 if success else 1)