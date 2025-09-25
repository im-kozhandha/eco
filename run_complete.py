"""Complete pipeline runner for EcoRouteRL."""

import os
import sys
import time
import subprocess
from datetime import datetime

def print_banner():
    """Print welcome banner."""
    print("🌱" + "="*78 + "🌱")
    print("🚀 ECOROUTERL: COMPLETE PIPELINE EXECUTION")
    print("🌱" + "="*78 + "🌱")
    print("🎯 Sustainability-Focused RL for Delivery Route Optimization")
    print("⏰ Started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print()

def create_directories():
    """Create necessary directories."""
    directories = ['models', 'logs', 'tensorboard_logs', 'results']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def check_dependencies():
    """Check if all dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'numpy', 'pandas', 'gymnasium', 'networkx', 
        'stable_baselines3', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies satisfied!")
    return True

def run_step(step_name, script_name, description, expected_duration):
    """Run a pipeline step."""
    print(f"\n{'='*80}")
    print(f"📋 STEP: {step_name}")
    print(f"📝 {description}")
    print(f"⏱️ Expected duration: {expected_duration}")
    print(f"🔄 Executing: python {script_name}")
    print('='*80)
    
    start_time = time.time()
    
    try:
        # Import and run the module
        if script_name == 'demo.py':
            from demo import run_quick_demo
            run_quick_demo()
        elif script_name == 'train_agent.py':
            from train_agent import train_agent
            model, env = train_agent()
        elif script_name == 'evaluate.py':
            from evaluate import main as eval_main
            eval_main()
        elif script_name == 'visualize.py':
            from visualize import main as viz_main
            viz_main()
        
        elapsed = time.time() - start_time
        print(f"\n✅ {step_name} completed in {elapsed:.1f} seconds")
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ {step_name} failed after {elapsed:.1f} seconds")
        print(f"Error: {str(e)}")
        print("Check the error above and fix any issues.")
        return False

def print_results_summary():
    """Print summary of generated files."""
    print(f"\n{'🎉'*80}")
    print("🏆 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'🎉'*80}")
    
    print("\n📁 Generated Files:")
    files_to_check = [
        ('demo_network.png', '🗺️ Network visualization'),
        ('models/ecorouterl_ppo.zip', '🤖 Trained RL model'),
        ('models/training_curves.png', '📈 Training progress'),
        ('evaluation_comparison.png', '📊 Method comparison'),
        ('rl_improvements.png', '📈 RL improvements'),
        ('ecorouterl_dashboard.png', '📋 Complete dashboard'),
        ('sustainability_report.txt', '📄 Impact report'),
        ('evaluation_results.json', '💾 Raw results data'),
    ]
    
    for filename, description in files_to_check:
        if os.path.exists(filename):
            print(f"  ✅ {description}: {filename}")
        else:
            print(f"  ❓ {description}: {filename} (not found)")
    
    print(f"\n🌱 EcoRouteRL Results Summary:")
    print("=" * 40)
    
    # Try to load and display key results
    try:
        import json
        if os.path.exists('evaluation_results.json'):
            with open('evaluation_results.json', 'r') as f:
                data = json.load(f)
            
            results = data['results']
            improvements = data.get('improvements', {})
            
            # Show best CO2 improvement
            if improvements:
                best_improvement = max(improvements.values(), 
                                     key=lambda x: x['co2_reduction_percent'])
                best_method = [k for k, v in improvements.items() 
                             if v == best_improvement][0]
                
                method_names = {
                    'dijkstra_distance': 'Dijkstra (Distance)',
                    'dijkstra_time': 'Dijkstra (Time)',
                    'greedy_nearest': 'Greedy Nearest',
                    'ortools_vrp': 'OR-Tools VRP'
                }
                
                print(f"🏆 Best CO₂ Reduction: {best_improvement['co2_reduction_percent']:.1f}%")
                print(f"🎯 vs {method_names.get(best_method, best_method)}")
                
                if 'rl_agent' in results:
                    rl_data = results['rl_agent']
                    print(f"📊 RL Success Rate: {rl_data['success_rate']*100:.1f}%")
                    print(f"🌍 RL Avg CO₂: {rl_data['avg_emission']:.2f}g per delivery")
    
    except Exception as e:
        print("📊 Results summary not available")
    
    print(f"\n🚀 Next Steps:")
    print("  • View ecorouterl_dashboard.png for complete analysis")
    print("  • Read sustainability_report.txt for impact details")  
    print("  • Modify config.py to experiment with parameters")
    print("  • Scale to larger networks or add new vehicle types")
    
    print(f"\n⏰ Pipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Run the complete EcoRouteRL pipeline."""
    start_time = time.time()
    
    print_banner()
    
    # Step 0: Setup
    print("🔧 Setting up environment...")
    create_directories()
    
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and try again.")
        return
    
    # Pipeline steps
    steps = [
        ("Demo & Baseline Test", "demo.py", 
         "Generate network, test baselines, visualize environment", "2-3 minutes"),
        ("RL Agent Training", "train_agent.py", 
         "Train PPO agent with 100K timesteps", "15-45 minutes"),
        ("Performance Evaluation", "evaluate.py", 
         "Compare RL agent vs all baselines", "3-5 minutes"),
        ("Create Visualizations", "visualize.py", 
         "Generate dashboard, reports, and plots", "2-3 minutes"),
    ]
    
    # Execute pipeline
    success_count = 0
    for step_name, script_name, description, duration in steps:
        if run_step(step_name, script_name, description, duration):
            success_count += 1
        else:
            print(f"\n💥 Pipeline stopped at {step_name}")
            print("Fix the error above and run the failed step individually:")
            print(f"python {script_name}")
            break
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n⏱️ Total execution time: {total_time/60:.1f} minutes")
    
    if success_count == len(steps):
        print_results_summary()
    else:
        print(f"\n⚠️ Pipeline partially completed: {success_count}/{len(steps)} steps")
    
    return success_count == len(steps)

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n🛑 Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)