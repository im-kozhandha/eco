# run_complete_enhanced.py - Enhanced pipeline with continuous training
"""Enhanced complete pipeline for EcoRouteRL with continuous training and live dashboard."""

import os
import sys
import time
import subprocess
import threading
from datetime import datetime

def print_banner():
    """Print welcome banner."""
    print("🌱" + "="*78 + "🌱")
    print("🚀 ECOROUTERL: ENHANCED COMPLETE PIPELINE")
    print("🌱" + "="*78 + "🌱")
    print("🎯 Continuous Learning & Live Dashboard System")
    print("⏰ Started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print()

def create_directories():
    """Create necessary directories."""
    directories = ['models', 'models/checkpoints', 'logs', 'tensorboard_logs', 'results']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def check_dependencies():
    """Check if all dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'numpy', 'pandas', 'gymnasium', 'networkx', 
        'stable_baselines3', 'matplotlib', 'seaborn',
        'flask', 'flask_socketio'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'flask_socketio':
                __import__('flask_socketio')
            else:
                __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        for package in missing_packages:
            if package == 'flask_socketio':
                subprocess.check_call([sys.executable, "-m", "pip", "install", "flask-socketio"])
            else:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("✅ All packages installed!")
    else:
        print("✅ All dependencies satisfied!")
    
    return True

def run_initial_demo():
    """Run initial demo and baseline comparison."""
    print(f"\n{'='*80}")
    print("📋 STEP 1: Initial Demo & Baseline Comparison")
    print("📝 Generate network, test baselines, create visualizations")
    print('='*80)
    
    try:
        from demo import run_quick_demo
        success = run_quick_demo()
        
        if success:
            print("✅ Demo completed successfully!")
            return True
        else:
            print("❌ Demo failed - check error messages above")
            return False
            
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        return False

def start_continuous_training(training_mode="adaptive"):
    """Start continuous training in background."""
    print(f"\n{'='*80}")
    print("📋 STEP 2: Continuous RL Training")
    print("📝 Train agent with automatic checkpointing and resuming")
    print(f"🎯 Mode: {training_mode}")
    print('='*80)
    
    def training_worker():
        try:
            from continuous_training import ContinuousTrainer
            trainer = ContinuousTrainer()
            
            # Try to resume existing training
            if trainer.load_existing_training():
                print("🔄 Resuming from previous training session...")
            else:
                print("🚀 Starting fresh training...")
                trainer.create_fresh_training()
            
            if training_mode == "adaptive":
                # Train until good performance (target reward = 300)
                trainer.run_adaptive_training(target_performance=300, max_hours=2)
            elif training_mode == "extended":
                # Train for extended period
                trainer.train_continuously(total_additional_timesteps=500000)
            else:
                # Standard training
                trainer.train_continuously(total_additional_timesteps=200000)
                
        except Exception as e:
            print(f"❌ Training failed: {e}")
    
    # Start training in background thread
    training_thread = threading.Thread(target=training_worker, daemon=True)
    training_thread.start()
    
    print("✅ Continuous training started in background!")
    print("📊 Training will continue while you use other features")
    
    return training_thread

def launch_live_dashboard():
    """Launch the live dashboard server."""
    print(f"\n{'='*80}")
    print("📋 STEP 3: Live Interactive Dashboard")
    print("📝 Futuristic real-time visualization with agent movement")
    print('='*80)
    
    def dashboard_worker():
        try:
            # Import and run dashboard server
            print("🌐 Starting dashboard server...")
            
            # Set environment variable to avoid flask reloader
            os.environ['WERKZEUG_RUN_MAIN'] = 'true'
            
            from dashboard_server import socketio, app
            socketio.run(app, host='0.0.0.0', port=5000, debug=False)
            
        except Exception as e:
            print(f"❌ Dashboard server failed: {e}")
    
    # Start dashboard in background thread
    dashboard_thread = threading.Thread(target=dashboard_worker, daemon=True)
    dashboard_thread.start()
    
    print("✅ Live dashboard launched!")
    print("🌐 Open http://localhost:5000 in your browser")
    print("🎮 Interactive controls available")
    
    return dashboard_thread

def run_periodic_evaluation():
    """Run periodic evaluation and comparison."""
    print(f"\n{'='*80}")
    print("📋 STEP 4: Continuous Evaluation")
    print("📝 Regular performance monitoring and comparison")
    print('='*80)
    
    def evaluation_worker():
        evaluation_count = 0
        while True:
            try:
                time.sleep(300)  # Wait 5 minutes between evaluations
                evaluation_count += 1
                
                print(f"\n🔬 Running evaluation #{evaluation_count}...")
                
                # Check if model exists before evaluation
                if os.path.exists("models/ecorouterl_ppo.zip"):
                    from evaluate import main as eval_main
                    eval_main()
                    print(f"✅ Evaluation #{evaluation_count} completed")
                    
                    # Generate updated visualizations
                    from visualize import main as viz_main
                    viz_main()
                    print(f"📊 Visualizations updated")
                else:
                    print(f"⏳ Model not ready yet, skipping evaluation #{evaluation_count}")
                    
            except Exception as e:
                print(f"⚠️ Evaluation #{evaluation_count} failed: {e}")
            
    # Start evaluation in background
    eval_thread = threading.Thread(target=evaluation_worker, daemon=True)
    eval_thread.start()
    
    print("✅ Periodic evaluation started!")
    print("📊 Evaluations will run every 5 minutes")
    
    return eval_thread

def print_usage_instructions():
    """Print usage instructions for the running system."""
    print(f"\n{'🎉'*80}")
    print("🏆 ECOROUTERL ENHANCED SYSTEM IS NOW RUNNING!")
    print(f"{'🎉'*80}")
    
    print("\n🌐 LIVE DASHBOARD:")
    print("  📱 Open: http://localhost:5000")
    print("  🎮 Features:")
    print("    • Real-time agent movement visualization")
    print("    • Live training progress charts")
    print("    • Interactive start/pause/reset controls")
    print("    • Futuristic green/black/yellow UI")
    print("    • Cute hand-drawn network elements")
    print("    • Sustainability metrics tracking")
    
    print("\n🤖 CONTINUOUS TRAINING:")
    print("  🔄 Status: Running in background")
    print("  💾 Auto-saves: Every 50K timesteps")
    print("  📈 Checkpoints: models/checkpoints/")
    print("  📊 Progress: Check dashboard or TensorBoard")
    print("  ⏸️ Control: Use dashboard or Ctrl+C to pause")
    
    print("\n📊 MONITORING:")
    print("  🔬 Evaluations: Every 5 minutes")
    print("  📈 TensorBoard: tensorboard --logdir ./tensorboard_logs")
    print("  📄 Reports: Auto-generated in project folder")
    print("  💾 Live data: models/live_metrics.json")
    
    print("\n🎯 WHAT'S HAPPENING:")
    print("  1. 🤖 Agent learns optimal eco-friendly routes")
    print("  2. 📊 Performance automatically tracked and compared")
    print("  3. 🌱 CO₂ reduction improvements measured in real-time")
    print("  4. 🎮 You can interact with the live dashboard")
    print("  5. 💾 All progress automatically saved for resuming")
    
    print("\n⌨️ CONTROLS:")
    print("  • Dashboard: http://localhost:5000 (interactive)")
    print("  • Ctrl+C: Safely stop all processes")
    print("  • python continuous_training.py: Manual training control")
    print("  • python evaluate.py: On-demand evaluation")
    print("  • python visualize.py: Generate static reports")

def main():
    """Run the enhanced complete pipeline."""
    start_time = time.time()
    
    print_banner()
    
    # Setup
    print("🔧 Setting up enhanced environment...")
    create_directories()
    
    if not check_dependencies():
        print("\n❌ Dependencies check failed")
        return False
    
    # Step 1: Initial Demo
    if not run_initial_demo():
        print("⚠️ Demo failed, but continuing with training...")
    
    # Step 2: Start continuous training
    training_thread = start_continuous_training("adaptive")
    
    # Step 3: Launch live dashboard
    dashboard_thread = launch_live_dashboard()
    
    # Step 4: Start periodic evaluation
    eval_thread = run_periodic_evaluation()
    
    # Wait a moment for servers to start
    time.sleep(3)
    
    # Print instructions
    print_usage_instructions()
    
    # Keep main thread alive and handle user input
    try:
        print(f"\n⏰ System started in {time.time()-start_time:.1f} seconds")
        print("\n💡 Press Ctrl+C to safely stop all processes...")
        
        # Main monitoring loop
        last_status_time = time.time()
        
        while True:
            time.sleep(10)  # Check every 10 seconds
            
            # Print periodic status
            current_time = time.time()
            if current_time - last_status_time > 300:  # Every 5 minutes
                runtime_hours = (current_time - start_time) / 3600
                print(f"\n📊 System Status (Runtime: {runtime_hours:.1f}h)")
                
                # Check if training is producing results
                if os.path.exists("models/live_metrics.json"):
                    try:
                        import json
                        with open("models/live_metrics.json", 'r') as f:
                            metrics = json.load(f)
                        print(f"  🤖 Training: {metrics.get('total_episodes', 0)} episodes")
                        print(f"  🏆 Best reward: {metrics.get('best_mean_reward', 0):.1f}")
                        print(f"  🌱 Training hours: {metrics.get('training_hours', 0):.1f}")
                    except:
                        print("  🔄 Training metrics updating...")
                else:
                    print("  ⏳ Training initializing...")
                
                print("  🌐 Dashboard: http://localhost:5000")
                print("  💡 Use Ctrl+C to stop safely")
                
                last_status_time = current_time
            
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down EcoRouteRL enhanced system...")
        print("💾 All training progress has been automatically saved!")
        print("🔄 You can resume training anytime by running this script again")
        
        # Give threads time to clean up
        time.sleep(2)
        
        print("✅ Shutdown complete!")
        return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print("🔧 Try running individual components:")
        print("  python demo.py")
        print("  python train_agent.py")
        print("  python dashboard_server.py")
        sys.exit(1)