# diagnostic.py - Debug EcoRouteRL Issues
"""Diagnostic script to identify and fix EcoRouteRL issues."""

import sys
import os
import traceback

def test_imports():
    """Test all required imports."""
    print("🔍 Testing imports...")
    
    imports_to_test = [
        ('numpy', 'import numpy as np'),
        ('pandas', 'import pandas as pd'),
        ('gymnasium', 'import gymnasium as gym'),
        ('networkx', 'import networkx as nx'),
        ('matplotlib', 'import matplotlib.pyplot as plt'),
        ('stable_baselines3', 'from stable_baselines3 import PPO'),
    ]
    
    failed_imports = []
    
    for name, import_stmt in imports_to_test:
        try:
            exec(import_stmt)
            print(f"  ✅ {name}")
        except ImportError as e:
            print(f"  ❌ {name}: {e}")
            failed_imports.append(name)
        except Exception as e:
            print(f"  ⚠️ {name}: {e}")
    
    return failed_imports

def test_matplotlib():
    """Test matplotlib configuration."""
    print("\n🎨 Testing matplotlib...")
    
    try:
        import matplotlib
        print(f"  📊 Matplotlib version: {matplotlib.__version__}")
        print(f"  🖥️ Current backend: {matplotlib.get_backend()}")
        
        # Try different backends
        backends_to_try = ['Agg', 'TkAgg', 'Qt5Agg']
        working_backend = None
        
        for backend in backends_to_try:
            try:
                matplotlib.use(backend, force=True)
                import matplotlib.pyplot as plt
                
                # Test simple plot
                plt.figure(figsize=(2, 2))
                plt.plot([1, 2, 3], [1, 4, 2])
                plt.savefig('test_plot.png')
                plt.close()
                
                print(f"  ✅ {backend} backend works")
                working_backend = backend
                
                # Clean up
                if os.path.exists('test_plot.png'):
                    os.remove('test_plot.png')
                break
                
            except Exception as e:
                print(f"  ❌ {backend} backend failed: {e}")
        
        if working_backend:
            print(f"  🎯 Recommended backend: {working_backend}")
        else:
            print("  ⚠️ No working matplotlib backend found")
            
    except Exception as e:
        print(f"  💥 Matplotlib test failed: {e}")

def test_project_files():
    """Test if all project files exist."""
    print("\n📁 Testing project files...")
    
    required_files = [
        'config.py',
        'utils.py', 
        'eco_route_env.py',
        'baselines.py',
        'train_agent.py',
        'evaluate.py',
        'visualize.py'
    ]
    
    missing_files = []
    
    for filename in required_files:
        if os.path.exists(filename):
            print(f"  ✅ {filename}")
            
            # Test if file can be imported
            try:
                module_name = filename.replace('.py', '')
                exec(f'import {module_name}')
                print(f"    ✅ Import successful")
            except Exception as e:
                print(f"    ❌ Import failed: {e}")
                
        else:
            print(f"  ❌ {filename} (missing)")
            missing_files.append(filename)
    
    return missing_files

def test_environment_creation():
    """Test creating the EcoRoute environment."""
    print("\n🏗️ Testing environment creation...")
    
    try:
        # Test config import
        from config import ENV_CONFIG
        print("  ✅ Config imported")
        
        # Test utils import
        from utils import create_road_network
        print("  ✅ Utils imported")
        
        # Test network creation
        graph = create_road_network(5, seed=42)  # Small network for testing
        print(f"  ✅ Network created: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        
        # Test environment creation
        from eco_route_env import EcoRouteEnv
        env = EcoRouteEnv({'num_customers': 4, 'max_steps': 10, 'seed': 42})
        print("  ✅ Environment created")
        
        # Test reset
        obs, info = env.reset()
        print(f"  ✅ Environment reset: obs shape {obs.shape}")
        
        # Test step
        action = 0
        obs, reward, done, truncated, info = env.step(action)
        print(f"  ✅ Environment step: reward={reward:.2f}")
        
    except Exception as e:
        print(f"  ❌ Environment test failed: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_baselines():
    """Test baseline algorithms."""
    print("\n🔄 Testing baseline algorithms...")
    
    try:
        from eco_route_env import EcoRouteEnv
        from baselines import run_all_baselines
        
        # Create small environment for testing
        env = EcoRouteEnv({'num_customers': 4, 'max_steps': 10, 'seed': 42})
        
        # Test baselines
        results = run_all_baselines(env.graph, env.warehouse, env.customers)
        print(f"  ✅ Baselines completed: {len(results)} methods tested")
        
        success_count = sum(1 for r in results.values() if r['success'])
        print(f"  📊 Successful methods: {success_count}/{len(results)}")
        
    except Exception as e:
        print(f"  ❌ Baseline test failed: {e}")
        traceback.print_exc()
        return False
    
    return True

def suggest_fixes(failed_imports, missing_files, env_works, baselines_work):
    """Suggest fixes for identified issues."""
    print("\n🔧 SUGGESTED FIXES:")
    print("=" * 40)
    
    if failed_imports:
        print("📦 Install missing packages:")
        for package in failed_imports:
            if package == 'stable_baselines3':
                print(f"  pip install stable-baselines3[extra]")
            else:
                print(f"  pip install {package}")
    
    if missing_files:
        print("📁 Create missing files:")
        for filename in missing_files:
            print(f"  Create {filename} from the provided code")
    
    if not env_works:
        print("🏗️ Environment issues:")
        print("  - Check config.py has correct ENV_CONFIG")
        print("  - Verify utils.py has create_road_network function")
        print("  - Check eco_route_env.py imports")
    
    if not baselines_work:
        print("🔄 Baseline issues:")
        print("  - Check baselines.py implementation")
        print("  - Verify networkx functions work correctly")
    
    print("\n💡 General fixes:")
    print("  - Use 'matplotlib.use(\"Agg\")' for headless systems")
    print("  - Run 'python diagnostic.py' to retest")
    print("  - Check Python version >= 3.8")

def main():
    """Run complete diagnostic."""
    print("🔬 ECOROUTERL DIAGNOSTIC TOOL")
    print("=" * 50)
    print(f"🐍 Python version: {sys.version}")
    print(f"📂 Working directory: {os.getcwd()}")
    print()
    
    # Run tests
    failed_imports = test_imports()
    test_matplotlib()
    missing_files = test_project_files()
    env_works = test_environment_creation()
    baselines_work = test_baselines()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    total_issues = len(failed_imports) + len(missing_files)
    if not env_works:
        total_issues += 1
    if not baselines_work:
        total_issues += 1
    
    if total_issues == 0:
        print("🎉 All tests passed! EcoRouteRL should work correctly.")
        print("   Try running: python run_complete.py")
    else:
        print(f"⚠️ Found {total_issues} issues that need fixing.")
        suggest_fixes(failed_imports, missing_files, env_works, baselines_work)

if __name__ == "__main__":
    main()
