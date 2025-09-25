"""Continuous training system for EcoRouteRL with automatic resuming."""

import os
import time
import pickle
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize
from eco_route_env import EcoRouteEnv
from config import TRAINING_CONFIG, ENV_CONFIG
import matplotlib.pyplot as plt
import json

class ContinuousTrainingCallback(BaseCallback):
    """Advanced callback for continuous training monitoring."""
    
    def __init__(self, save_freq=10000, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.episode_rewards = []
        self.episode_emissions = []
        self.episode_times = []
        self.episode_distances = []
        self.episode_success_rates = []
        self.best_mean_reward = -np.inf
        self.training_start_time = time.time()
        
        # Create training log file
        self.log_file = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.performance_data = []
    
    def _on_training_start(self):
        """Called before training starts."""
        with open(self.log_file, 'w') as f:
            f.write("🚀 EcoRouteRL Continuous Training Log\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n")
    
    def _on_step(self) -> bool:
        """Called at each training step."""
        # Collect episode data
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    episode_reward = info['episode']['r']
                    self.episode_rewards.append(episode_reward)
                    
                    # Extract custom metrics if available
                    if 'cumulative_metrics' in info:
                        metrics = info['cumulative_metrics']
                        self.episode_emissions.append(metrics.get('emission', 0))
                        self.episode_times.append(metrics.get('time', 0))
                        self.episode_distances.append(metrics.get('distance', 0))
                    
                    # Check if new best performance
                    if len(self.episode_rewards) >= 100:  # Need enough episodes for average
                        mean_reward = np.mean(self.episode_rewards[-100:])
                        if mean_reward > self.best_mean_reward:
                            self.best_mean_reward = mean_reward
                            self._save_best_model()
                            self._log_milestone(f"🏆 New best mean reward: {mean_reward:.2f}")
        
        # Periodic saves and logging
        if self.n_calls % self.save_freq == 0:
            self._save_training_state()
            self._update_live_dashboard()
            self._log_progress()
        
        return True
    
    def _save_best_model(self):
        """Save the best performing model."""
        self.model.save("models/ecorouterl_best")
        
    def _save_training_state(self):
        """Save current training state for resuming."""
        training_state = {
            'episode_rewards': self.episode_rewards,
            'episode_emissions': self.episode_emissions,
            'episode_times': self.episode_times,
            'episode_distances': self.episode_distances,
            'best_mean_reward': self.best_mean_reward,
            'total_timesteps': self.n_calls,
            'training_time': time.time() - self.training_start_time
        }
        
        with open('models/training_state.pkl', 'wb') as f:
            pickle.dump(training_state, f)
    
    def _update_live_dashboard(self):
        """Update live performance metrics for dashboard."""
        if len(self.episode_rewards) > 0:
            recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
            recent_emissions = self.episode_emissions[-100:] if len(self.episode_emissions) >= 100 else self.episode_emissions
            
            dashboard_data = {
                'timestamp': datetime.now().isoformat(),
                'total_timesteps': self.n_calls,
                'total_episodes': len(self.episode_rewards),
                'mean_reward_100': np.mean(recent_rewards) if recent_rewards else 0,
                'mean_emission_100': np.mean(recent_emissions) if recent_emissions else 0,
                'best_mean_reward': self.best_mean_reward,
                'training_hours': (time.time() - self.training_start_time) / 3600,
                'recent_rewards': recent_rewards[-20:],  # Last 20 for real-time plot
                'recent_emissions': recent_emissions[-20:] if recent_emissions else []
            }
            
            # Save for live dashboard
            with open('models/live_metrics.json', 'w') as f:
                json.dump(dashboard_data, f)
    
    def _log_progress(self):
        """Log training progress."""
        if len(self.episode_rewards) > 0:
            recent_mean = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
            recent_emission = np.mean(self.episode_emissions[-100:]) if len(self.episode_emissions) >= 100 else np.mean(self.episode_emissions) if self.episode_emissions else 0
            
            log_msg = f"Step {self.n_calls:,} | Episodes: {len(self.episode_rewards)} | Mean Reward: {recent_mean:.2f} | Mean CO₂: {recent_emission:.2f}g"
            print(f"📊 {log_msg}")
            
            with open(self.log_file, 'a') as f:
                f.write(f"{datetime.now().strftime('%H:%M:%S')} - {log_msg}\n")
    
    def _log_milestone(self, message):
        """Log important milestones."""
        print(f"🎯 {message}")
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")

class ContinuousTrainer:
    """Manages continuous training with resuming capability."""
    
    def __init__(self, config=None):
        self.config = config or TRAINING_CONFIG
        self.env_config = ENV_CONFIG
        self.model = None
        self.env = None
        self.training_state = None
        
    def load_existing_training(self):
        """Load existing model and training state."""
        model_path = "models/ecorouterl_ppo.zip"
        state_path = "models/training_state.pkl"
        
        if os.path.exists(model_path) and os.path.exists(state_path):
            print("🔄 Resuming from existing training...")
            
            # Load model
            self.model = PPO.load(model_path)
            
            # Load training state
            with open(state_path, 'rb') as f:
                self.training_state = pickle.load(f)
            
            print(f"📈 Loaded model with {len(self.training_state['episode_rewards'])} episodes")
            print(f"🏆 Previous best reward: {self.training_state['best_mean_reward']:.2f}")
            print(f"⏱️ Total training time: {self.training_state['training_time']/3600:.1f} hours")
            
            return True
        
        return False
    
    def create_fresh_training(self):
        """Create new training setup."""
        print("🚀 Starting fresh training...")
        
        # Create environment
        def make_env():
            return EcoRouteEnv(self.env_config)
        
        self.env = make_vec_env(make_env, n_envs=4)
        self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True, gamma=self.config['gamma'])
        
        # Create model
        self.model = PPO(
            'MlpPolicy',
            self.env,
            learning_rate=self.config['learning_rate'],
            n_steps=2048,
            batch_size=self.config['batch_size'],
            n_epochs=self.config['n_epochs'],
            gamma=self.config['gamma'],
            gae_lambda=self.config['gae_lambda'],
            clip_range=self.config['clip_range'],
            verbose=1,
            tensorboard_log=self.config['tensorboard_log']
        )
    
    def train_continuously(self, total_additional_timesteps=None):
        """Train continuously with automatic saving."""
        if total_additional_timesteps is None:
            total_additional_timesteps = self.config['total_timesteps']
        
        # Setup callbacks
        continuous_callback = ContinuousTrainingCallback(save_freq=10000)
        
        checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path='./models/checkpoints/',
            name_prefix='ecorouterl_checkpoint'
        )
        
        # Create evaluation environment
        eval_env = make_vec_env(lambda: EcoRouteEnv(self.env_config), n_envs=1)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path='./models/',
            log_path='./logs/',
            eval_freq=25000,
            deterministic=True,
            render=False
        )
        
        print(f"🎯 Training for {total_additional_timesteps:,} additional timesteps...")
        print("💡 Press Ctrl+C to safely stop and save progress")
        
        try:
            self.model.learn(
                total_timesteps=total_additional_timesteps,
                callback=[continuous_callback, checkpoint_callback, eval_callback],
                progress_bar=True,
                reset_num_timesteps=False  # Don't reset timestep counter
            )
            
        except KeyboardInterrupt:
            print("\n⏸️ Training interrupted by user")
            print("💾 Saving current progress...")
            
        finally:
            # Always save at the end
            self.model.save("models/ecorouterl_ppo")
            if self.env:
                self.env.save("models/vec_normalize.pkl")
            
            print("✅ Training progress saved!")
            print(f"📊 Total episodes completed: {len(continuous_callback.episode_rewards)}")
            
            return continuous_callback
    
    def run_adaptive_training(self, target_performance=None, max_hours=24):
        """Run training until target performance or time limit."""
        print("🎯 Starting adaptive training...")
        print(f"🎪 Target: {target_performance if target_performance else 'Time limited'}")
        print(f"⏰ Max duration: {max_hours} hours")
        
        start_time = time.time()
        training_cycles = 0
        
        while True:
            training_cycles += 1
            print(f"\n🔄 Training Cycle {training_cycles}")
            
            # Train for one cycle (50K timesteps)
            callback = self.train_continuously(total_additional_timesteps=50000)
            
            # Check if target reached
            if target_performance and callback.best_mean_reward >= target_performance:
                print(f"🎉 Target performance reached: {callback.best_mean_reward:.2f}")
                break
            
            # Check time limit
            elapsed_hours = (time.time() - start_time) / 3600
            if elapsed_hours >= max_hours:
                print(f"⏰ Time limit reached: {elapsed_hours:.1f} hours")
                break
            
            print(f"📈 Current best: {callback.best_mean_reward:.2f}, Target: {target_performance}")
            print(f"⏱️ Elapsed: {elapsed_hours:.1f}h / {max_hours}h")

def main():
    """Main training function."""
    print("🌱 EcoRouteRL Continuous Training System")
    print("="*50)
    
    # Create directories
    os.makedirs('models/checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    trainer = ContinuousTrainer()
    
    # Try to resume existing training
    if trainer.load_existing_training():
        print("\n🎮 Training Options:")
        print("1. Continue training (additional 100K timesteps)")
        print("2. Adaptive training (until target performance)")
        print("3. Extended training (custom timesteps)")
        
        choice = input("\nChoose option (1-3): ").strip()
        
        if choice == "2":
            target = float(input("Enter target mean reward (e.g., 500): "))
            trainer.run_adaptive_training(target_performance=target)
        elif choice == "3":
            timesteps = int(input("Enter additional timesteps (e.g., 200000): "))
            trainer.train_continuously(total_additional_timesteps=timesteps)
        else:
            trainer.train_continuously()  # Default: 100K more
    else:
        # Fresh training
        trainer.create_fresh_training()
        trainer.train_continuously()
    
    print("\n🎉 Training completed!")
    print("📊 Check models/live_metrics.json for latest performance data")

if __name__ == "__main__":
    main()