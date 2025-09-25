"""Training script for RL agent."""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
import matplotlib.pyplot as plt
from eco_route_env import EcoRouteEnv
from config import TRAINING_CONFIG, ENV_CONFIG
import pickle

class MetricsCallback(BaseCallback):
    """Custom callback to track training metrics."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_emissions = []
        self.episode_times = []
        self.episode_distances = []
        self.success_rates = []
    
    def _on_step(self) -> bool:
        # Collect episode statistics
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    episode_info = info['episode']
                    self.episode_rewards.append(episode_info['r'])
                
                # Track cumulative metrics if available
                if 'cumulative_metrics' in info:
                    metrics = info['cumulative_metrics']
                    self.episode_emissions.append(metrics['emission'])
                    self.episode_times.append(metrics['time'])
                    self.episode_distances.append(metrics['distance'])
        
        return True

def train_agent():
    """Train the RL agent."""
    print("🚀 Starting EcoRouteRL agent training...")
    
    # Create training environment
    def make_env():
        return EcoRouteEnv(ENV_CONFIG)
    
    # Create vectorized environment
    env = make_vec_env(make_env, n_envs=4)
    
    # Wrap with VecNormalize for better training stability
    env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=TRAINING_CONFIG['gamma'])
    
    # Create evaluation environment
    eval_env = make_vec_env(make_env, n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
    
    # Initialize PPO agent
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=TRAINING_CONFIG['learning_rate'],
        n_steps=2048,
        batch_size=TRAINING_CONFIG['batch_size'],
        n_epochs=TRAINING_CONFIG['n_epochs'],
        gamma=TRAINING_CONFIG['gamma'],
        gae_lambda=TRAINING_CONFIG['gae_lambda'],
        clip_range=TRAINING_CONFIG['clip_range'],
        verbose=TRAINING_CONFIG['verbose'],
        tensorboard_log=TRAINING_CONFIG['tensorboard_log']
    )
    
    # Create callbacks
    metrics_callback = MetricsCallback()
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/',
        log_path='./logs/',
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # Train the model
    print(f"Training for {TRAINING_CONFIG['total_timesteps']} timesteps...")
    model.learn(
        total_timesteps=TRAINING_CONFIG['total_timesteps'],
        callback=[metrics_callback, eval_callback],
        progress_bar=True
    )
    
    # Save the trained model and normalization parameters
    model.save("models/ecorouterl_ppo")
    env.save("models/vec_normalize.pkl")
    
    # Save training metrics
    training_data = {
        'episode_rewards': metrics_callback.episode_rewards,
        'episode_emissions': metrics_callback.episode_emissions,
        'episode_times': metrics_callback.episode_times,
        'episode_distances': metrics_callback.episode_distances
    }
    
    with open('models/training_metrics.pkl', 'wb') as f:
        pickle.dump(training_data, f)
    
    print("✅ Training completed!")
    print(f"📊 Model saved to models/ecorouterl_ppo")
    print(f"📈 View training progress with: tensorboard --logdir {TRAINING_CONFIG['tensorboard_log']}")
    
    # Plot training curves
    plot_training_curves(training_data)
    
    return model, env

def plot_training_curves(training_data):
    """Plot training progress curves."""
    if not training_data['episode_rewards']:
        print("No training data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot rewards
    axes[0, 0].plot(training_data['episode_rewards'])
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True)
    
    # Plot emissions
    if training_data['episode_emissions']:
        axes[0, 1].plot(training_data['episode_emissions'])
        axes[0, 1].set_title('Episode Emissions')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Total CO₂ (g)')
        axes[0, 1].grid(True)
    
    # Plot times
    if training_data['episode_times']:
        axes[1, 0].plot(training_data['episode_times'])
        axes[1, 0].set_title('Episode Times')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Total Time (min)')
        axes[1, 0].grid(True)
    
    # Plot distances
    if training_data['episode_distances']:
        axes[1, 1].plot(training_data['episode_distances'])
        axes[1, 1].set_title('Episode Distances')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Total Distance (km)')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("📊 Training curves saved to models/training_curves.png")

if __name__ == "__main__":
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('tensorboard_logs', exist_ok=True)
    
    # Train the agent
    model, env = train_agent()
