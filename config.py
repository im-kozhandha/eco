"""Configuration parameters for EcoRouteRL."""

import numpy as np

# Environment Configuration
ENV_CONFIG = {
    'num_customers': 12,  # Excluding warehouse
    'grid_size': 5,       # For grid network generation
    'max_steps': 20,      # Maximum steps per episode
    'seed': 42,           # Reproducible seed
}

# Vehicle Configuration
VEHICLE_CONFIG = {
    'petrol_4w': {
        'emission_factor': 180,  # gCO2/km
        'avg_speed': 40,         # km/h
        'name': '4W Petrol'
    },
    'electric': {
        'emission_factor': 50,   # gCO2/km (including electricity generation)
        'avg_speed': 35,         # km/h
        'name': 'Electric'
    }
}

# Reward Configuration
REWARD_CONFIG = {
    # Step rewards weights
    'w_distance': 0.2,
    'w_time': 0.3,
    'w_emission': 0.5,
    
    # Terminal rewards
    'success_reward': 1000,
    'failure_penalty': -500,
    'ontime_bonus': 200,
    'beta_emission': 0.001,  # Terminal emission penalty
    'gamma_time': 10,        # Terminal time penalty
    
    # Normalization references
    'distance_ref': 100,     # km
    'time_ref': 60,          # minutes
    'max_emission': 1000,    # gCO2
}

# Training Configuration
TRAINING_CONFIG = {
    'total_timesteps': 100000,
    'learning_rate': 3e-4,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'verbose': 1,
    'tensorboard_log': "./tensorboard_logs/"
}

# Network Generation
NETWORK_CONFIG = {
    'distance_range': (5, 25),    # km
    'traffic_multiplier_range': (0.8, 1.5),  # Speed variation
    'time_of_day_hours': 8,       # Working hours
}