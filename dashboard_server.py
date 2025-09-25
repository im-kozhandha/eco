# dashboard_server.py - Live Dashboard Integration
"""Real-time dashboard server for EcoRouteRL with live agent data."""

import json
import time
import threading
from datetime import datetime
import numpy as np
from flask import Flask, render_template_string, jsonify
from flask_socketio import SocketIO, emit
import os
from eco_route_env import EcoRouteEnv
from config import ENV_CONFIG
from stable_baselines3 import PPO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ecorouterl-dashboard'
socketio = SocketIO(app, cors_allowed_origins="*")

class LiveDashboard:
    """Real-time dashboard data manager."""
    
    def __init__(self):
        self.is_training = False
        self.current_episode = 1
        self.agent_position = {"x": 50, "y": 50, "current_node": 0}
        self.visited_nodes = set([0])
        self.network_data = []
        self.episode_metrics = {
            "reward": 0,
            "emission": 0,
            "distance": 0,
            "time": 0,
            "step": 0
        }
        self.training_history = []
        self.realtime_metrics = []
        self.best_reward = 0
        self.env = None
        self.model = None
        self.current_obs = None
        
        self.setup_environment()
    
    def setup_environment(self):
        """Initialize environment and load model if available."""
        self.env = EcoRouteEnv(ENV_CONFIG)
        
        # Generate network data for frontend
        self.network_data = []
        pos_scale = 90  # Scale positions to fit 0-100 range
        
        for node_id, data in self.env.graph.nodes(data=True):
            if 'pos' in data:
                x, y = data['pos']
                # Normalize positions to 10-90 range (avoid edges)
                norm_x = 10 + (x / 100) * pos_scale
                norm_y = 10 + (y / 100) * pos_scale
            else:
                # Random positions if not set
                norm_x = 10 + np.random.random() * pos_scale
                norm_y = 10 + np.random.random() * pos_scale
            
            self.network_data.append({
                "id": node_id,
                "x": norm_x,
                "y": norm_y,
                "type": "warehouse" if node_id == 0 else "customer",
                "visited": node_id in self.visited_nodes
            })
        
        # Try to load trained model
        try:
            if os.path.exists("models/ecorouterl_ppo.zip"):
                self.model = PPO.load("models/ecorouterl_ppo")
                print("✅ Loaded trained model for live demo")
            else:
                print("⚠️ No trained model found - using random agent")
        except Exception as e:
            print(f"⚠️ Model loading failed: {e}")
    
    def start_training_simulation(self):
        """Start live training simulation."""
        if self.is_training:
            return
        
        self.is_training = True
        self.current_obs, _ = self.env.reset()
        self.reset_episode()
        
        # Start simulation thread
        training_thread = threading.Thread(target=self._training_loop)
        training_thread.daemon = True
        training_thread.start()
    
    def stop_training_simulation(self):
        """Stop training simulation."""
        self.is_training = False
    
    def reset_simulation(self):
        """Reset the entire simulation."""
        self.stop_training_simulation()
        time.sleep(0.5)  # Let training loop finish
        
        self.current_episode = 1
        self.visited_nodes = set([0])
        self.agent_position = {"x": 50, "y": 50, "current_node": 0}
        self.episode_metrics = {
            "reward": 0, "emission": 0, "distance": 0, "time": 0, "step": 0
        }
        self.realtime_metrics = []
        
        # Update network data
        for node in self.network_data:
            node['visited'] = node['id'] in self.visited_nodes
        
        # Reset environment
        if self.env:
            self.current_obs, _ = self.env.reset()
    
    def reset_episode(self):
        """Reset current episode."""
        self.visited_nodes = set([0])
        self.agent_position = {"x": self.network_data[0]["x"], 
                              "y": self.network_data[0]["y"], 
                              "current_node": 0}
        self.episode_metrics = {"reward": 0, "emission": 0, "distance": 0, "time": 0, "step": 0}
        
        # Update network visited status
        for node in self.network_data:
            node['visited'] = node['id'] in self.visited_nodes
    
    def _training_loop(self):
        """Main training simulation loop."""
        while self.is_training:
            try:
                # Get unvisited customer nodes
                unvisited_customers = [node for node in self.network_data 
                                     if node['type'] == 'customer' and not node['visited']]
                
                if not unvisited_customers:
                    # Episode complete
                    self._complete_episode()
                    time.sleep(1)
                    continue
                
                # Choose next action
                if self.model and self.current_obs is not None:
                    # Use trained model
                    action, _ = self.model.predict(self.current_obs, deterministic=True)
                    
                    # Verify action is valid
                    valid_actions = np.where(self.env.unserved_customers)[0]
                    if action not in valid_actions and len(valid_actions) > 0:
                        action = np.random.choice(valid_actions)
                else:
                    # Random action for demo
                    action = np.random.randint(0, len(unvisited_customers))
                    if action < len(self.env.customers):
                        target_customer_id = self.env.customers[action]
                        if target_customer_id in self.visited_nodes:
                            # Find an actually unvisited customer
                            unvisited_ids = [node['id'] for node in unvisited_customers]
                            if unvisited_ids:
                                target_customer_id = np.random.choice(unvisited_ids)
                                action = self.env.customers.index(target_customer_id)
                
                # Execute action in environment
                if self.env and action < len(self.env.customers):
                    target_node_id = self.env.customers[action]
                    
                    # Take step in environment
                    obs, reward, done, truncated, info = self.env.step(action)
                    self.current_obs = obs
                    
                    # Update agent position
                    target_node = next(node for node in self.network_data if node['id'] == target_node_id)
                    self.agent_position = {
                        "x": target_node['x'], 
                        "y": target_node['y'],
                        "current_node": target_node_id
                    }
                    
                    # Mark as visited
                    self.visited_nodes.add(target_node_id)
                    for node in self.network_data:
                        if node['id'] == target_node_id:
                            node['visited'] = True
                    
                    # Update metrics
                    self.episode_metrics["reward"] += reward
                    self.episode_metrics["step"] += 1
                    
                    if 'cumulative_metrics' in info:
                        cum_metrics = info['cumulative_metrics']
                        self.episode_metrics["emission"] = cum_metrics.get('emission', 0)
                        self.episode_metrics["distance"] = cum_metrics.get('distance', 0)
                        self.episode_metrics["time"] = cum_metrics.get('time', 0)
                    
                    # Add to realtime data
                    self.realtime_metrics.append({
                        "step": self.episode_metrics["step"],
                        "reward": reward,
                        "emission": self.episode_metrics["emission"],
                        "co2Rate": self.episode_metrics["emission"] / max(self.episode_metrics["distance"], 1)
                    })
                    
                    # Keep only last 20 points for realtime chart
                    self.realtime_metrics = self.realtime_metrics[-20:]
                    
                    # Emit update to dashboard
                    self._emit_update()
                
                # Wait between steps
                time.sleep(0.8)
                
            except Exception as e:
                print(f"Training loop error: {e}")
                time.sleep(1)
    
    def _complete_episode(self):
        """Complete current episode and start new one."""
        # Update best reward
        episode_reward = self.episode_metrics["reward"]
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
        
        # Add to training history
        self.training_history.append({
            "episode": self.current_episode,
            "reward": episode_reward,
            "emission": self.episode_metrics["emission"],
            "distance": self.episode_metrics["distance"],
            "time": self.episode_metrics["time"],
            "success": len(self.visited_nodes) > len([n for n in self.network_data if n['type'] == 'customer']) * 0.8
        })
        
        # Keep only last 100 episodes
        self.training_history = self.training_history[-100:]
        
        # Reset for next episode
        self.current_episode += 1
        if self.env:
            self.current_obs, _ = self.env.reset()
        self.reset_episode()
        
        self._emit_update()
    
    def _emit_update(self):
        """Emit live update to dashboard."""
        data = self.get_dashboard_data()
        socketio.emit('dashboard_update', data)
    
    def get_dashboard_data(self):
        """Get complete dashboard data."""
        return {
            "agent_position": self.agent_position,
            "visited_nodes": list(self.visited_nodes),
            "network_nodes": self.network_data,
            "episode_metrics": self.episode_metrics,
            "training_history": self.training_history[-20:],  # Last 20 episodes
            "realtime_metrics": self.realtime_metrics,
            "best_reward": self.best_reward,
            "current_episode": self.current_episode,
            "is_training": self.is_training
        }

# Global dashboard instance
dashboard = LiveDashboard()

# Flask routes
@app.route('/')
def index():
    """Serve the dashboard HTML."""
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>EcoRouteRL Live Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: 'Courier New', monospace;
            background: linear-gradient(45deg, #000000, #1a1a1a);
            color: #22c55e;
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }
        .dashboard {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            height: 90vh;
        }
        .panel {
            background: rgba(0, 0, 0, 0.7);
            border: 2px solid #22c55e;
            border-radius: 20px;
            padding: 20px;
            backdrop-filter: blur(10px);
        }
        .network-viz {
            position: relative;
            height: 400px;
            background: radial-gradient(circle, #1a4a3a 0%, #000000 100%);
            border-radius: 15px;
            overflow: hidden;
        }
        .node {
            position: absolute;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            font-weight: bold;
            transition: all 0.5s ease;
        }
        .warehouse {
            background: #eab308;
            color: #000;
            box-shadow: 0 0 20px #eab308;
        }
        .customer {
            background: #6b7280;
            color: #fff;
        }
        .customer.visited {
            background: #22c55e;
            box-shadow: 0 0 15px #22c55e;
        }
        .agent {
            position: absolute;
            width: 15px;
            height: 15px;
            background: #ef4444;
            border-radius: 50%;
            box-shadow: 0 0 20px #ef4444;
            transition: all 0.8s ease;
            z-index: 10;
        }
        .controls {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .btn {
            background: #22c55e;
            color: #000;
            border: none;
            padding: 10px 20px;
            border-radius: 10px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s;
        }
        .btn:hover {
            background: #16a34a;
            transform: scale(1.05);
        }
        .btn.pause {
            background: #eab308;
        }
        .metrics {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin: 20px 0;
        }
        .metric {
            background: rgba(34, 197, 94, 0.1);
            border: 1px solid #22c55e;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .metric-label {
            font-size: 12px;
            opacity: 0.8;
        }
        .chart-container {
            height: 200px;
            margin: 20px 0;
        }
        .status {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 20px;
        }
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #6b7280;
        }
        .status-dot.active {
            background: #22c55e;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🌱 EcoRouteRL Live Dashboard</h1>
        <div class="status">
            <div id="statusDot" class="status-dot"></div>
            <span id="statusText">Standby</span>
        </div>
    </div>
    
    <div class="dashboard">
        <div class="panel">
            <h2>🗺️ Live Agent Movement</h2>
            <div class="controls">
                <button id="startBtn" class="btn" onclick="toggleTraining()">▶️ Start Training</button>
                <button id="resetBtn" class="btn" onclick="resetSimulation()">🔄 Reset</button>
            </div>
            
            <div class="network-viz" id="networkViz">
                <div id="agent" class="agent"></div>
            </div>
            
            <div class="metrics">
                <div class="metric">
                    <div id="rewardValue" class="metric-value">0</div>
                    <div class="metric-label">Episode Reward</div>
                </div>
                <div class="metric">
                    <div id="emissionValue" class="metric-value">0</div>
                    <div class="metric-label">CO₂ Emissions (g)</div>
                </div>
                <div class="metric">
                    <div id="distanceValue" class="metric-value">0</div>
                    <div class="metric-label">Distance (km)</div>
                </div>
                <div class="metric">
                    <div id="stepValue" class="metric-value">0</div>
                    <div class="metric-label">Steps</div>
                </div>
            </div>
        </div>
        
        <div class="panel">
            <h2>📊 Training Progress</h2>
            <div class="chart-container" id="rewardChart"></div>
            
            <h3>🌱 Sustainability Metrics</h3>
            <div class="chart-container" id="emissionChart"></div>
            
            <div class="metric">
                <div id="bestReward" class="metric-value">0</div>
                <div class="metric-label">🏆 Best Episode Reward</div>
            </div>
            
            <div class="metric">
                <div id="episodeCount" class="metric-value">1</div>
                <div class="metric-label">📈 Current Episode</div>
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        let isTraining = false;
        let networkNodes = [];
        
        // Connect to server
        socket.on('connect', function() {
            console.log('Connected to dashboard server');
            socket.emit('get_initial_data');
        });
        
        // Receive dashboard updates
        socket.on('dashboard_update', function(data) {
            updateDashboard(data);
        });
        
        function toggleTraining() {
            const btn = document.getElementById('startBtn');
            const statusDot = document.getElementById('statusDot');
            const statusText = document.getElementById('statusText');
            
            if (isTraining) {
                socket.emit('stop_training');
                btn.innerHTML = '▶️ Start Training';
                btn.className = 'btn';
                statusDot.className = 'status-dot';
                statusText.textContent = 'Standby';
                isTraining = false;
            } else {
                socket.emit('start_training');
                btn.innerHTML = '⏸️ Pause Training';
                btn.className = 'btn pause';
                statusDot.className = 'status-dot active';
                statusText.textContent = 'Training Active';
                isTraining = true;
            }
        }
        
        function resetSimulation() {
            socket.emit('reset_simulation');
            document.getElementById('rewardValue').textContent = '0';
            document.getElementById('emissionValue').textContent = '0';
            document.getElementById('distanceValue').textContent = '0';
            document.getElementById('stepValue').textContent = '0';
        }
        
        function updateDashboard(data) {
            // Update agent position
            const agent = document.getElementById('agent');
            const agentPos = data.agent_position;
            agent.style.left = agentPos.x + '%';
            agent.style.top = agentPos.y + '%';
            
            // Update metrics
            const metrics = data.episode_metrics;
            document.getElementById('rewardValue').textContent = metrics.reward.toFixed(1);
            document.getElementById('emissionValue').textContent = metrics.emission.toFixed(1);
            document.getElementById('distanceValue').textContent = metrics.distance.toFixed(1);
            document.getElementById('stepValue').textContent = metrics.step;
            document.getElementById('bestReward').textContent = data.best_reward.toFixed(0);
            document.getElementById('episodeCount').textContent = data.current_episode;
            
            // Update network nodes
            updateNetworkNodes(data.network_nodes, data.visited_nodes);
            
            // Update charts
            updateCharts(data.training_history, data.realtime_metrics);
            
            // Update training status
            isTraining = data.is_training;
        }
        
        function updateNetworkNodes(nodes, visitedIds) {
            const viz = document.getElementById('networkViz');
            
            // Clear existing nodes
            const existingNodes = viz.querySelectorAll('.node');
            existingNodes.forEach(node => node.remove());
            
            // Add nodes
            nodes.forEach(node => {
                const nodeEl = document.createElement('div');
                nodeEl.className = `node ${node.type}${visitedIds.includes(node.id) ? ' visited' : ''}`;
                nodeEl.style.left = node.x + '%';
                nodeEl.style.top = node.y + '%';
                nodeEl.textContent = node.id;
                viz.appendChild(nodeEl);
            });
        }
        
        function updateCharts(trainingData, realtimeData) {
            // Reward chart
            if (trainingData.length > 0) {
                const rewardTrace = {
                    x: trainingData.map(d => d.episode),
                    y: trainingData.map(d => d.reward),
                    type: 'scatter',
                    mode: 'lines',
                    line: { color: '#22c55e', width: 2 },
                    name: 'Episode Reward'
                };
                
                Plotly.newPlot('rewardChart', [rewardTrace], {
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    font: { color: '#22c55e' },
                    xaxis: { color: '#22c55e' },
                    yaxis: { color: '#22c55e' },
                    margin: { l: 40, r: 20, t: 20, b: 40 }
                });
            }
            
            // Emission chart
            if (realtimeData.length > 0) {
                const emissionTrace = {
                    x: realtimeData.map(d => d.step),
                    y: realtimeData.map(d => d.co2Rate),
                    type: 'scatter',
                    mode: 'lines',
                    line: { color: '#eab308', width: 2 },
                    name: 'CO₂ Rate'
                };
                
                Plotly.newPlot('emissionChart', [emissionTrace], {
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    font: { color: '#eab308' },
                    xaxis: { color: '#eab308' },
                    yaxis: { color: '#eab308' },
                    margin: { l: 40, r: 20, t: 20, b: 40 }
                });
            }
        }
        
        // Initialize
        socket.emit('get_initial_data');
    </script>
</body>
</html>
"""
    return html_template

@app.route('/api/data')
def api_data():
    """API endpoint for dashboard data."""
    return jsonify(dashboard.get_dashboard_data())

# Socket.IO events
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print('Dashboard client connected')
    emit('dashboard_update', dashboard.get_dashboard_data())

@socketio.on('get_initial_data')
def handle_get_initial_data():
    """Send initial dashboard data."""
    emit('dashboard_update', dashboard.get_dashboard_data())

@socketio.on('start_training')
def handle_start_training():
    """Start training simulation."""
    dashboard.start_training_simulation()
    print('Training simulation started')

@socketio.on('stop_training')
def handle_stop_training():
    """Stop training simulation."""
    dashboard.stop_training_simulation()
    print('Training simulation stopped')

@socketio.on('reset_simulation')
def handle_reset_simulation():
    """Reset simulation."""
    dashboard.reset_simulation()
    emit('dashboard_update', dashboard.get_dashboard_data())
    print('Simulation reset')

if __name__ == '__main__':
    print("🚀 Starting EcoRouteRL Live Dashboard Server...")
    print("📊 Dashboard available at: http://localhost:5000")
    print("🌱 Features:")
    print("  • Real-time agent movement visualization")
    print("  • Live training metrics and charts") 
    print("  • Interactive controls (start/pause/reset)")
    print("  • Sustainability impact tracking")
    print("  • Futuristic green/black/yellow design")
    print()
    print("💡 Usage:")
    print("  1. Open http://localhost:5000 in your browser")
    print("  2. Click 'Start Training' to begin live simulation")
    print("  3. Watch the agent learn eco-friendly routes!")
    print("  4. Use Ctrl+C to stop the server")
    print()
    
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard server stopped")