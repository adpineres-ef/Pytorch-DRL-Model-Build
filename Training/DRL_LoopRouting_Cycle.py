import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical # For sampling actions
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import time
import os
from Solvers import generate_optimal_route_pytorch, solve_mip, solve_heuristic, solve_regret2_heuristic, solve_lp_relaxation
import warnings
warnings.filterwarnings("ignore")
summary_rows = []
for NUM_NODES in [10,15,20,25]:
    print("Nodes:",NUM_NODES)
    #set input dates
    date = "_2024-12-09"
    truck = "VAN"
    mpg = 6.5
    # Get current working directory
    cwd = "C://GitHub//Pytorch-DRL-Model-Build//Training"
    print("Current working directory:", cwd)
    # Build file paths relative to current directory
    file_name1 = os.path.join(cwd, f"rate_q2_west_{truck}{date}.csv")
    file_name2 = os.path.join(cwd, "duration_west.csv")
    file_name3 = os.path.join(cwd, f"prob_west_{truck}{date}.csv")
    file_name4 = os.path.join(cwd, f"load_av_west_{truck}{date}.csv")
    file_name5 = os.path.join(cwd, "distance_west.csv")
    file_name6 = os.path.join(cwd, f"diesel_west{date}.csv")
    file_name7 = os.path.join(cwd, "labels_west.csv")
    # Read the Excel file into a DataFrame
    rate_matrix = pd.read_csv(file_name1,header= None)
    time_matrix = pd.read_csv(file_name2,header = None)
    markov_matrix = pd.read_csv(file_name3,header=None)
    loads_matrix = pd.read_csv(file_name4,header=None)
    distance_matrix = pd.read_csv(file_name5,header=None)
    diesel_matrix = pd.read_csv(file_name6,header=None)
    hub_labels = pd.read_csv(file_name7,header=None)

    revenue_matrix = rate_matrix * distance_matrix
    revenue_matrix[loads_matrix <= 1] = 0
    diesel_matrix = diesel_matrix / mpg
    var_cost_matrix = 1.2 * distance_matrix
    cost_matrix = distance_matrix * diesel_matrix
    cost_matrix = cost_matrix + 163
    cost_matrix = cost_matrix + var_cost_matrix
    reward_matrix = revenue_matrix - cost_matrix
    time_matrix *= 0.9

    ACTION_SPACE_SIZE = NUM_NODES

    # Problem Constraints (Keep the same duration limits for consistency)
    DURATION_LIMIT = 60.0
    DURATION_TOLERANCE = 0.10
    MIN_DURATION = DURATION_LIMIT * (1 - DURATION_TOLERANCE)
    MAX_DURATION = DURATION_LIMIT * (1 + DURATION_TOLERANCE)
    BIG_M_PENALTY = -1e9 # Large negative number for rewards

    # Use a fixed seed for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    time_matrix = time_matrix.iloc[:NUM_NODES, :NUM_NODES]
    reward_matrix = reward_matrix.iloc[:NUM_NODES, :NUM_NODES]
    # Round for clarity
    time_matrix = np.round(time_matrix, 1)
    reward_matrix = np.round(reward_matrix, 0)

    # Create DataFrames for easy viewing
    time_df = pd.DataFrame(time_matrix, index=range(NUM_NODES), columns=range(NUM_NODES))
    reward_df = pd.DataFrame(reward_matrix, index=range(NUM_NODES), columns=range(NUM_NODES))
    # Apply Big M penalty to reward matrix diagonal (used by DRL and Heuristic)
    reward_matrix_penalized = reward_matrix.copy()
    reward_array = reward_matrix_penalized.to_numpy()
    np.fill_diagonal(reward_array, BIG_M_PENALTY)
    reward_matrix_penalized = pd.DataFrame(reward_array, index=reward_matrix_penalized.index, columns=reward_matrix_penalized.columns)
    # DRL Hyperparameters (Can potentially reduce episodes/steps for smaller problem)
    STATE_SIZE = 2 # (current_node_index, time_elapsed_normalized)
    LEARNING_RATE = 0.0001
    GAMMA = 0.95 # Discount factor

    EPSILON_START = 1.0
    EPSILON_END = 0.05 
    EPSILON_DECAY_STEPS = max(50000, NUM_NODES * 10000)

    BUFFER_SIZE = 10000 
    BATCH_SIZE = 32 

    NUM_EPISODES = 60000 
    MAX_STEPS_PER_EPISODE = 50
    TARGET_UPDATE_FREQ = 50 

    REWARD_SCALE_FACTOR = 100.0

    # Rewards / Penalties 
    RETURN_SUCCESS_BONUS = 1000.0 / REWARD_SCALE_FACTOR      
    TIME_VIOLATION_PENALTY = -100.0 / REWARD_SCALE_FACTOR  
    INCOMPLETE_PENALTY = -200.0 / REWARD_SCALE_FACTOR    

    # PyTorch Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    def get_qnetwork_sizes(num_nodes):
        hidden1 = max(128, num_nodes * 8)
        hidden2 = max(256, num_nodes * 8)
        hidden3 = max(64, num_nodes * 4)
        return hidden1, hidden2, hidden3

    class QNetwork(nn.Module):
        def __init__(self, state_size, action_size, num_nodes):
            super(QNetwork, self).__init__()
            h1, h2, h3 = get_qnetwork_sizes(num_nodes)
            self.fc1 = nn.Linear(state_size, h1)
            self.bn1 = nn.BatchNorm1d(h1)
            self.fc2 = nn.Linear(h1, h2)
            self.bn2 = nn.BatchNorm1d(h2)
            self.fc3 = nn.Linear(h2, h3)
            self.bn3 = nn.BatchNorm1d(h3)
            self.fc4 = nn.Linear(h3, action_size)
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.xavier_uniform_(self.fc3.weight)
            nn.init.xavier_uniform_(self.fc4.weight)

        def forward(self, state):
            x = F.relu(self.bn1(self.fc1(state)))
            x = F.relu(self.bn2(self.fc2(x)))
            x = F.relu(self.bn3(self.fc3(x)))
            return self.fc4(x)

    class DQNAgent_PyTorch:
        def __init__(self, state_size, action_size, learning_rate, gamma, buffer_size, batch_size, device, num_nodes):
            self.state_size = state_size
            self.action_size = action_size
            self.memory = deque(maxlen=buffer_size)
            self.gamma = gamma
            self.batch_size = batch_size
            self.device = device

            # Use the smaller QNetwork
            self.policy_net = QNetwork(state_size, action_size, num_nodes).to(self.device)
            self.target_net = QNetwork(state_size, action_size, num_nodes).to(self.device)
            self.update_target_model()
            self.target_net.eval()

            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
            self.loss_function = nn.MSELoss()
            self.epsilon = EPSILON_START

        def update_target_model(self):
            self.target_net.load_state_dict(self.policy_net.state_dict())

        def remember(self, state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done))

        def act(self, state, invalid_actions=None): # Added invalid_actions parameter
            """Selects an action using epsilon-greedy strategy, avoiding invalid actions."""
            if invalid_actions is None:
                invalid_actions = set()

            current_node_index = int(state[0]) # Assuming state[0] is node index

            # Add current node itself to invalid actions for this step
            current_step_invalid_actions = invalid_actions.union({current_node_index})

            possible_actions = list(range(self.action_size))
            valid_actions = [a for a in possible_actions if a not in current_step_invalid_actions]

            # If no valid actions are possible (shouldn't normally happen unless trapped)
            if not valid_actions:
                # Fallback: maybe allow returning home if that's the only invalid action?
                # Or just return a dummy action (e.g., 0) - the environment should handle this.
                # Let's return current_node to signal being stuck, though env should handle.
                # print(f"Warning: No valid actions from node {current_node_index} with invalid set {current_step_invalid_actions}")
                return current_node_index # Return current node to signal being stuck

            if random.random() <= self.epsilon:
                # Explore: Choose randomly from valid actions
                return random.choice(valid_actions)
            else:
                # Exploit: Choose the best action from Q-values among valid ones
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                self.policy_net.eval()
                with torch.no_grad():
                    q_values = self.policy_net(state_tensor)
                self.policy_net.train()

                q_values_numpy = q_values.cpu().data.numpy()[0]

                # Mask invalid actions by setting their Q-values to -infinity
                for invalid_action in current_step_invalid_actions:
                    if 0 <= invalid_action < self.action_size:
                        q_values_numpy[invalid_action] = -np.inf

                # Choose the best among the remaining valid actions
                best_action = np.argmax(q_values_numpy)

                # Sanity check if argmax still picked an invalid action (e.g., all are -inf)
                if q_values_numpy[best_action] == -np.inf:
                    # print(f"Warning: All valid actions have -inf Q-value from node {current_node_index}. Choosing randomly from valid.")
                    # Fallback to random choice among valid if exploitation leads nowhere
                    if valid_actions: # Ensure valid_actions is not empty
                        return random.choice(valid_actions)
                    else: # Truly stuck
                        return current_node_index # Signal stuck

                return best_action

        def replay(self):
            if len(self.memory) < self.batch_size:
                return 0.0
            minibatch = random.sample(self.memory, self.batch_size)
            states = torch.from_numpy(np.vstack([e[0] for e in minibatch])).float().to(self.device)
            actions = torch.from_numpy(np.vstack([e[1] for e in minibatch])).long().to(self.device)
            rewards = torch.from_numpy(np.vstack([e[2] for e in minibatch])).float().to(self.device)
            next_states = torch.from_numpy(np.vstack([e[3] for e in minibatch])).float().to(self.device)
            dones = torch.from_numpy(np.vstack([e[4] for e in minibatch]).astype(np.uint8)).float().to(self.device)

            with torch.no_grad():
                target_q_next = self.target_net(next_states)
                max_q_next = target_q_next.max(1)[0].unsqueeze(1)
                target_q_values = rewards + (self.gamma * max_q_next * (1 - dones))

            current_q_values = self.policy_net(states)
            action_q_values = current_q_values.gather(1, actions)
            loss = self.loss_function(action_q_values, target_q_values)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss.item()

        def decay_epsilon(self, current_step):
            self.epsilon = max(EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END) * (current_step / EPSILON_DECAY_STEPS))

        def load(self, path):
            try:
                self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
                self.update_target_model()
                print(f"Model weights loaded from {path}")
            except Exception as e:
                print(f"Error loading model weights: {e}")

        def save(self, path):
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(self.policy_net.state_dict(), path)
                print(f"Model weights saved to {path}")
            except Exception as e:
                print(f"Error saving model weights: {e}")
    import optuna
    def get_optuna_episodes(num_nodes):
        if num_nodes <= 10:
            return 2000
        elif num_nodes <= 15:
            return 4000
        elif num_nodes <= 20:
            return 8000
        else:
            return 12000
    def objective(trial):
        # Set fixed random seed for reproducibility
        num_episodes = get_optuna_episodes(NUM_NODES)
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Suggest hyperparameters (narrowed ranges if needed)
        learning_rate = trial.suggest_loguniform('learning_rate', 5e-5, 5e-4)
        gamma = trial.suggest_float('gamma', 0.93, 0.97)
        epsilon_start = trial.suggest_float('epsilon_start', 0.95, 1.0)
        epsilon_end = trial.suggest_float('epsilon_end', 0.03, 0.07)
        epsilon_decay_steps = trial.suggest_int('epsilon_decay_steps', int(0.8*EPSILON_DECAY_STEPS), int(1.2*EPSILON_DECAY_STEPS))
        buffer_size = trial.suggest_int('buffer_size', 8000, 15000)
        batch_size = trial.suggest_int('batch_size', 24, 40)
        max_steps_per_episode = trial.suggest_int('max_steps_per_episode', int(0.8*MAX_STEPS_PER_EPISODE), int(1.2*MAX_STEPS_PER_EPISODE))
        target_update_freq = trial.suggest_int('target_update_freq', 40, 60)
        hidden1 = trial.suggest_int('hidden1', max(64, NUM_NODES*4), max(256, NUM_NODES*16))
        hidden2 = trial.suggest_int('hidden2', max(64, NUM_NODES*4), max(256, NUM_NODES*16))
        hidden3 = trial.suggest_int('hidden3', max(32, NUM_NODES*2), max(128, NUM_NODES*8))

        # Define QNetwork with trial sizes
        class QNetworkOptuna(nn.Module):
            def __init__(self, state_size, action_size):
                super().__init__()
                self.fc1 = nn.Linear(state_size, hidden1)
                self.bn1 = nn.BatchNorm1d(hidden1)
                self.fc2 = nn.Linear(hidden1, hidden2)
                self.bn2 = nn.BatchNorm1d(hidden2)
                self.fc3 = nn.Linear(hidden2, hidden3)
                self.bn3 = nn.BatchNorm1d(hidden3)
                self.fc4 = nn.Linear(hidden3, action_size)
                nn.init.xavier_uniform_(self.fc1.weight)
                nn.init.xavier_uniform_(self.fc2.weight)
                nn.init.xavier_uniform_(self.fc3.weight)
                nn.init.xavier_uniform_(self.fc4.weight)
            def forward(self, state):
                x = F.relu(self.bn1(self.fc1(state)))
                x = F.relu(self.bn2(self.fc2(x)))
                x = F.relu(self.bn3(self.fc3(x)))
                return self.fc4(x)

        class DQNAgentOptuna(DQNAgent_PyTorch):
            def __init__(self, state_size, action_size, device):
                super().__init__(
                    state_size, action_size, learning_rate, gamma,
                    buffer_size, batch_size, device, NUM_NODES
                )
                self.policy_net = QNetworkOptuna(state_size, action_size).to(device)
                self.target_net = QNetworkOptuna(state_size, action_size).to(device)
                self.update_target_model()
                self.target_net.eval()
                self.epsilon = epsilon_start

            def decay_epsilon(self, current_step):
                self.epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (current_step / epsilon_decay_steps))

        agent = DQNAgentOptuna(STATE_SIZE, ACTION_SPACE_SIZE, device)
        total_steps = 0
        rewards = []

        # Training phase
        for ep in range(num_episodes):
            start_node = 0  # Fixed for reproducibility
            current_node = start_node
            time_elapsed = 0.0
            state = np.array([current_node, time_elapsed / MAX_DURATION], dtype=np.float32)
            episode_reward = 0
            for step in range(max_steps_per_episode):
                action = agent.act(state)
                next_node = action
                step_time = time_matrix[current_node][next_node]
                step_reward = reward_matrix[current_node][next_node] / REWARD_SCALE_FACTOR
                next_time_elapsed = time_elapsed + step_time
                next_state = np.array([next_node, min(next_time_elapsed, MAX_DURATION) / MAX_DURATION], dtype=np.float32)
                terminal_reward = 0
                done = False
                if next_node == start_node:
                    if MIN_DURATION <= next_time_elapsed <= MAX_DURATION:
                        terminal_reward = RETURN_SUCCESS_BONUS
                        done = True
                    elif next_time_elapsed < MIN_DURATION:
                        terminal_reward = INCOMPLETE_PENALTY
                        done = True
                    else:
                        terminal_reward = TIME_VIOLATION_PENALTY
                        done = True
                elif next_time_elapsed > MAX_DURATION:
                    terminal_reward = TIME_VIOLATION_PENALTY
                    done = True
                total_reward_experience = step_reward + terminal_reward
                agent.remember(state, action, total_reward_experience, next_state, done)
                state = next_state
                current_node = next_node
                time_elapsed = next_time_elapsed
                episode_reward += step_reward
                total_steps += 1
                agent.decay_epsilon(total_steps)
                agent.replay()
                if done:
                    episode_reward += terminal_reward
                    break
            rewards.append(episode_reward)

        # Validation phase (evaluate on fixed start node, no exploration)
        agent.epsilon = 0.0
        val_rewards = []
        for _ in range(10):
            start_node = 0
            current_node = start_node
            time_elapsed = 0.0
            state = np.array([current_node, time_elapsed / MAX_DURATION], dtype=np.float32)
            episode_reward = 0
            for step in range(max_steps_per_episode):
                action = agent.act(state)
                next_node = action
                step_time = time_matrix[current_node][next_node]
                step_reward = reward_matrix[current_node][next_node] / REWARD_SCALE_FACTOR
                next_time_elapsed = time_elapsed + step_time
                next_state = np.array([next_node, min(next_time_elapsed, MAX_DURATION) / MAX_DURATION], dtype=np.float32)
                terminal_reward = 0
                done = False
                if next_node == start_node:
                    if MIN_DURATION <= next_time_elapsed <= MAX_DURATION:
                        terminal_reward = RETURN_SUCCESS_BONUS
                        done = True
                    elif next_time_elapsed < MIN_DURATION:
                        terminal_reward = INCOMPLETE_PENALTY
                        done = True
                    else:
                        terminal_reward = TIME_VIOLATION_PENALTY
                        done = True
                elif next_time_elapsed > MAX_DURATION:
                    terminal_reward = TIME_VIOLATION_PENALTY
                    done = True
                episode_reward += step_reward
                if done:
                    episode_reward += terminal_reward
                    break
                state = next_state
                current_node = next_node
                time_elapsed = next_time_elapsed
            val_rewards.append(episode_reward)

        return np.mean(val_rewards)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    # Use best Optuna parameters
    best = study.best_params

    LEARNING_RATE = best['learning_rate']
    GAMMA = best['gamma']
    EPSILON_START = best['epsilon_start']
    EPSILON_END = best['epsilon_end']
    EPSILON_DECAY_STEPS = best['epsilon_decay_steps']
    BUFFER_SIZE = best['buffer_size']
    BATCH_SIZE = best['batch_size']
    MAX_STEPS_PER_EPISODE = best['max_steps_per_episode']
    TARGET_UPDATE_FREQ = best['target_update_freq']

    def get_num_episodes(num_nodes):
        # You can tune this formula based on your experiments
        if num_nodes <= 10:
            return 20000
        elif num_nodes <= 15:
            return 50000
        elif num_nodes <= 20:
            return 100000
        else:
            return 150000

    # After Optuna optimization:
    NUM_EPISODES = get_num_episodes(NUM_NODES)
    
    def get_qnetwork_sizes(num_nodes):
        hidden1 = best['hidden1']
        hidden2 = best['hidden2']
        hidden3 = best['hidden3']
        return hidden1, hidden2, hidden3

    # Now create your agent and start training as usual
    drl_agent = DQNAgent_PyTorch(
        state_size=STATE_SIZE,
        action_size=ACTION_SPACE_SIZE,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        device=device,
        num_nodes=NUM_NODES
    )
    # Training History
    drl_episode_rewards = []
    drl_episode_losses = []
    drl_total_steps = 0
    drl_start_train_time = time.time()

    print("Starting DRL Training...")

    #for start_node in range(0,NUM_NODES-1):
    #print(start_node)
    last_reward = 0
    episode = 0
    counter = 0
    while episode < NUM_EPISODES and counter < 2:
    #for episode in range(NUM_EPISODES):
        start_node = random.randint(0, NUM_NODES-1)
        current_node = start_node
        time_elapsed = 0.0
        state = np.array([current_node, time_elapsed / MAX_DURATION], dtype=np.float32)

        episode_reward = 0
        episode_loss_sum = 0
        steps_in_episode = 0
        done = False

        for step in range(MAX_STEPS_PER_EPISODE):
            action = drl_agent.act(state)
            next_node = action
            step_time = time_matrix[current_node][next_node]
            # Use penalized reward matrix for DRL training decisions (implicitly via learned Q)
            # but store experience based on actual rewards + terminal bonus/penalty
            step_reward = reward_matrix[current_node][next_node] / REWARD_SCALE_FACTOR # Base reward for the step

            next_time_elapsed = time_elapsed + step_time
            next_state = np.array([next_node, min(next_time_elapsed, MAX_DURATION) / MAX_DURATION], dtype=np.float32)

            terminal_reward = 0
            done = False

            # Termination checks (Same logic as before)
            if next_node == start_node:
                if MIN_DURATION <= next_time_elapsed <= MAX_DURATION:
                    terminal_reward = RETURN_SUCCESS_BONUS
                    done = True
                elif next_time_elapsed < MIN_DURATION:
                    terminal_reward = INCOMPLETE_PENALTY
                    done = True
                else: # > MAX_DURATION
                    terminal_reward = TIME_VIOLATION_PENALTY
                    done = True
            elif next_time_elapsed > MAX_DURATION:
                terminal_reward = TIME_VIOLATION_PENALTY
                done = True

            # Total reward for the experience tuple
            total_reward_experience = step_reward + terminal_reward

            drl_agent.remember(state, action, total_reward_experience, next_state, done)

            state = next_state
            current_node = next_node
            time_elapsed = next_time_elapsed
            episode_reward += step_reward # Track sum of actual step rewards
            steps_in_episode += 1
            drl_total_steps += 1

            drl_agent.decay_epsilon(drl_total_steps)
            loss = drl_agent.replay()
            if loss > 0: episode_loss_sum += loss
            if drl_total_steps % TARGET_UPDATE_FREQ == 0: drl_agent.update_target_model()
            if done:
                episode_reward += terminal_reward # Add final bonus/penalty for logging
                break

        drl_episode_rewards.append(episode_reward)
        avg_loss = episode_loss_sum / steps_in_episode if steps_in_episode > 0 else 0
        drl_episode_losses.append(avg_loss)
        episode = episode + 1
        if (episode + 1) % (NUM_EPISODES // 10) == 0: # Print progress 10 times
            print(f"DRL Episode: {episode + 1}/{NUM_EPISODES}, Steps: {steps_in_episode}, Total Steps: {drl_total_steps}, Reward: {episode_reward:.0f}, Avg Loss: {avg_loss:.4f}, Epsilon: {drl_agent.epsilon:.3f}")
            if last_reward == episode_reward:
                counter = counter + 1
            last_reward = episode_reward

    drl_training_time = time.time() - drl_start_train_time
    print(f"\nDRL Training Finished. Total time: {drl_training_time:.2f} seconds")
    
    
    import time
    results = []
    mip_times = []
    heuristic_times = []
    drl_inference_times = []
    heuristic2_times = []
    lp_times = []

    print("\n--- Running Solvers for Each Start Node ---")

    for s_node in range(NUM_NODES):
        print(f"Solving for Start Node: {s_node}")
        result_row = {'Start Node': s_node}

        # First solve LP relaxation
        t0 = time.time()
        lp_status, lp_bound = solve_lp_relaxation(s_node, time_matrix, reward_matrix, MIN_DURATION, MAX_DURATION, NUM_NODES)
        lp_times.append(time.time() - t0)
        result_row['LP Status'] = lp_status
        result_row['LP Upper Bound'] = lp_bound
        
        # Solve with DRL
        t0 = time.time()
        drl_route, drl_reward, drl_duration = generate_optimal_route_pytorch(drl_agent, s_node, time_matrix, reward_matrix,NUM_NODES,MAX_DURATION,MIN_DURATION)
        drl_inference_times.append(time.time() - t0)
        result_row['DRL Route'] = drl_route
        result_row['DRL Reward'] = drl_reward if drl_route else -np.inf
        print(drl_route)
        result_row['DRL Duration'] = drl_duration if drl_route else np.inf
        result_row['DRL Valid'] = drl_route is not None and drl_route[0] == drl_route[-1]

        # Solve with MIP
        t0 = time.time()
        # Use original (non-penalized) reward matrix for MIP objective
        mip_status, mip_route, mip_reward, mip_duration = solve_mip(s_node, time_matrix, reward_matrix, MIN_DURATION, MAX_DURATION, NUM_NODES)
        mip_times.append(time.time() - t0)
        result_row['MIP Status'] = mip_status
        result_row['MIP Route'] = mip_route
        result_row['MIP Reward'] = mip_reward if mip_status == 'Optimal' else -np.inf
        result_row['MIP Duration'] = mip_duration if mip_status == 'Optimal' else np.inf
        result_row['MIP Valid'] = mip_status == 'Optimal' and mip_route is not None

        # Solve with Heuristic
        t0 = time.time()
        # Use original reward matrix for heuristic evaluation
        heu_status, heu_route, heu_reward, heu_duration, heu_valid = solve_heuristic(s_node, time_matrix, reward_matrix, MIN_DURATION, MAX_DURATION, NUM_NODES)
        heuristic_times.append(time.time() - t0)
        result_row['Heuristic Route'] = heu_route
        result_row['Heuristic Reward'] = heu_reward if heu_valid else (heu_reward if heu_route else -np.inf) # Show reward even if duration invalid
        result_row['Heuristic Duration'] = heu_duration if heu_route else np.inf
        result_row['Heuristic Valid'] = heu_valid

        t0 = time.time()
        # Use original reward matrix for heuristic evaluation
        reg_status, reg_route, reg_reward, reg_duration, reg_valid = solve_regret2_heuristic(s_node, time_matrix, reward_matrix, MIN_DURATION, MAX_DURATION, NUM_NODES)
        heuristic2_times.append(time.time() - t0)
        result_row['Regret2 Route'] = reg_route
        result_row['Regret2 Reward'] = reg_reward if reg_valid else (reg_reward if reg_route else -np.inf)
        result_row['Regret2 Duration'] = reg_duration if reg_route else np.inf
        result_row['Regret2 Valid'] = reg_valid
        # Add gap calculations as for other heuristics
        
        # Calculate Optimality Gaps (relative to MIP if MIP is optimal)
        mip_opt_reward = result_row['MIP Reward']
        if result_row['MIP Valid']:
            drl_gap = ((mip_opt_reward - result_row['DRL Reward']) / abs(mip_opt_reward)) * 100 if abs(mip_opt_reward) > 1e-6 and result_row['DRL Valid'] else float('nan')
            heu_gap = ((mip_opt_reward - result_row['Heuristic Reward']) / abs(mip_opt_reward)) * 100 if abs(mip_opt_reward) > 1e-6 and result_row['Heuristic Valid'] else float('nan')
            reg_gap = ((mip_opt_reward - result_row['Regret2 Reward']) / abs(mip_opt_reward)) * 100 if abs(mip_opt_reward) > 1e-6 and reg_valid else float('nan')
        else:
            drl_gap = float('nan')
            heu_gap = float('nan')
            reg_gap = float('nan')
        # Add gap calculations relative to LP bound
        if lp_status == 'Optimal':
            mip_lp_gap = ((lp_bound - result_row['MIP Reward']) / abs(lp_bound)) * 100 if result_row['MIP Valid'] else float('nan')
            drl_lp_gap = ((lp_bound - result_row['DRL Reward']) / abs(lp_bound)) * 100 if result_row['DRL Valid'] else float('nan')
            heu_lp_gap = ((lp_bound - result_row['Heuristic Reward']) / abs(lp_bound)) * 100 if result_row['Heuristic Valid'] else float('nan')
            reg_lp_gap = ((lp_bound - result_row['Regret2 Reward']) / abs(lp_bound)) * 100 if reg_valid else float('nan')
        else:
            mip_lp_gap = float('nan')
            drl_lp_gap = float('nan')
            heu_lp_gap = float('nan')  
            reg_lp_gap = float('nan')  
        result_row['MIP-LP Gap (%)'] = mip_lp_gap
        result_row['DRL-LP Gap (%)'] = drl_lp_gap
        result_row['Heuristic-LP Gap (%)'] = heu_lp_gap
        result_row['Regret2-LP Gap (%)'] = reg_lp_gap
        result_row['DRL Gap (%)'] = drl_gap
        result_row['Heuristic Gap (%)'] = heu_gap
        result_row['Regret2 Gap (%)'] = reg_gap

        results.append(result_row)
    results_df = pd.DataFrame(results)
    # Average Rewards and Gaps
    avg_lp_bound = results_df['LP Upper Bound'].mean()
    avg_mip_reward = results_df.loc[results_df['MIP Valid'], 'MIP Reward'].mean()
    avg_drl_reward = results_df.loc[results_df['DRL Valid'], 'DRL Reward'].mean()
    avg_heu_reward = results_df.loc[results_df['Heuristic Valid'], 'Heuristic Reward'].mean()
    avg_regret2_reward = results_df.loc[results_df['Regret2 Valid'], 'Regret2 Reward'].mean()
    summary_rows.append({
        'Node Size': NUM_NODES,
        'LP Upper Bound': avg_lp_bound,
        'MIP Avg Reward': avg_mip_reward,
        'DRL Avg Reward': avg_drl_reward,
        'Heuristic Avg Reward': avg_heu_reward,
        'Regret2 Avg Reward': avg_regret2_reward
    })
    # Format gaps for display
    results_df['DRL Gap (%)'] = results_df['DRL Gap (%)'].map('{:.1f}'.format, na_action='ignore')
    results_df['Heuristic Gap (%)'] = results_df['Heuristic Gap (%)'].map('{:.1f}'.format, na_action='ignore')
node_summary_df = pd.DataFrame(summary_rows)
node_summary_df.to_excel("DRL_Routing_Summary.xlsx", index=False)