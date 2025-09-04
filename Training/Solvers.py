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
import pulp 
import math
import os


def generate_optimal_route_pytorch(agent, start_node, time_matrix, reward_matrix, NUM_NODES,MAX_DURATION,MIN_DURATION):
        """
        Generates a route using the learned policy (greedy selection),
        preventing revisits to intermediate nodes.
        If max_steps is reached, attempts forced return if valid.
        Validates final duration window.
        """
        max_steps=2*NUM_NODES  # Allow up to twice the number of nodes in steps
        agent.epsilon = 0
        agent.policy_net.eval()
        current_node = start_node
        time_elapsed = 0.0
        state = np.array([current_node, time_elapsed / MAX_DURATION], dtype=np.float32)
        route = [start_node]
        visited_intermediate_nodes = set() # Keep track of nodes visited *other than* start_node
        total_reward = 0.0
        returned_home = False

        with torch.no_grad():
            for step in range(max_steps):
                # --- Action Selection ---
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(agent.device)
                q_values = agent.policy_net(state_tensor)
                q_values_numpy = q_values.cpu().data.numpy()[0]

                # --- Masking Invalid Actions ---
                # 1. Don't stay in the same node
                q_values_numpy[current_node] = -np.inf

                # 2. Don't visit intermediate nodes already visited
                for visited_node_idx in visited_intermediate_nodes:
                    if 0 <= visited_node_idx < len(q_values_numpy): # Bounds check
                        q_values_numpy[visited_node_idx] = -np.inf

                # --- Choose Best Valid Action ---
                next_node = np.argmax(q_values_numpy)

                # Check if any valid action exists
                if q_values_numpy[next_node] == -np.inf:
                    # No valid moves possible (maybe all unvisited nodes violate time or Q-values are terrible)
                    # Try forcing return home immediately if possible
                    # print(f"DRL: Stuck at node {current_node}. No valid non-visited moves. Trying return home.")
                    if current_node != start_node:
                        return_time = time_matrix[current_node][start_node]
                        if time_elapsed + return_time <= MAX_DURATION + 1e-6:
                            next_node = start_node # Override choice to return home
                            # print("DRL: Forcing return home as only option.")
                        else:
                            # print(f"DRL Error: Stuck at node {current_node}. Cannot return home within duration.")
                            returned_home = False
                            break # Cannot proceed
                    else: # Stuck at start node? Should not happen if mask works.
                        returned_home = False
                        break


                # --- Simulate Step ---
                step_time = time_matrix[current_node][next_node]
                step_reward = reward_matrix[current_node][next_node]

                # --- Check immediate time violation (should be less likely now with stuck check) ---
                if time_elapsed + step_time > MAX_DURATION + 1e-6 and next_node != start_node:
                    # print(f"DRL: Next step to {next_node} violates MAX_DURATION. Stopping.")
                    returned_home = False
                    break

                # --- Update State ---
                time_elapsed += step_time
                total_reward += step_reward
                current_node = next_node
                route.append(current_node)
                # Add to visited set *only if* it's not the start node
                if current_node != start_node:
                    visited_intermediate_nodes.add(current_node)

                state = np.array([current_node, min(time_elapsed, MAX_DURATION) / MAX_DURATION], dtype=np.float32)

                # --- Check for Natural Return ---
                if current_node == start_node:
                    returned_home = True
                    break

            # --- End of Step Loop ---

            # --- Handle Forced Return if max_steps reached ---
            # (This logic might be less necessary now but keep as fallback)
            if not returned_home and current_node != start_node:
                # print(f"DRL: Max steps reached, attempting forced return from {current_node} to {start_node}")
                return_time = time_matrix[current_node][start_node]
                return_reward = reward_matrix[current_node][start_node]
                if time_elapsed + return_time <= MAX_DURATION + 1e-6:
                    time_elapsed += return_time
                    total_reward += return_reward
                    current_node = start_node
                    route.append(start_node)
                    returned_home = True
                # else: # Forced return violates time
                    # returned_home remains False

        # --- Final Validation ---
        agent.policy_net.train()

        is_cycle = returned_home and route[0] == start_node and route[-1] == start_node and len(route)>1

        if not is_cycle:
            return None, -np.inf, np.inf # Failed route

        is_valid_duration = MIN_DURATION <= time_elapsed <= MAX_DURATION

        # Check for duplicate intermediate nodes (should be prevented by logic above)
        intermediate_nodes = route[1:-1]
        has_duplicates = len(intermediate_nodes) != len(set(intermediate_nodes))
        if has_duplicates:
            print(f"Warning: DRL route {route} has duplicate intermediate nodes despite masking!")
            # Treat as invalid? Or just note it. Let's return it but validity check below will fail if needed.

        if is_valid_duration and not has_duplicates:
            return route, total_reward, time_elapsed # Valid cycle found
        else:
            # Cycle formed, but duration or node visit is invalid
            return route, total_reward, time_elapsed # Return invalid route details
def solve_mip(start_node, time_m, reward_m, min_d, max_d, num_n):
    """Solves the VRP variant using MIP for a given start node."""
    nodes = list(range(num_n))
    other_nodes = [n for n in nodes if n != start_node]

    # Create the model
    prob = pulp.LpProblem(f"VRP_Cycle_{start_node}", pulp.LpMaximize)

    # Decision Variables
    x = pulp.LpVariable.dicts("Route", (nodes, nodes), 0, 1, pulp.LpBinary)
    u = pulp.LpVariable.dicts("MTZ", nodes, 1, num_n - 1, pulp.LpContinuous)

    # Objective Function
    prob += pulp.lpSum(reward_m[i][j] * x[i][j] for i in nodes for j in nodes if i != j)

    # Constraints
    # 1. Degree Constraints
    for k in nodes:
        prob += pulp.lpSum(x[k][j] for j in nodes if k != j) == pulp.lpSum(x[j][k] for j in nodes if k != j)
        if k == start_node:
            prob += pulp.lpSum(x[start_node][j] for j in nodes if j != start_node) == 1
            prob += pulp.lpSum(x[j][start_node] for j in nodes if j != start_node) == 1
        else:
             prob += pulp.lpSum(x[j][k] for j in nodes if j != k) <= 1

    # 2. Duration Constraints
    total_time = pulp.lpSum(time_m[i][j] * x[i][j] for i in nodes for j in nodes if i != j)
    prob += total_time >= min_d
    prob += total_time <= max_d

    # 3. Subtour Elimination (MTZ)
    for i in other_nodes:
        prob += u[i] >= 1
        for j in other_nodes:
            if i != j:
                 prob += u[i] - u[j] + 1 <= (num_n - 1) * (1 - x[i][j])

    # Solve the problem
    solver = pulp.PULP_CBC_CMD(msg=0)
    prob.solve(solver)

    # Extract results
    status = pulp.LpStatus[prob.status]
    route = None
    total_reward = -np.inf
    total_duration = np.inf

    if status == 'Optimal':
        total_reward = pulp.value(prob.objective)
        total_duration = pulp.value(total_time)

        # --- Modified Route Reconstruction ---
        try: # Add a try-except block for safety during reconstruction
            current_node = start_node
            route = [start_node]
            visited_count = 0 # Safety counter

            while visited_count <= num_n: # Limit search depth
                found_next = False
                for j in nodes:
                    # Check if arc variable exists, is not None, and is selected (> 0.99)
                    # Also ensure j is not the current node
                    if j != current_node and \
                       x[current_node][j] is not None and \
                       x[current_node][j].varValue is not None and \
                       x[current_node][j].varValue > 0.99:

                        route.append(j)
                        current_node = j
                        found_next = True
                        break # Move to the next node in the path

                visited_count += 1

                if current_node == start_node: # Successfully completed the cycle
                    break
                if not found_next: # Dead end found during reconstruction
                    # print(f"MIP Route Reconstruction Error: Dead end at node {current_node} for start {start_node}.")
                    route = None # Invalid route
                    break
                if visited_count > num_n: # Avoid infinite loops / too many steps
                    # print(f"MIP Route Reconstruction Error: Route too long for start {start_node}. Path: {route}")
                    route = None # Invalid route
                    break

            # Final validation of reconstructed route
            if route is None or route[0] != start_node or route[-1] != start_node:
                 # print(f"MIP Route Reconstruction resulted in invalid path for start {start_node}. Route: {route}")
                 route = None
                 # If route is invalid, reset reward/duration derived from MIP objective
                 total_reward = -np.inf
                 total_duration = np.inf
                 status = 'Error_In_Route' # Update status to reflect this

        except Exception as e:
            print(f"Exception during MIP route reconstruction for start {start_node}: {e}")
            route = None
            total_reward = -np.inf
            total_duration = np.inf
            status = 'Error_Exception'
        # --- End of Modified Route Reconstruction ---

    # Ensure reward/duration are consistent if route is None
    if route is None:
         total_reward = -np.inf
         total_duration = np.inf
         # Update status if it was 'Optimal' but route failed
         if status == 'Optimal': status = 'Optimal_Route_Fail'


    return status, route, total_reward, total_duration

def solve_heuristic(start_node, time_m, reward_m, min_d, max_d, num_n):
    """
    Solves using a simple greedy heuristic:
    1. Always move to highest reward available node
    2. Return to start node when no more valid moves
    """
    current_node = start_node
    time_elapsed = 0.0
    total_reward = 0.0
    route = [start_node]
    visited = {start_node}
    return_threshold = 0.85 * max_d

    while True:  # Continue until we break
        # Find next highest reward move
        best_reward = -np.inf
        best_next_node = None
        
        for next_node in range(num_n):
            if next_node != current_node and next_node not in visited:
                step_time = time_m[current_node][next_node]
                return_time = time_m[next_node][start_node]  # Time to get back home
                step_reward = reward_m[current_node][next_node]
                
                # Check if we can make this move and still get back home
                total_future_time = time_elapsed + step_time + return_time
                if total_future_time <= max_d:
                    if step_reward > best_reward:
                        best_reward = step_reward
                        best_next_node = next_node

        if best_next_node is not None:
            # Make the move to the highest reward node
            time_elapsed += time_m[current_node][best_next_node]
            total_reward += reward_m[current_node][best_next_node]
            current_node = best_next_node
            route.append(current_node)
            visited.add(current_node)
        else:
            # No more valid moves, return home
            return_time = time_m[current_node][start_node]
            if current_node != start_node:  # Only if we're not already home
                time_elapsed += return_time
                total_reward += reward_m[current_node][start_node]
                route.append(start_node)
            break
        # If we've reached the time threshold, try to return home
        if time_elapsed >= return_threshold:
            return_time = time_m[current_node][start_node]
            time_elapsed += return_time
            total_reward += reward_m[current_node][start_node]
            route.append(start_node)
            break
    # Final validation
    is_valid = False
    status = "Infeasible"
    
    if route[-1] == start_node and len(route) > 1:
        if min_d <= time_elapsed <= max_d:
            status = "Optimal"
            is_valid = True

    return status, route, total_reward, time_elapsed, is_valid
def solve_lp_relaxation(start_node, time_m, reward_m, min_d, max_d, num_n):
    """Solves LP relaxation of the VRP variant for upper bound."""
    nodes = list(range(num_n))
    other_nodes = [n for n in nodes if n != start_node]

    # Create LP model
    lp_prob = pulp.LpProblem(f"VRP_LP_Relaxation_{start_node}", pulp.LpMaximize)
    
    # Decision Variables (continuous between 0 and 1)
    x = pulp.LpVariable.dicts("Route", (nodes, nodes), 0, 1, pulp.LpContinuous)
    u = pulp.LpVariable.dicts("MTZ", nodes, 1, num_n - 1, pulp.LpContinuous)

    # Same objective and constraints as MIP
    lp_prob += pulp.lpSum(reward_m[i][j] * x[i][j] for i in nodes for j in nodes if i != j)

    for k in nodes:
        lp_prob += pulp.lpSum(x[k][j] for j in nodes if k != j) == pulp.lpSum(x[j][k] for j in nodes if k != j)
        if k == start_node:
            lp_prob += pulp.lpSum(x[start_node][j] for j in nodes if j != start_node) == 1
            lp_prob += pulp.lpSum(x[j][start_node] for j in nodes if j != start_node) == 1
        else:
            lp_prob += pulp.lpSum(x[j][k] for j in nodes if j != k) <= 1

    total_time = pulp.lpSum(time_m[i][j] * x[i][j] for i in nodes for j in nodes if i != j)
    lp_prob += total_time >= min_d
    lp_prob += total_time <= max_d

    # Solve LP
    solver = pulp.PULP_CBC_CMD(msg=0)
    lp_prob.solve(solver)

    status = pulp.LpStatus[lp_prob.status]
    upper_bound = pulp.value(lp_prob.objective) if status == 'Optimal' else np.inf

    return status, upper_bound
def solve_regret2_heuristic(start_node, time_m, reward_m, min_d, max_d, num_n):
    """
    Regret-2 heuristic: At each step, select the node where the difference between the best and second-best reward/time ratio is largest.
    """
    current_node = start_node
    time_elapsed = 0.0
    total_reward = 0.0
    route = [start_node]
    visited = {start_node}
    while True:
        candidates = []
        for next_node in range(num_n):
            if next_node != current_node and next_node not in visited:
                step_time = time_m[current_node][next_node]
                return_time = time_m[next_node][start_node]
                total_future_time = time_elapsed + step_time + return_time
                if total_future_time <= max_d:
                    # Calculate reward/time ratio for all possible next steps from this candidate
                    ratios = []
                    for future_node in range(num_n):
                        if future_node != next_node and future_node not in visited:
                            future_time = time_m[next_node][future_node]
                            future_reward = reward_m[next_node][future_node]
                            if future_time > 0:
                                ratios.append(future_reward / future_time)
                    ratios = sorted(ratios, reverse=True)
                    best_ratio = reward_m[current_node][next_node] / step_time if step_time > 0 else 0
                    second_best = ratios[0] if ratios else 0
                    regret = best_ratio - second_best
                    candidates.append((regret, next_node, step_time, reward_m[current_node][next_node]))
        if candidates:
            # Pick node with highest regret
            candidates.sort(reverse=True)
            _, best_next_node, step_time, step_reward = candidates[0]
            time_elapsed += step_time
            total_reward += step_reward
            current_node = best_next_node
            route.append(current_node)
            visited.add(current_node)
        else:
            # Return home
            if current_node != start_node:
                time_elapsed += time_m[current_node][start_node]
                total_reward += reward_m[current_node][start_node]
                route.append(start_node)
            break
    is_valid = route[-1] == start_node and min_d <= time_elapsed <= max_d
    status = "Optimal" if is_valid else "Infeasible"
    return status, route, total_reward, time_elapsed, is_valid
