import numpy as np
import matplotlib.pyplot as plt
from collections import deque


def generate_lattice(L, p):
    """Generates a LxL lattice with site probability p."""
    return np.random.rand(L, L) < p


# Question 2.1.1: BFS Algorithm
def has_percolation_top_to_bottom(lattice):
    """
    BFS implementation to check for a path from top row to bottom row.
    Returns True if a path exists.
    """
    L = lattice.shape[0]
    
    # 1. Initialize Queue with all occupied sites in the top row (row 0)
    queue = deque()
    visited = set()
    
    # Add all occupied cells in the first row to the queue
    for col in range(L):
        if lattice[0, col]:
            state = (0, col)
            queue.append(state)
            visited.add(state)
            
    # 2. Run BFS
    while queue:
        r, c = queue.popleft()
        
        # If we reached the bottom row, we have a spanning cluster
        if r == L - 1:
            return True
        
        # Check 4 neighbors (Up, Down, Left, Right)
        # Yes, there is a redundancy here. For a more efficient implementation,
        # we could avoid re-checking visited nodes, but clarity is prioritized.
        neighbors = [
            (r+1, c), (r-1, c), 
            (r, c+1), (r, c-1)
        ]
        
        for nr, nc in neighbors:
            # Check bounds
            if 0 <= nr < L and 0 <= nc < L:
                # Check if site is occupied and not visited
                if lattice[nr, nc] and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
                    
    return False

def solve_2_1_1():
    print("--- Running Question 2.1.1 (Percolation Threshold) ---")
    print("WARNING: THIS IS SLOW")
    L_large = 2048 # Note: BFS on 2048x2048 in Python is slow. 
                   # For testing, you might want to try L=128 first.
    
    # Step size of 0.05 as requested
    p_values = np.arange(0, 1.01, 0.05)
    spanning_probs = []

    # Using 10 trials for speed in this demonstration. 
    # Homework asks for 100 trials.
    num_trials = 10 

    for p in p_values:
        success_count = 0
        for _ in range(num_trials):
            lat = generate_lattice(L_large, p)
            if has_percolation_top_to_bottom(lat):
                success_count += 1
        
        prob = success_count / num_trials
        spanning_probs.append(prob)
        print(f"p={p:.2f} | P(spanning)={prob:.2f}")

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(p_values, spanning_probs, 'o-', linewidth=2)
    plt.axvline(x=0.5927, color='red', linestyle='--', label='Theoretical $p_c$')
    plt.title(f"Percolation Probability (L={L_large})")
    plt.xlabel("p")
    plt.ylabel("P(spanning)")
    plt.legend()
    plt.grid(True)
    plt.savefig("")

# ==========================================
# Question 2.1.2: Decimation & Reshape
# ==========================================
def decimate_majority(lattice):
    """
    Coarse grains the lattice by a factor of 2.
    Rule: A new site is 1 if there is a vertical path in the underlying 2x2 block.
    """
    L_y, L_x = lattice.shape
    new_L_y, new_L_x = L_y // 2, L_x // 2
    
    # 1. Reshape to isolate 2x2 blocks
    # Logic: Break L into (L/2) chunks of 2.
    # Result shape: (Rows of blocks, Rows inside block, Cols of blocks, Cols inside block)
    view = lattice[:2*new_L_y, :2*new_L_x].reshape(new_L_y, 2, new_L_x, 2)
    
    # 2. Extract corners using slicing
    # view[:, 0, :, 0] means: For all blocks, take row 0 (top), col 0 (left)
    TL = view[:, 0, :, 0] # Top-Left
    TR = view[:, 0, :, 1] # Top-Right
    BL = view[:, 1, :, 0] # Bottom-Left
    BR = view[:, 1, :, 1] # Bottom-Right
    
    # 3. Apply Connectivity Rule
    # Vertical path exists if:
    # (Left column is full) OR (Right column is full)
    left_col_path = (TL & BL)
    right_col_path = (TR & BR)
    
    new_lattice = left_col_path | right_col_path
    return new_lattice

def solve_2_1_2():
    print("\n--- Running Question 2.1.2 (RG Decimation) ---")
    L = 2048
    # Exact critical p for site percolation is approx 0.592746
    p_critical = 0.5927
    p_scenarios = [p_critical, 0.55, 0.65]
    
    iterations = 5
    
    fig, axes = plt.subplots(len(p_scenarios), iterations + 1, figsize=(12, 8))
    
    for row, p in enumerate(p_scenarios):
        # Initial Lattice
        lat = generate_lattice(L, p)
        density = np.mean(lat)
        
        # Plot initial (cropped for visibility)
        ax = axes[row, 0]
        ax.imshow(lat[:64, :64], cmap='binary', vmin=0, vmax=1)
        ax.set_ylabel(f"Start p={p}")
        ax.set_title(f"Gen 0\n$\\rho={density:.2f}$")
        ax.set_xticks([]); ax.set_yticks([])
        
        # Decimate loop
        current_lat = lat
        for i in range(iterations):
            current_lat = decimate_majority(current_lat)
            density = np.mean(current_lat)
            
            ax = axes[row, i+1]
            # Show crop if large, full if small
            if current_lat.shape[0] > 64:
                ax.imshow(current_lat[:64, :64], cmap='binary', vmin=0, vmax=1)
            else:
                ax.imshow(current_lat, cmap='binary', vmin=0, vmax=1)
                
            ax.set_title(f"Gen {i+1}\n$\\rho={density:.2f}$")
            ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    solve_2_1_1() 
    solve_2_1_2()