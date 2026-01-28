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


# Question 2.1.2: Decimation & Reshape
def decimate_spanning_rule(lattice):
    """
    Coarse grains the lattice by a factor of 2 using the SPANNING rule.
    (Mistakenly called 'majority' in the prompt text, but defined as connectivity).
    """
    L_y, L_x = lattice.shape
    new_L_y, new_L_x = L_y // 2, L_x // 2
    
    # 1. Reshape to isolate 2x2 blocks
    # Shape: (Rows of blocks, 2, Cols of blocks, 2)
    view = lattice[:2*new_L_y, :2*new_L_x].reshape(new_L_y, 2, new_L_x, 2)
    
    # 2. Extract pixels
    TL = view[:, 0, :, 0] # Top-Left
    TR = view[:, 0, :, 1] # Top-Right
    BL = view[:, 1, :, 0] # Bottom-Left
    BR = view[:, 1, :, 1] # Bottom-Right
    
    # 3. Apply Vertical Path Rule
    # Path exists if Left Col connects OR Right Col connects
    path_exists = (TL & BL) | (TR & BR)
    
    return path_exists

def run_exercise_2_1_2():
    print("Running Exercise 2.1.2...")
    
    # Parameters explicitly requested
    L_start = 2048
    scenarios = [0.5927, 0.55, 0.65] # p_c, p < p_c, p > p_c
    
    # We will store data for the final plot
    results = {} 

    # Create figure for the lattice images (Visualizing the RG flow)
    # 3 rows (scenarios), 5 columns (generations 0 to 4)
    fig_img, axes = plt.subplots(3, 5, figsize=(15, 9))
    fig_img.suptitle("Visualizing RG Flow (Decimation of Single Realization)", fontsize=16)

    for idx, p in enumerate(scenarios):
        # 1. "Choose a single realization" 
        current_lattice = generate_lattice(L_start, p)
        
        # Store density history for this specific realization
        densities = []
        
        # 2. Iteratively Decimate
        # We will do 5 steps (Generation 0 to 4)
        for gen in range(5):
            # Calculate "percentage of active sites" [cite: 37]
            density = np.mean(current_lattice)
            densities.append(density)
            
            # Plot the lattice (taking a crop if it's too huge to see details)
            ax = axes[idx, gen]
            
            # If lattice is huge, zoom in on top-left 64x64 corner so we can see pixels
            # If lattice is small, show the whole thing
            if current_lattice.shape[0] > 64:
                display_data = current_lattice[:64, :64]
                title_extra = "(Zoom)"
            else:
                display_data = current_lattice
                title_extra = "(Full)"
                
            ax.imshow(display_data, cmap='binary', vmin=0, vmax=1)
            ax.set_title(f"p={p}, Gen {gen}\nActive: {density:.1%}")
            ax.axis('off')
            
            # Decimate for next round
            if gen < 4: # Don't decimate after the last plot
                current_lattice = decimate_spanning_rule(current_lattice)
        
        results[p] = densities

    plt.tight_layout()
    plt.show()

    # 3. "Plot the percentage of active sites as a function of the decimation" [cite: 37]
    plt.figure(figsize=(10, 6))
    generations = range(5)
    
    for p, densities in results.items():
        if p == 0.5927:
            label = f"$p=p_c \\approx {p}$ (Critical)"
            style = 'o--r'
        elif p == 0.55:
            label = f"$p={p}$ (Sub-critical)"
            style = 'o-b'
        else:
            label = f"$p={p}$ (Super-critical)"
            style = 'o-g'
            
        plt.plot(generations, densities, style, label=label, linewidth=2)

    plt.title("RG Flow: Percentage of Active Sites vs Decimation Step")
    plt.xlabel("Decimation Step (Generation)")
    plt.ylabel("Percentage of Active Sites")
    plt.axhline(0, color='k', linestyle=':', alpha=0.3)
    plt.axhline(1, color='k', linestyle=':', alpha=0.3)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # solve_2_1_1()
    solve_2_1_2()