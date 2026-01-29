import numpy as np
import matplotlib.pyplot as plt
import numba
from collections import deque


def generate_lattice(L, p):
    """Generates a LxL lattice with site probability p."""
    return np.random.rand(L, L) < p


# Question 2.1.1: BFS Algorithm
import numpy as np
from numba import njit

# We use 'int16' for coordinates because 2048 fits inside 32,767.
# This cuts memory usage in half compared to standard integers.
@numba.njit()
def has_percolation_numba_large(lattice):
    L = lattice.shape[0]
    
    # 1. Pre-allocate memory for the worst-case scenario (every cell in queue)
    # 2048 * 2048 = ~4.2 million items. 
    # Using int16, these two arrays only take ~16MB of RAM total. Trivial.
    queue_r = np.empty(L * L, dtype=np.int16)
    queue_c = np.empty(L * L, dtype=np.int16)
    
    head = 0
    tail = 0
    
    # Initialize top row
    for col in range(L):
        if lattice[0, col] == 1:
            lattice[0, col] = 0
            queue_r[tail] = 0
            queue_c[tail] = col
            tail += 1
            
    while head < tail:
        # "Pop" from the head
        r = queue_r[head]
        c = queue_c[head]
        head += 1
        
        # Check Win Condition
        if r == L - 1:
            return True
        
        # Check Neighbors (Manual unroll is fastest for Numba)
        # We check bounds + occupancy in one go to keep the pipeline full
        
        # DOWN
        if r + 1 < L and lattice[r + 1, c] == 1:
            lattice[r + 1, c] = 0 # Mark visited
            queue_r[tail] = r + 1
            queue_c[tail] = c
            tail += 1
            
        # UP
        if r - 1 >= 0 and lattice[r - 1, c] == 1:
            lattice[r - 1, c] = 0
            queue_r[tail] = r - 1
            queue_c[tail] = c
            tail += 1
            
        # RIGHT
        if c + 1 < L and lattice[r, c + 1] == 1:
            lattice[r, c + 1] = 0
            queue_r[tail] = r
            queue_c[tail] = c + 1
            tail += 1

        # LEFT
        if c - 1 >= 0 and lattice[r, c - 1] == 1:
            lattice[r, c - 1] = 0
            queue_r[tail] = r
            queue_c[tail] = c - 1
            tail += 1
                
    return False
numba.njit()
def has_percolation_top_to_bottom(lattice):
    """
    Optimized BFS. Destroys 'lattice' content to mark visited sites 
    (turns 1s into 0s) to avoid using a slow 'visited' set.
    """
    L = lattice.shape[0]
    queue = deque()

    # Possible moves
    moves = ((1, 0), (-1, 0), (0, 1), (0, -1))
    
    # Add start nodes from the top row.
    for col in range(L):
        # "1" means it has a tree.
        if lattice[0, col] == 1:
            # Mark visited by "burning" the place.
            lattice[0, col] = 0
            # Add to queue to check for neighbors
            queue.append((0, col))

    while queue:
        row, col = queue.popleft()

        for row_move, col_move in moves:
            test_row, test_col = row + row_move, col + col_move
            
            # Check bounds
            if (test_row < 0 or test_row >= L) or (test_col < 0 or test_col >= L):
                continue

            if lattice[test_row, test_col] == 1:
                # If we reached the bottom, we have percolation
                if test_row == L - 1:
                    return True

                lattice[test_row, test_col] = 0  # Mark visited
                queue.append((test_row, test_col))

    return False


def solve_2_1_1():
    print("Running Question 2.1.1 (Percolation Threshold)")
    print("WARNING: THIS IS SLOW")
    L = 2048
    
    # Step size of 0.05 as requested in the homework
    p_values = np.arange(0, 1.01, 0.05)
    y_values = []

    num_trials = 100

    for p in p_values:
        percolates_count = 0
        for _ in range(num_trials):
            lat = generate_lattice(L, p)
            if has_percolation_numba_large(lat):
                percolates_count += 1
        
        prob = percolates_count / num_trials
        y_values.append(prob)
        print(f"p={p:.2f} | P(spanning)={prob:.2f}")

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(p_values, y_values, 'o-', linewidth=2)
    plt.axvline(x=0.593, color='red', linestyle='--', label='Theoretical $p_c$')
    plt.axvline(x=0.618, color='red', linestyle='--', label='RG Approx $p_c$')
    plt.title(f"Percolation Probability (L={L})")
    plt.xlabel("p")
    plt.ylabel("P(spanning)")
    plt.legend()
    plt.grid(True)
    plt.savefig("percolation_probability.png")


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

def solve_2_1_2():
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
    plt.savefig("rg_flow_lattices.png")

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
    solve_2_1_1()
    solve_2_1_2()