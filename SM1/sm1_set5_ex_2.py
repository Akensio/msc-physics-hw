import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

# --- Configuration ---
L_SIZE = 2048
TRIALS = 100  # Number of realizations for part 1
P_STEPS = np.arange(0, 1.05, 0.05)
PC_ESTIMATE = 0.593  # From tutorial text


def check_percolation(lattice):
    """
    Checks if there is a path from the top row to the bottom row using
    Nearest Neighbor (NN) connectivity.
    """
    # Structure defines connectivity: [[0,1,0],[1,1,1],[0,1,0]] is NN (4-neighbors)
    structure = [[0, 1, 0],
                 [1, 1, 1],
                 [0, 1, 0]]
    
    labeled_array, num_features = label(lattice, structure=structure)
    
    if num_features == 0:
        return False
    
    # Get labels present in the top and bottom rows
    top_labels = np.unique(labeled_array[0, :])
    bottom_labels = np.unique(labeled_array[-1, :])
    
    # 0 is the background (empty sites), remove it
    top_labels = top_labels[top_labels > 0]
    bottom_labels = bottom_labels[bottom_labels > 0]
    
    # Check intersection
    return not set(top_labels).isdisjoint(bottom_labels)

def decimate_lattice(lattice):
    """
    Performs majority decimation on 2x2 blocks.
    Rule: A block is 'active' if there is a path from top to bottom WITHIN the 2x2 block.
    
    In a 2x2 block with NN connectivity:
    [[a, b],
     [c, d]]
    A path from top (a, b) to bottom (c, d) exists ONLY if:
    (a AND c) is True  OR  (b AND d) is True.
    """
    # Slice the array to get 2x2 windows
    # Top-left (a): rows 0,2,4... cols 0,2,4...
    a = lattice[0::2, 0::2]
    # Top-right (b): rows 0,2,4... cols 1,3,5...
    b = lattice[0::2, 1::2]
    # Bottom-left (c): rows 1,3,5... cols 0,2,4...
    c = lattice[1::2, 0::2]
    # Bottom-right (d): rows 1,3,5... cols 1,3,5...
    d = lattice[1::2, 1::2]
    
    # Apply the percolation rule for the 2x2 block
    # Note: This matches the polynomial 2p^2 - p^4 derived in standard texts and the tutorial
    new_lattice = (a & c) | (b & d)
    
    return new_lattice.astype(int)

def ex_2_1_percolation_probability():
    print("Running Part 1: Probability of Percolation...")
    percolation_probs = []

    for p in P_STEPS:
        success_count = 0
        for _ in range(TRIALS):
            # Generate random lattice (1 = filled, 0 = empty)
            lattice = (np.random.rand(L_SIZE, L_SIZE) < p).astype(int)
            if check_percolation(lattice):
                success_count += 1
        
        prob = success_count / TRIALS
        percolation_probs.append(prob)
        print(f"p={p:.2f}, P(percolate)={prob:.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(P_STEPS, percolation_probs, 'o-', label='Simulation')
    plt.axvline(x=PC_ESTIMATE, color='r', linestyle='--', label=f'Theoretical $p_c \\approx {PC_ESTIMATE}$')
    plt.xlabel('Occupation Probability $p$')
    plt.ylabel('Percolation Probability $P(p)$')
    plt.title('Percolation Transition on 2048x2048 Lattice')
    plt.legend()
    plt.grid(True)
    plt.show()

def ex_2_2_rg_decimation():
    print("\nRunning Part 2: RG Decimation...")
    p_values = [0.55, PC_ESTIMATE, 0.65]
    colors = ['blue', 'green', 'red']
    labels = ['Sub-critical ($p=0.55$)', 'Critical ($p \\approx p_c$)', 'Super-critical ($p=0.65$)']

    plt.figure(figsize=(12, 8))

    for idx, p in enumerate(p_values):
        # Single realization
        lattice = (np.random.rand(L_SIZE, L_SIZE) < p).astype(int)
        current_lattice = lattice
        densities = []
        
        # Store initial state for visualization
        initial_lattice = current_lattice.copy()
        
        # Perform decimation until lattice is too small
        step = 0
        while current_lattice.shape[0] >= 2:
            densities.append(np.mean(current_lattice))
            current_lattice = decimate_lattice(current_lattice)
            step += 1
            
        densities.append(np.mean(current_lattice)) # Final point
        
        # Plot active site percentage vs decimation step
        plt.plot(range(len(densities)), densities, 'o-', color=colors[idx], label=labels[idx])
        
        # Show the decimated lattices (Optional visualization logic)
        # We display the state after 8 steps (2048 -> 2^3 = 8 sized lattice) just as an example
        # Or we can simply describe the result.

    plt.xlabel('RG Step (Decimation Iteration)')
    plt.ylabel('Density of Active Sites')
    plt.title('RG Flow of Active Site Density')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Optional: Visualize specific decimated lattices for the critical case
    lattice_pc = (np.random.rand(L_SIZE, L_SIZE) < PC_ESTIMATE).astype(int)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    fig.suptitle(f'Visualizing Decimation at Critical Point $p={PC_ESTIMATE}$')

    current = lattice_pc
    steps_to_show = [0, 3, 6, 9] # Original, and after some decimations

    plot_idx = 0
    for i in range(10):
        if i in steps_to_show:
            ax = axes[plot_idx]
            ax.imshow(current, cmap='binary', interpolation='nearest')
            ax.set_title(f'Step {i}\nSize: {current.shape[0]}x{current.shape[0]}')
            ax.axis('off')
            plot_idx += 1
        if current.shape[0] < 2: break
        current = decimate_lattice(current)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ex_2_1_percolation_probability()
    ex_2_2_rg_decimation()