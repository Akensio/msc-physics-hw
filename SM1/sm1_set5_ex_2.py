from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import numba


def generate_lattice(L, p):
    """Generates a L*L lattice with site probability p."""
    return np.random.rand(L, L) < p


# Question 2.1.1: BFS Algorithm
@numba.njit()
def has_percolation_numba_large(lattice):
    L = lattice.shape[0]
    
    # We allocate memory for numba ahead of time to avoid dynamic resizing (optimization)
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
        
        # Check percolation condition
        if r == L - 1:
            return True
        
        # Check neighbors (it's written in a very straightforward way that makes it faster with Numba)
        
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


def solve_2_1_1():
    print("Running Question 2.1.1 (Percolation Threshold)")
    print("WARNING: THIS IS SLOW")
    L = 2048

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


# Question 2.1.2: Decimation
def decimate_spanning_rule(lattice):
    """
    Coarse grains the lattice by a factor of 2 using the SPANNING rule.
    """
    L_y, L_x = lattice.shape
    new_L_y, new_L_x = L_y // 2, L_x // 2
    
    # 1. Reshape to isolate 2x2 blocks
    # Shape: (Rows of blocks, 2, Cols of blocks, 2)
    view = lattice[:2*new_L_y, :2*new_L_x].reshape(new_L_y, 2, new_L_x, 2)

    TL = view[:, 0, :, 0]
    TR = view[:, 0, :, 1]
    BL = view[:, 1, :, 0]
    BR = view[:, 1, :, 1]

    # Path exists if left column connects OR right column connects
    path_exists = (TL & BL) | (TR & BR)
    
    return path_exists

def solve_2_1_2():
    print("Running Exercise 2.1.2...")
    
    L = 2048
    SCENARIOS = [0.55, 0.593, 0.618, 0.65]
    STEPS = 10

    results = {} 

    fig_img, axes = plt.subplots(len(SCENARIOS), STEPS, figsize=(15, 9))
    fig_img.suptitle("Visualizing RG Flow (Decimation of Single Realization)", fontsize=16)

    for idx, p in enumerate(SCENARIOS):
        current_lattice = generate_lattice(L, p)
        densities = []
        
        # Decimations
        for gen in range(STEPS):
            # Add column headers and row labels, but only on the edges.
            if idx == 0:
                axes[idx, gen].set_title(f"Gen {gen}\n($L={current_lattice.shape[0]}$)", fontsize=12, fontweight='bold')
            if gen == 0:
                axes[idx, gen].set_ylabel(f"$p = {p}$", fontsize=14, fontweight='bold', labelpad=10, rotation=90)

            # Calculate "percentage of active sites"
            density = np.mean(current_lattice)
            densities.append(density)
            
            # Plot the lattice
            ax = axes[idx, gen]
            ax.imshow(current_lattice, cmap='binary', vmin=0, vmax=1)
            ax.set_title(f"Active: {density:.1%}")
            ax.axis('off')
            
            # Decimate for next round
            current_lattice = decimate_spanning_rule(current_lattice)
        
        results[p] = densities

    plt.tight_layout()
    plt.savefig("rg_flow_lattices.png")

    # Plot the RG flow of active site percentages
    plt.figure(figsize=(10, 6))
    generations = range(STEPS)
    
    colors = cm.plasma(np.linspace(0, 0.85, len(results)))
    
    for (p, densities), color in zip(results.items(), colors):
        plt.plot(generations, densities, marker='o', linestyle='-', linewidth=2, color=color, label=f"$p={p}$")

    plt.title("RG Flow: Percentage of Active Sites vs Decimation Step")
    plt.xlabel("Decimation Step (Generation)")
    plt.ylabel("Percentage of Active Sites")
    plt.axhline(0, color='k', linestyle=':', alpha=0.3)
    plt.axhline(1, color='k', linestyle=':', alpha=0.3)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("rg_flow_active_sites.png")


if __name__ == "__main__":
    solve_2_1_1()
    solve_2_1_2()