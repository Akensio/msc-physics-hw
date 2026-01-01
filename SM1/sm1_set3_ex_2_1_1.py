import numpy as np
import matplotlib.pyplot as plt

# --- Parameters from Problem Statement ---
J = 1.0         # Interaction strength [cite: 19]
N = 128         # Number of spins [cite: 19]
T = 2.5         # Temperature [cite: 19]
beta = 1.0 / T

# Simulation settings
n_eq = 2000     # Sweeps for equilibration (convergence)
n_sweeps = 1000 # Sweeps for data collection (problem suggests ~1000, 10k is smoother)

def calculate_magnetization(spins):
    """Calculate magnetization per spin."""
    return np.mean(spins)

def metropolis_step_direct(spins, current_m):
    """
    Standard single-spin Metropolis step for 1D Ising model.
    Returns updated spins and magnetization.
    """
    for _ in range(N):
        # Pick a random site.
        # While some implementations might take permutations of the sites and iterate these,
        # such practices introduce memory, so to keep the process strictly memoryless
        # and not accidentally break the detailed balance, we pick random sites.
        i = np.random.randint(N)
        s = spins[i]
        
        # Calculate Energy Change (Periodic Boundary Conditions)
        # Neighbors: (i-1) and (i+1) with wrap-around
        n_sum = spins[(i-1)%N] + spins[(i+1)%N]
        dE = 2 * J * s * n_sum
        
        # Metropolis Acceptance Criterion
        if dE < 0 or np.random.rand() < np.exp(-beta * dE):
            spins[i] *= -1
            # Update magnetization incrementally to save time
            current_m += -2 * s / N
            
    return spins, current_m

# --- Simulation ---
# 1. Initialize random spins
spins = np.random.choice([-1, 1], size=N)
m_current = calculate_magnetization(spins)

# 2. Equilibrate (Reach thermal equilibrium)
for _ in range(n_eq):
    spins, m_current = metropolis_step_direct(spins, m_current)

# 3. Collect Data
m_history = []
for _ in range(n_sweeps):
    spins, m_current = metropolis_step_direct(spins, m_current)
    m_history.append(m_current)


# --- Plotting ---
bins = np.linspace(-1 - 1.0/N, 1 + 1.0/N, N + 2)
plt.figure(figsize=(8, 5))
plt.hist(m_history, bins=bins, density=True, color='skyblue', edgecolor='black', alpha=0.7)
plt.title(f'Direct Sampling Distribution ($N={N}, T={T}$)')
plt.xlabel('Magnetization $m$')
plt.ylabel('Probability Density $P_{direct}(m)$')
plt.axvline(x=0.5, color='r', linestyle='--', label='$m=0.5$ threshold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Numerical check for the region m >= 0.5
count_rare = np.sum(np.array(m_history) >= 0.5)
print(f"Number of samples with m >= 0.5: {count_rare}")