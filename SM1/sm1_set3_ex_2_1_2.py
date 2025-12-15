import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
J = 1.0
N = 128
T = 2.5
beta = 1.0 / T

# Umbrella Parameters
k = 200.0
m0 = 0.6

# Simulation settings
n_eq = 2000
n_sweeps = 20000 

def get_umbrella_potential(m):
    return 0.5 * k * (m - m0)**2

def metropolis_step_umbrella(spins, current_m):
    for _ in range(N):
        i = np.random.randint(N)
        s = spins[i]
        
        # Proposed change
        # flip s -> -s implies delta_m = -2s/N
        m_new = current_m - (2.0 * s / N)
        
        # Energy Changes
        # 1. Ising Energy
        n_sum = spins[(i-1)%N] + spins[(i+1)%N]
        dE_ising = 2 * J * s * n_sum
        
        # 2. Umbrella Potential
        dW = get_umbrella_potential(m_new) - get_umbrella_potential(current_m)
        
        # Total
        dH = dE_ising + dW
        
        if dH < 0 or np.random.rand() < np.exp(-beta * dH):
            spins[i] *= -1
            current_m = m_new
            
    return spins, current_m

# --- Run Simulation ---
spins = np.random.choice([-1, 1], size=N)
m_current = np.mean(spins)

# Equilibrate
for _ in range(n_eq):
    spins, m_current = metropolis_step_umbrella(spins, m_current)

# Collect Data
m_history = []
for _ in range(n_sweeps):
    spins, m_current = metropolis_step_umbrella(spins, m_current)
    m_history.append(m_current)

# --- Plotting with "Discrete-Aware" Bins ---
# Values of m are: -1, -1 + 2/N, -1 + 4/N, ... 1
# Spacing is 2/N.
# We want bin edges to be at the MIDPOINTS between these values.
# Start edge: -1 - 1/N
# End edge: 1 + 1/N
# Number of edges: N + 2 (to create N+1 bins)
bins = np.linspace(-1 - 1.0/N, 1 + 1.0/N, N + 2)

plt.figure(figsize=(10, 6))
plt.hist(m_history, bins=bins, density=True, color='salmon', edgecolor='black', alpha=0.7)
plt.title(f'Biased Umbrella Sampling ($k={k}, m_0={m0}$)')
plt.xlabel('Magnetization $m$')
plt.ylabel('Biased Probability $P_U(m)$')
plt.axvline(x=0.5, color='red', linestyle='--', label='Target Region')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()