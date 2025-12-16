import numpy as np
import matplotlib.pyplot as plt
import numba as nb

# --- 1. Parameters & Setup (Same as your Part 2 code) ---
J = 1.0
N = 128
T = 2.5
beta = 1.0 / T

# Umbrella Parameters
k = 200.0
m0 = 0.6

# Simulation settings
n_eq = 2000
n_sweeps = 50000  # Increased sweeps slightly to get a smoother histogram for reweighting

@nb.njit()
def get_umbrella_potential(m):
    return 0.5 * k * (m - m0)**2

@nb.njit()
def metropolis_step_umbrella(spins, current_m):
    for _ in range(N):
        i = np.random.randint(N)
        s = spins[i]
        
        # Proposed change: flip s -> -s implies delta_m = -2s/N
        m_new = current_m - (2.0 * s / N)
        
        # Energy Changes
        # 1. Ising Energy (Nearest Neighbor)
        n_sum = spins[(i-1)%N] + spins[(i+1)%N]
        dE_ising = 2 * J * s * n_sum
        
        # 2. Umbrella Potential Change
        dW = get_umbrella_potential(m_new) - get_umbrella_potential(current_m)
        
        # Total dH
        dH = dE_ising + dW
        
        if dH < 0 or np.random.rand() < np.exp(-beta * dH):
            spins[i] *= -1
            current_m = m_new
            
    return spins, current_m

# --- 2. Run Simulation to get P_U(m) ---
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

# --- 3. Reweighting Analysis ---

# Define bins exactly as in Part 2 so they center on valid magnetization steps
bins = np.linspace(-1 - 1.0/N, 1 + 1.0/N, N + 2)

# Get the biased histogram counts (P_U)
# density=True gives a probability density-like scaling, but for discrete reweighting 
# it is often easier to work with raw counts and normalize at the end.
counts, bin_edges = np.histogram(m_history, bins=bins)

# Calculate the center of each bin (these are the valid m values: -1, ..., 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Calculate the Reweighting Factor: e^{+beta * W(m)}
# Note: P_true(m) ~ P_umbrella(m) * exp(beta * W(m))
weights = np.exp(beta * get_umbrella_potential(bin_centers))

# Recover the unnormalized true probability
# We avoid multiplying where counts are 0 to keep things clean
P_recovered_unnormalized = counts * weights

# Normalize the result so the sum of probabilities equals 1
# (This assumes we are viewing this as a PMF over the discrete states)
total_prob = np.sum(P_recovered_unnormalized)

if total_prob > 0:
    P_recovered = P_recovered_unnormalized / total_prob
else:
    P_recovered = P_recovered_unnormalized # Should not happen if data exists

# --- 4. Plotting ---
plt.figure(figsize=(10, 6))

# Plot only the recovered distribution (P_recovered)
# We use a bar plot or stem plot because the data is discrete
width = 2.0/N
plt.bar(bin_centers, P_recovered, width=width, color='skyblue', edgecolor='blue', alpha=0.7, label='Reweighted $P(m)$')

plt.title(f'Reweighted Probability Distribution\n(Recovered from Umbrella Sampling at $m_0={m0}$)')
plt.xlabel('Magnetization $m$')
plt.ylabel('Probability $P(m)$')
# plt.yscale('log') # Log scale is usually best to see the rare event statistics
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()

plt.show()