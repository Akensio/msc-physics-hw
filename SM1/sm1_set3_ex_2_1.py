import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
J = 1.0
N = 128
T = 2.5
beta = 1.0 / T
k = 200.0         # Spring constant for Umbrella
m0 = 0.6          # Target magnetization for Umbrella
n_sweeps = 1000  # Sweeps for equilibration/collection
n_eq = 2000       # Equilibration sweeps

def calc_energy(spins):
    """Calculate standard 1D Ising energy (PBC)."""
    interactions = spins * np.roll(spins, -1)
    return -J * np.sum(interactions)

def calc_mag(spins):
    """Calculate magnetization per spin."""
    return np.mean(spins)

def get_umbrella_potential(m):
    """Calculate the bias potential W(m)."""
    return 0.5 * k * (m - m0)**2

def metropolis_step_direct(spins, current_m):
    """Standard Metropolis step."""
    for _ in range(N):
        i = np.random.randint(N)
        s = spins[i]
        # Neighbors (PBC)
        n_sum = spins[(i-1)%N] + spins[(i+1)%N]
        dE = 2 * J * s * n_sum
        
        if dE < 0 or np.random.rand() < np.exp(-beta * dE):
            spins[i] *= -1
            # Update magnetization incrementally
            current_m += -2 * s / N
    return spins, current_m

def metropolis_step_umbrella(spins, current_m):
    """Metropolis step with Umbrella Potential."""
    for _ in range(N):
        i = np.random.randint(N)
        s = spins[i]
        n_sum = spins[(i-1)%N] + spins[(i+1)%N]
        
        # 1. Standard Ising Energy Change
        dE_ising = 2 * J * s * n_sum
        
        # 2. Umbrella Potential Change
        m_old = current_m
        m_new = current_m + (-2 * s / N)
        dW = get_umbrella_potential(m_new) - get_umbrella_potential(m_old)
        
        # Total effective energy change
        dH_eff = dE_ising + dW
        
        if dH_eff < 0 or np.random.rand() < np.exp(-beta * dH_eff):
            spins[i] *= -1
            current_m = m_new
            
    return spins, current_m

# --- 1. Direct Sampling ---
spins = np.random.choice([-1, 1], size=N)
m_current = calc_mag(spins)

# Equilibrate
for _ in range(n_eq):
    spins, m_current = metropolis_step_direct(spins, m_current)

# Collect
m_direct_history = []
for _ in range(n_sweeps):
    spins, m_current = metropolis_step_direct(spins, m_current)
    m_direct_history.append(m_current)

# --- 2. Umbrella Sampling ---
spins_u = np.random.choice([-1, 1], size=N)
m_u_current = calc_mag(spins_u)

# Equilibrate
for _ in range(n_eq):
    spins_u, m_u_current = metropolis_step_umbrella(spins_u, m_u_current)

# Collect
m_umbrella_history = []
for _ in range(n_sweeps):
    spins_u, m_u_current = metropolis_step_umbrella(spins_u, m_u_current)
    m_umbrella_history.append(m_u_current)

# --- 3. Analysis & Reweighting ---
bins = np.linspace(-1.1, 1.1, 100)

# Histograms
P_direct, _ = np.histogram(m_direct_history, bins=bins, density=True)
P_umbrella, _ = np.histogram(m_umbrella_history, bins=bins, density=True)
bin_centers = 0.5 * (bins[1:] + bins[:-1])

# Reweighting: P(m) ~ P_U(m) * exp(beta * W(m))
W_m = get_umbrella_potential(bin_centers)
weights = np.exp(beta * W_m)

# To avoid numerical overflow/underflow, we usually work in log space or normalize carefully.
# Here we just multiply directly as numbers shouldn't explode for these params.
P_reweighted_unnorm = P_umbrella * weights

# Normalize the reweighted distribution specifically in the window of interest
# or globally if we had full coverage.
# For comparison, let's normalize the peak to 1 for visualization or density sum to 1.
norm_factor = np.sum(P_reweighted_unnorm) * (bins[1] - bins[0])
if norm_factor > 0:
    P_reweighted = P_reweighted_unnorm / norm_factor
else:
    P_reweighted = np.zeros_like(P_reweighted_unnorm)

# --- Exact Solution (Gaussian Approx for 1D Ising) ---
# Correlation length xi ~ 1/ln(coth(beta J)).
# Variance sigma^2 = (1/N) * exp(2 beta J) * (1 + ...) roughly.
# Exact susceptibility chi = beta * exp(2*beta*J).
chi = beta * np.exp(2 * beta * J)
sigma_m = np.sqrt(chi / (beta * N)) # Fluctuation dissipation thm
P_exact = (1.0 / (np.sqrt(2 * np.pi) * sigma_m)) * np.exp(-bin_centers**2 / (2 * sigma_m**2))


# --- Plotting ---
plt.figure(figsize=(10, 6))

# Plot Direct
plt.plot(bin_centers, P_direct, label='Direct Sampling', color='blue', alpha=0.6)

# Plot Reweighted (Only trusted where we had umbrella samples)
mask = P_umbrella > 0.01 # filter noise
plt.plot(bin_centers[mask], P_reweighted[mask], 'o', label='Reweighted Umbrella', color='red')

# Plot Exact
plt.plot(bin_centers, P_exact, '--', label='Gaussian Approx (Exact Limit)', color='black')

plt.title(f'Magnetization Distribution (N={N}, T={T})')
plt.xlabel('Magnetization m')
plt.ylabel('P(m)')
plt.yscale('log') # Log scale is crucial to see rare events
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.xlim(-0.2, 0.9)
plt.show()