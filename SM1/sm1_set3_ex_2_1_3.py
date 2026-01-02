import numpy as np
import matplotlib.pyplot as plt
from sm1_set3_ex_2_1_1 import simulate_ising_direct
from sm1_set3_ex_2_1_2 import simulate_ising_umbrella, calculate_bias_potential

# --- 1. Setup & Parameters ---
# (Assumes functions from 2_1_1 and 2_1_2 are imported)
J = 1.0
N = 128
T = 2.5
beta = 1.0 / T
k = 200.0   # Umbrella strength
m0 = 0.8    # Umbrella center

# --- 2. Run Simulations ---
print("Running Direct Sampling (Reference)...")
m_history_direct = simulate_ising_direct(J, beta, N, n_eq=2000, n_sweeps=5000)

print("Running Umbrella Sampling (Biased)...")
m_history_umbrella = simulate_ising_umbrella(J, beta, N, n_eq=2000, n_sweeps=5000, spring_k=k, target_m0=m0)

# --- 3. Histogramming ---
# Define bins exactly the same for both
bins = np.linspace(-1 - 1.0/N, 1 + 1.0/N, N + 2)
bin_centers = 0.5 * (bins[1:] + bins[:-1])
bin_width = bins[1] - bins[0]

# Calculate raw probability densities
P_direct, _ = np.histogram(m_history_direct, bins=bins, density=True)
P_U, _ = np.histogram(m_history_umbrella, bins=bins, density=True)

# --- 4. Reweighting (The Identity) ---
# Identity: P_true(m) = C * P_biased(m) * exp(beta * W(m))
# Calculate the bias potential W(m) for all bin centers
W_m = calculate_bias_potential(bin_centers, k, m0)

# Calculate the reweighting factor e^{beta * W(m)}
reweight_factor = np.exp(beta * W_m)

# Unnormalized Reweighted Distribution
P_US_unnorm = P_U * reweight_factor

# --- 5. Normalization (The "Stitching" Method) ---
# Find C by matching P_direct and P_umbrella_unnorm in their overlap region
overlap_mask = (P_direct > 0.01) & (P_US_unnorm > 0)

if np.sum(overlap_mask) == 0:
    print("Warning: No sufficient overlap found. Check simulation parameters.")
    raise ValueError("No overlap between Direct and Umbrella distributions.")
else:
    # Calculate ratio P_direct / P_umbrella for all overlap bins
    ratios = P_direct[overlap_mask] / P_US_unnorm[overlap_mask]
    C = np.mean(ratios)
    print(f"Normalization Constant C found: {C:.4e}")

# Apply normalization
P_umbrella_reweighted = P_US_unnorm * C

# --- 6. Exact Solution (Gaussian Approximation) ---
eta = np.tanh(beta * J)
sigma2 = (1.0 / N) * (1 + eta) / (1 - eta)
P_exact = (1.0 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-bin_centers**2 / (2 * sigma2))

# --- 7. Plotting ---
plt.figure(figsize=(10, 6))

# A. Plot Direct Sampling (The Middle) - kept as scatter for clarity
plt.plot(bin_centers, P_direct, 'o', label='Direct Sampling', 
         color='skyblue', markersize=4, alpha=0.9, zorder=3)

# B. Plot Reweighted Umbrella as a HISTOGRAM (Bar Chart)
# We use bin_centers for x and P_umbrella_reweighted for height
# We filter out zero values to avoid cluttering the log-plot bottom
valid_indices = P_umbrella_reweighted > 0
plt.bar(bin_centers[valid_indices], P_umbrella_reweighted[valid_indices], 
        width=bin_width, label='Reweighted Umbrella', color='salmon', 
        edgecolor='darkred', alpha=0.6, linewidth=0.5, zorder=2)

# C. Plot Exact Solution
plt.plot(bin_centers, P_exact, 'k--', label='Exact (Gaussian Approx)', linewidth=1.5, alpha=0.8, zorder=4)

# Formatting
# plt.yscale('log') 
plt.xlabel('Magnetization $m$')
plt.ylabel('Probability Density $P(m)$ (Log Scale)')
plt.title(f'Stitching Distributions: Direct vs Umbrella Reweighted\n($N={N}, T={T}, k={k}, m_0={m0}$)')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.xlim(-0.2, 0.8) 
plt.ylim(1e-10, 10) 

plt.show()

# Compare specific values at m >= 0.5
print("\nComparison in Tail Region (m >= 0.5):")
tail_mask = bin_centers >= 0.5
print(f"Mean Prob (Reweighted): {np.mean(P_umbrella_reweighted[tail_mask]):.4e}")
print(f"Mean Prob (Exact):      {np.mean(P_exact[tail_mask]):.4e}")