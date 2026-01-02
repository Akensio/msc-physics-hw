import numpy as np
import matplotlib.pyplot as plt
from sm1_set3_ex_2_1_1 import simulate_ising_direct
from sm1_set3_ex_2_1_2 import simulate_ising_umbrella


# Simulation Parameters
J = 1.0
N = 128
T = 2.5
beta = 1.0 / T
k = 200.0
m0 = 0.8


def get_approximate_solution(beta, J, N, bin_centers):
    """
    Returns the Gaussian approximation for the 1D Ising model magnetization.
    Variance sigma^2 = (1/N) * exp(2*beta*J).
    """
    eta = np.tanh(beta * J)
    sigma2 = (1.0 / N) * (1 + eta) / (1 - eta)
    P_exact = (1.0 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-bin_centers**2 / (2 * sigma2))
    return P_exact


def reweight_and_stitch(m_direct, m_umbrella, k, m0, beta, N):
    """
    Performs the Reweighting and Stitching (normalization) steps.
    
    Returns:
        bin_centers: x-axis values
        P_direct: Normalized probability from direct sampling
        P_reweighted: Normalized, reweighted probability from umbrella sampling
        C: The calculated normalization constant
    """
    # 1. Setup Bins
    bins = np.linspace(-1 - 1.0/N, 1 + 1.0/N, N + 2)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    
    # 2. Raw Histograms
    P_direct, _ = np.histogram(m_direct, bins=bins, density=True)
    P_umbrella_biased, _ = np.histogram(m_umbrella, bins=bins, density=True)
    
    # 3. Reweighting: remove the bias e^{beta * W(m)}
    # (Using the calculate_bias_potential function from 2.1.2)
    W_m = 0.5 * k * (bin_centers - m0)**2
    reweight_factor = np.exp(beta * W_m)
    P_umbrella_unnorm = P_umbrella_biased * reweight_factor
    
    # 4. Stitching: Find C using the overlap region
    # Overlap is where both histograms have valid data
    overlap_mask = (P_direct > 0.01) & (P_umbrella_unnorm > 0)
    
    if np.sum(overlap_mask) == 0:
        print("Warning: No overlap found between Direct and Umbrella. Check k or m0.")
        C = 1.0
    else:
        ratios = P_direct[overlap_mask] / P_umbrella_unnorm[overlap_mask]
        C = np.mean(ratios)
        
    P_reweighted = P_umbrella_unnorm * C
    
    return bin_centers, P_direct, P_reweighted, C


def plot_stitching_results(bin_centers, P_direct, P_reweighted, P_approx, params):
    """
    Plots the comparison: Direct, Umbrella and Gaussian Approximation.
    """
    N, T, k, m0 = params['N'], params['T'], params['k'], params['m0']
    bin_width = bin_centers[1] - bin_centers[0]
    
    plt.figure(figsize=(10, 6))

    # A. Direct Sampling
    plt.bar(bin_centers, P_direct, width=bin_width, label='Direct Sampling', 
             color='skyblue', alpha=0.4, zorder=1)

    # B. Reweighted Umbrella (Histogram/Bar)
    valid = P_reweighted > 0
    plt.bar(bin_centers[valid], P_reweighted[valid], width=bin_width, 
            label='Reweighted Umbrella', color='salmon', edgecolor='darkred', 
            alpha=0.6, linewidth=0.5, zorder=2)

    # C. Calculated Approximate Solution (the Gaussian we found in 2.1.1)
    plt.plot(bin_centers, P_approx, 'k--', label='Calculated Gaussian Approximation', 
             linewidth=1.5, alpha=0.8, zorder=4)

    # plt.yscale('log')
    plt.xlabel('Magnetization $m$')
    plt.ylabel('Probability Density $P(m)$ (Log Scale)')
    plt.title(f'Stitching Distributions: Direct vs Umbrella Reweighted\n($N={N}, T={T}, k={k}, m_0={m0}$)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()

if __name__ == "__main__":
    # 1. Run Simulations (using functions from 2.1.1 and 2.1.2)
    print("Running Simulations...")
    m_history_direct = simulate_ising_direct(J, beta, N, n_eq=2000, n_sweeps=1000)
    m_history_umbrella = simulate_ising_umbrella(J, beta, N, n_eq=2000, n_sweeps=1000, spring_k=k, target_m0=m0)

    # 2. Process Data
    centers, P_direct, P_reweighted, C = reweight_and_stitch(
        m_history_direct, m_history_umbrella, k, m0, beta, N
    )
    print(f"Stitching Constant C: {C:.4e}")

    # 3. Get Approximate Solution for comparison
    P_approx = get_approximate_solution(beta, J, N, centers)

    # 4. Plot
    params = {'N': N, 'T': T, 'k': k, 'm0': m0}
    plot_stitching_results(centers, P_direct, P_reweighted, P_approx, params)