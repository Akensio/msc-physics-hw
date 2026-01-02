import numpy as np
import matplotlib.pyplot as plt

# Simulation Parameters
J = 1.0         # Interaction strength
N = 128  # Number of spins
T = 2.5         # Temperature
beta = 1.0 / T

# Umbrella Sampling Parameters
k = 200.0 # Harmonic oscillator constant for the bias potential
m0 = 0.6  # Target magnetization center

# Simulation Settings
n_eq = 2000     # Sweeps for equilibration
n_sweeps = 1000 # Sweeps for data collection


# --- Helper Functions ---
def calculate_magnetization(spins):
    """Calculate magnetization per spin."""
    return np.mean(spins)


def calculate_bias_potential(m, spring_k, target_m0):
    """
    Calculate the harmonic bias potential W(m).
    W(m) = 0.5 * k * (m - m0)^2
    """
    return 0.5 * spring_k * (m - target_m0)**2


def metropolis_step_umbrella(spins, current_m, coupling_J, beta, number_of_spins, spring_k, target_m0):
    """
    Metropolis step for 1D Ising model with an added Umbrella Bias.
    The effective Hamiltonian is H_U = H_ising + W(m).
    Returns updated spins and magnetization.
    """
    for _ in range(number_of_spins):
        # Pick a random site, ensuring the memoryless property (for detailed balance)
        i = np.random.randint(number_of_spins)
        s = spins[i]
        
        # Proposed change: flip spin s -> -s
        # The change in magnetization is -2*s / N
        m_new = current_m - (2.0 * s / number_of_spins)
        
        # 1. Calculate Ising Energy Change (dE)
        # Neighbors with periodic boundary conditions
        n_sum = spins[(i-1)%number_of_spins] + spins[(i+1)%number_of_spins]
        dE_ising = 2 * coupling_J * s * n_sum
        
        # 2. Calculate Bias Potential Change (dW)
        # dW = W(m_new) - W(m_old)
        w_old = calculate_bias_potential(current_m, spring_k, target_m0)
        w_new = calculate_bias_potential(m_new, spring_k, target_m0)
        dW = w_new - w_old
        
        # Total change in effective Hamiltonian
        dH_total = dE_ising + dW
        
        # Metropolis Acceptance Criterion
        # We accept if the total effective energy decreases, or with probability exp(-beta * dH)
        if (dH_total < 0) or (np.random.rand() < np.exp(-beta * dH_total)):
            spins[i] *= -1
            current_m = m_new
            
    return spins, current_m


def simulate_ising_umbrella(coupling_J, beta, number_of_spins, n_eq, n_sweeps, spring_k, target_m0):
    """Simulate the 1D Ising model using Umbrella Sampling (biased)."""
    # --- Simulation ---
    # 1. Initialize random spins
    spins = np.random.choice([-1, 1], size=number_of_spins)
    m_current = calculate_magnetization(spins)
    
    # 2. Equilibrate (Reach thermal equilibrium with the BIAS turned on)
    for _ in range(n_eq):
        spins, m_current = metropolis_step_umbrella(
            spins, m_current, coupling_J, beta, number_of_spins, spring_k, target_m0
        )
        
    # 3. Collect Data
    m_history = []
    for _ in range(n_sweeps):
        spins, m_current = metropolis_step_umbrella(
            spins, m_current, coupling_J, beta, number_of_spins, spring_k, target_m0
        )
        m_history.append(m_current)
        
    return m_history


# Run the Simulation
m_history_umbrella = simulate_ising_umbrella(J, beta, N, n_eq, n_sweeps, k, m0)


# --- Plotting ---
# Bin edges are placed at midpoints between possible magnetization values.
bins = np.linspace(-1 - 1.0/N, 1 + 1.0/N, N + 2)

plt.figure(figsize=(10, 6))
plt.hist(m_history_umbrella, bins=bins, density=True, color='salmon', edgecolor='black', alpha=0.7)

plt.title(f'Biased Umbrella Sampling Distribution\n($N={N}, T={T}, k={k}, m_0={m0}$)')
plt.xlabel('Magnetization $m$')
plt.ylabel('Biased Probability Density $P_U(m)$')

plt.axvline(x=0.5, color='r', linestyle='--', label='$m=0.5$ threshold')
plt.axvline(x=m0, color='b', linestyle='--', label=f'Umbrella Bias Center $m_0={m0}$')

plt.legend()
plt.grid(True, alpha=0.3)
plt.show()