import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
N = 1000
J = 1.0
steps_per_T = 20
T_start = 1.0
T_stop = 0.1
T_steps = 100


def run_metropolis_sweep(spins, counts, q, N, J, beta):
    """
    Performs one Monte Carlo Sweep (N attempts).
    This ensures every spin is updated on average once per sweep.
    """
    for _ in range(N):
        # 1. Pick random spin
        i = np.random.randint(0, N)
        old_state = spins[i]
        
        # 2. Pick new random state
        new_state = np.random.randint(0, q)
        while new_state == old_state:
            new_state = np.random.randint(0, q)
            
        # 3. Calculate Energy Change
        # dE is derived from Mean Field Energy: E = -(J/2N) * sum(N_k^2)
        N_old = counts[old_state]
        N_new = counts[new_state]
        
        # Change in counts: N_new goes up, N_old goes down
        dE = -(J / N) * (N_new - N_old + 1)
        
        # 4. Metropolis Acceptance
        if dE < 0 or np.random.rand() < np.exp(-beta * dE):
            spins[i] = new_state
            counts[old_state] -= 1
            counts[new_state] += 1


def get_magnetization(state_counts, q, N):
    """
    Calculates the scalar order parameter M based on state counts.
    M defined such that M=0 is disordered, M=1 is ordered.
    Formula: M^2 = (q * sum(mk^2) - 1) / (q-1)
    """
    sum_mk_sq = np.sum((state_counts / N)**2)
    M_sq = (q * sum_mk_sq - 1) / (q - 1)
    # Clip to 0 to avoid tiny negative numbers from float precision errors
    return np.sqrt(max(0, M_sq))


def run_mean_field_potts(q, N, T_steps, J, steps_per_temp):
    """
    Main simulation loop.
    Iterates over temperatures and calls the stepping function.
    """
    # Initialize spins random state 0 to q-1
    spins = np.random.randint(0, q, N)
    
    # Initialize counts of each of the q states
    counts = np.zeros(q, dtype=int)
    for s in spins:
        counts[s] += 1
        
    avg_mag = []
    
    for T in T_steps:
        beta = 1.0 / T if T > 0 else 1e6
        m = 0
        
        for step in range(steps_per_temp):
            run_metropolis_sweep(spins, counts, q, N, J, beta)
            
            # Measure Order Parameter (only in second half to let system equilibrate for each temperature)
            if step > steps_per_temp // 2:
                m += get_magnetization(counts, q, N)
                
        # Average the measurements for this temperature
        measurements_count = steps_per_temp // 2
        avg_mag.append(m / measurements_count)

    return avg_mag

# Temperature range
temps = np.linspace(T_start, T_stop, T_steps) 

# --- SIMULATION ---
print("Simulating q=2 (Ising)...")
mag_q2 = run_mean_field_potts(q=2, N=N, T_steps=temps, J=J, steps_per_temp=steps_per_T)

print("Simulating q=3 (Potts)...")
mag_q3 = run_mean_field_potts(q=3, N=N, T_steps=temps, J=J, steps_per_temp=steps_per_T)

# --- PLOTTING ---
plt.figure(figsize=(10, 6))

plt.plot(temps, mag_q2, 'b-o', label='$q=2$ (2nd Order)', markersize=4)

plt.plot(temps, mag_q3, 'r-s', label='$q=3$ (1st Order)', markersize=4)

plt.xlabel('Temperature ($\frac{k_B T}{J}$)')
plt.ylabel('Order Parameter $M$')
plt.title(f'Mean Field Potts Model (N={N})')

plt.grid(True)
plt.legend()
plt.gca().invert_xaxis() 
plt.savefig('mean_field_potts.png')
plt.show()