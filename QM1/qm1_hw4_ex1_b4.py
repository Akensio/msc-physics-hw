import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

# Parameters
n_levels = 6
x0_range = np.linspace(-5, 5, 150)  # Guiding center sweep
a = 0.0                             # Fixed wall position
x_min = -15.0                       # Extended to avoid numerical artifacts
N = 1500

# Containers for energy data
energies = np.zeros((len(x0_range), n_levels))

# Calculation loop
for i, x0 in enumerate(x0_range):
    # Setup grid for the allowed region (left of the wall)
    x = np.linspace(x_min, a, N)
    dx = x[1] - x[0]
    
    # Potential: U(x) = 1/2 * (x - x0)^2
    U = 0.5 * (x - x0)**2
    
    # Finite difference Hamiltonian
    main_diag = 1.0 / (dx**2) + U
    off_diag = -0.5 / (dx**2) * np.ones(len(x)-1)
    
    # Solve for the first n_levels eigenvalues
    vals, _ = eigh_tridiagonal(main_diag, off_diag, select='i', select_range=(0, n_levels-1))
    energies[i, :] = vals

# --- Plotting ---
plt.figure(figsize=(12, 8))

# Apply the plasma color convention from your previous code
colors = plt.cm.plasma(np.linspace(0, 0.8, n_levels))

for n in range(n_levels):
    # Dispersion curve
    plt.plot(x0_range, energies[:, n], color=colors[n], lw=2.5, label=f'n={n}')
    
    # Bulk energy baseline (Horizontal dotted line)
    E_bulk = n + 0.5
    plt.axhline(y=E_bulk, color='gray', linestyle=':', alpha=0.3)

# Aesthetic formatting
plt.axvline(x=a, color='red', lw=2, label='Wall Position')
plt.title('Landau Level Dispersion: Energy $\\epsilon_n$ vs. Guiding Center $x_0$ (Schematically)', fontsize=16)
plt.xlabel('Guiding Center Position ($x_0$)', fontsize=14)
plt.ylabel('Energy', fontsize=14)

# Y-limit adjusted to show the skyrocketing "Wedge" energy
plt.ylim(0, 15)
plt.xlim(x0_range[0], x0_range[-1])

plt.grid(True, which='both', linestyle='--', alpha=0.2)
plt.legend(loc='upper left', fontsize='medium')

# Formatting to match your "No ticks" preference if desired
plt.xticks([]) 
plt.yticks([])

plt.tight_layout()
plt.savefig('dispersion_relation_landau.png')
