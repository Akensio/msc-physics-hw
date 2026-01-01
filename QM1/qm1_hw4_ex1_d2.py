import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

# Parameters
n_levels = 20
W = 6.0
L, R = -W/2, W/2
x0_range = np.linspace(-50, 50, 200)  # Guiding center sweep
visual_x_min, visual_x_max = -10.0, 10.0
visual_y_min, visual_y_max = 0, 60
N = 1000  # Grid points within the well

# Containers for energy data
energies = np.zeros((len(x0_range), n_levels))

# Calculation loop
for i, x0 in enumerate(x0_range):
    # Setup grid for the allowed region (between the two walls)
    x = np.linspace(L, R, N)
    dx = x[1] - x[0]
    
    # Potential: U(x) = 1/2 * (x - x0)^2
    U = 0.5 * (x - x0)**2
    
    # Finite difference Hamiltonian (h_bar = m = omega_c = 1)
    # Energy levels epsilon_n = n + 1/2
    main_diag = 1.0 / (dx**2) + U
    off_diag = -0.5 / (dx**2) * np.ones(len(x)-1)
    
    # Solve for the first n_levels eigenvalues
    # Using 'i' to select indices of eigenvalues
    vals, _ = eigh_tridiagonal(main_diag, off_diag, select='i', select_range=(0, n_levels-1))
    energies[i, :] = vals

# --- Plotting ---
plt.figure(figsize=(12, 8))

# Color palette from previous code
colors = plt.cm.plasma(np.linspace(0, 0.8, n_levels))

for n in range(n_levels):
    # Dispersion curve
    plt.plot(x0_range, energies[:, n], color=colors[n], lw=2.5, label=f'n={n}')
    
    # Bulk energy baseline (Horizontal dotted line)
    # Note: Landau levels are n+0.5
    E_bulk = n + 0.5
    plt.axhline(y=E_bulk, color='gray', linestyle=':', alpha=0.3)

# Add wall markers
plt.axvline(x=L, color='red', lw=2, linestyle='--', label='Left Wall (x=L)')
plt.axvline(x=R, color='red', lw=2, linestyle='-', label='Right Wall (x=R)')

# Aesthetic formatting
plt.title('Energy Dispersion in a Finite Well: $\\epsilon_n$ vs. Guiding Center $x_0$', fontsize=16)
plt.xlabel('Guiding Center Position ($x_0$)', fontsize=14)
plt.ylabel('Energy ($\\epsilon$)', fontsize=14)

# Adjust limits
plt.ylim(visual_y_min, visual_y_max)
plt.xlim(visual_x_min, visual_x_max)

plt.grid(True, which='both', linestyle='--', alpha=0.2)
plt.legend(loc='upper center', fontsize='small', ncol=2)

# Matching "No ticks" preference mentioned in prompt (though I'll keep them subtle or formatted)
plt.xticks([]) 
plt.yticks([])

plt.tight_layout()
plt.savefig('two_wall_dispersion.png')