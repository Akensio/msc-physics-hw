import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

# --- Constants & Grid Parameters (From Conventions) ---
l = 1.0
a = -1.0             # Wall Position
n_levels = 4
N = 1000
x_min = -50.0        # Integration/Solver range
dx = (a - x_min) / (N - 1)
x_grid = np.linspace(x_min, a, N)

# Guiding Center Range: from deep bulk (-10) to inside wall (+5)
x0_range = np.linspace(-10, 1, 100)

def get_total_current(x0, n):
    """Solves the Hamiltonian for a given x0 and returns integrated current Iy."""
    # 1. Potential for the current guiding center
    U = 0.5 * (x_grid - x0)**2
    
    # 2. Hamiltonian Construction (eigh_tridiagonal)
    main_diag = 1.0 / (dx**2) + U
    off_diag = -0.5 / (dx**2) * np.ones(N - 1)
    
    # 3. Solve for the specific Landau Level n
    vals, vecs = eigh_tridiagonal(main_diag, off_diag, select='i', select_range=(n, n))
    
    phi = vecs[:, 0]
    # Normalize the 1D density
    rho = (phi**2) / np.trapezoid(phi**2, x_grid)
    
    # 4. Integrated Current Iy
    # Formula (Eq 71/72): Iy = integral[ q * omega * (x0 - x) * rho(x) ]
    # Using q = -1, omega = 1, l = 1
    current_density = -1.0 * (x0 - x_grid) * rho
    total_current = np.trapezoid(current_density, x_grid)
    
    return total_current

# --- Execution ---
plt.figure(figsize=(10, 6))
colors = plt.cm.plasma(np.linspace(0, 0.8, n_levels))

for n in range(n_levels):
    Iy_vals = [get_total_current(x0, n) for x0 in x0_range]
    plt.plot(x0_range, np.abs(Iy_vals), label=f'n={n}', color=colors[n], lw=2.5)

# --- Discussion & Styling ---
plt.title('Total Current $I_y$ vs. Guiding Center $x_0$', fontsize=14)
plt.xlabel('Guiding Center Position ($x_0$)', fontsize=12)
plt.ylabel('Integrated Current $I_y$', fontsize=12)

# Marking the Bulk vs Edge
plt.grid(True, which='both', linestyle=':', alpha=0.5)
plt.legend()
plt.tight_layout()

plt.savefig('total_current_vs_guiding_center.png')