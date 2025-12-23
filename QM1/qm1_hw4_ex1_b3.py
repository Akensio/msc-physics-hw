import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

# Constants
l = 1.0
x0 = 0.0
a = -1  # Wall is 2 units to the left of the center
n_levels = 30
x_min, x_max = -50.0, 50.0
visual_x_min, visual_x_max = -10.0, 10.0
visual_y_min, visual_y_max = 0, 60
N = 1000

# 1. Numerical (Shifted) eigenfunctions
x_shifted = np.linspace(x_min, a, N)
dx = x_shifted[1] - x_shifted[0]
U_shifted = 0.5 * (x_shifted - x0)**2
main_diag = 1.0 / (dx**2) + U_shifted
off_diag = -0.5 / (dx**2) * np.ones(N-1)

# Solve for eigenvalues
vals_shifted, vecs_shifted = eigh_tridiagonal(main_diag, off_diag, 
                                             select='i', select_range=(0, n_levels-1))

# Plotting
plt.figure(figsize=(12, 10))
x_pot = np.linspace(x_min, x_max, N)
plt.plot(x_pot, 0.5*(x_pot-x0)**2, 'k-', lw=1.5, alpha=0.3, label='Potential $U(x)$')
plt.axvline(x=a, color='red', lw=3, label='Potential Wall ($x=a$)')

# SCALE FIX: Reduce wiggle height to see the Energy Shift
wiggle_scale = 0.25 
colors = plt.cm.plasma(np.linspace(0, 0.8, n_levels))

for n in range(n_levels):
    E_orig = n + 0.5
    E_shift = vals_shifted[n]
    
    # Original Energy (Dotted Gray)
    plt.axhline(y=E_orig, xmax=0.5, color='gray', linestyle=':', lw=1, alpha=0.4)
    
    # Shifted Energy baseline (Dashed Color)
    plt.plot([x_min, a], [E_shift, E_shift], color=colors[n], linestyle='--', lw=1.5)
    
    # Plot Wiggles
    psi_shift = vecs_shifted[:, n]
    prob_shift = (psi_shift**2) / dx * 4 # Times 4 schematically for the plot
    plt.plot(x_shifted, E_shift + prob_shift * wiggle_scale, color=colors[n], lw=2)
    plt.fill_between(x_shifted, E_shift, E_shift + prob_shift * wiggle_scale, 
                     color=colors[n], alpha=0.2, label=f'n={n}')

# SPACING LABELS: This proves the gaps are not constant
spacings = np.diff(vals_shifted)
for i in range(len(spacings)):
    y_mid = (vals_shifted[i] + vals_shifted[i+1]) / 2
    plt.text(visual_x_min + 0.5, y_mid, f'Δ={spacings[i]:.2f}', fontsize=10, fontweight='bold')

plt.title('Landau Levels: Shifted Energy and Variable Spacing', fontsize=16)
plt.ylabel('Energy', fontsize=12)
plt.xlim(visual_x_min, visual_x_max)
plt.ylim(visual_y_min, visual_y_max) # Energy rises significantly!
plt.legend(loc='upper right', fontsize='small')
plt.tight_layout()
plt.savefig('fixed_scaling_landau_levels.png')