import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from math import factorial
from scipy.linalg import eigh_tridiagonal

# Parameters
l = 1.0
x0 = 0.5  # Slightly off-center to see the asymmetry
W = 5.0
L, R = -W/2, W/2
n_levels = 10
x_min, x_max = -50.0, 50.0
visual_x_min, visual_x_max = -8.0, 8.0
visual_y_min, visual_y_max = 0, 20
N = 1000

# 1. Analytic (Unperturbed) eigenfunctions
def harmonic_wavefunction(n, x, x0=0.0, l=1.0):
    prefactor = 1.0 / np.sqrt(2**n * factorial(n) * np.sqrt(np.pi) * l)
    h_n = hermite(n)
    return prefactor * np.exp(-((x - x0)**2) / (2 * l**2)) * h_n((x - x0)/l)

# 2. Numerical (Well-confined) eigenfunctions
x_well = np.linspace(L, R, N)
dx = x_well[1] - x_well[0]
U_well = 0.5 * (x_well - x0)**2
main_diag = 1.0 / (dx**2) + U_well
off_diag = -0.5 / (dx**2) * np.ones(N-1)
vals_well, vecs_well = eigh_tridiagonal(main_diag, off_diag, select='i', select_range=(0, n_levels-1))

# Plotting
plt.figure(figsize=(12, 10))

# Full range grid for unperturbed display
x_full = np.linspace(x_min, x_max, N*2)
U_full = 0.5 * (x_full - x0)**2

# Potential and Walls
plt.plot(x_full, U_full, 'k-', lw=1.5, alpha=0.3, label='Effective Potential $U(x)$')
plt.axvline(x=L, color='red', lw=2.5, linestyle='--', label='Left Wall ($x=L$)')
plt.axvline(x=R, color='red', lw=2.5, linestyle='-', label='Right Wall ($x=R$)')

scale = 1.5
colors = plt.cm.plasma(np.linspace(0, 0.8, n_levels))

for n in range(n_levels):
    # Unperturbed (Schematic dotted background)
    E_orig = n + 0.5
    psi_orig = harmonic_wavefunction(n, x_full, x0, l)
    prob_orig = psi_orig**2
    plt.plot(x_full, E_orig + prob_orig * scale, color=colors[n], linestyle=':', alpha=0.3)
    plt.fill_between(x_full, E_orig, E_orig + prob_orig * scale, color=colors[n], alpha=0.05)
    
    # Well-Confined (Solid shifted levels)
    E_well = vals_well[n]
    psi_well = vecs_well[:, n]
    # Normalizing numerical probability density correctly for plot scale
    prob_well = (psi_well**2) / dx 
    
    plt.plot(x_well, E_well + prob_well * (scale * 0.5), color=colors[n], lw=2.5, label=f'n={n} (Confined)')
    plt.fill_between(x_well, E_well, E_well + prob_well * (scale * 0.5), color=colors[n], alpha=0.3)
    
    # Energy lines
    # Unperturbed line (across whole plot)
    plt.axhline(y=E_orig, color='gray', linestyle=':', lw=1, alpha=0.2)
    # Confined level line (only between walls)
    plt.plot([L, R], [E_well, E_well], color=colors[n], linestyle='--', lw=1.5, alpha=0.6)

plt.title('Landau Levels in a Finite Well (Schematically)', fontsize=16)
plt.xlabel('x', fontsize=12)
plt.ylabel('Energy / Probability Density', fontsize=12)
plt.xlim(visual_x_min, visual_x_max)
plt.ylim(visual_y_min, visual_y_max)

# Ticks off as per user preference
plt.xticks([])
plt.yticks([])

plt.legend(loc='upper right', fontsize='small')
plt.tight_layout()
plt.savefig('two_wall_eigenfunctions.png')