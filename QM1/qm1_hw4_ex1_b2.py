import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from math import factorial
from scipy.linalg import eigh_tridiagonal

# Constants
l = 1.0
x0 = 0.0
a = 1.647
n_levels = 5
x_min, x_max = -5.0, 5.0
N = 1000

# 1. Analytic (Unperturbed) eigenfunctions
x_full = np.linspace(x_min, x_max, N)
def harmonic_wavefunction(n, x, l=1.0):
    prefactor = 1.0 / np.sqrt(2**n * factorial(n) * np.sqrt(np.pi) * l)
    h_n = hermite(n)
    return prefactor * np.exp(-(x**2) / (2 * l**2)) * h_n(x/l)

# 2. Numerical (Shifted) eigenfunctions
x_shifted = np.linspace(x_min, a, N)
dx = x_shifted[1] - x_shifted[0]
U_shifted = 0.5 * (x_shifted - x0)**2
main_diag = 1.0 / (dx**2) + U_shifted
off_diag = -0.5 / (dx**2) * np.ones(N-1)
vals_shifted, vecs_shifted = eigh_tridiagonal(main_diag, off_diag, select='i', select_range=(0, 5))

# Plotting
plt.figure(figsize=(12, 10))

# Potential and Wall
x_pot = np.linspace(x_min, x_max, N)
U = 0.5 * (x_pot - x0)**2
plt.plot(x_pot, U, 'k-', lw=1.5, alpha=0.5, label='Effective Potential $U(x)$')
plt.axvline(x=a, color='red', lw=3, label='Potential Wall ($x=a$)')

scale = 0.8
colors = plt.cm.plasma(np.linspace(0, 0.8, n_levels))

for n in range(n_levels):
    # Unperturbed
    E_orig = n + 0.5
    psi_orig = harmonic_wavefunction(n, x_full, l)
    prob_orig = psi_orig**2
    plt.plot(x_full, E_orig + prob_orig * scale, color=colors[n], linestyle=':', alpha=0.6)
    plt.fill_between(x_full, E_orig, E_orig + prob_orig * scale, color=colors[n], alpha=0.1)
    
    # Shifted
    E_shift = vals_shifted[n]
    psi_shift = vecs_shifted[:, n]
    prob_shift = (psi_shift**2) / dx * 4 # Times 4 schematically for the plot
    plt.plot(x_shifted, E_shift + prob_shift * (scale * 0.1), color=colors[n], lw=2, label=f'n={n} (Shifted)')
    plt.fill_between(x_shifted, E_shift, E_shift + prob_shift * (scale * 0.1), color=colors[n], alpha=0.3)
    
    # Energy lines
    plt.axhline(y=E_orig, xmax=(x_max-x_min)/(x_max-x_min), color='gray', linestyle=':', lw=1, alpha=0.3)
    plt.axhline(y=E_shift, xmax=(a-x_min)/(x_max-x_min), color=colors[n], linestyle='--', lw=1, alpha=0.5)

plt.title('Landau Levels (Schematically)', fontsize=16)
plt.xlabel('x', fontsize=12)
plt.ylabel('Energy / Probability Density', fontsize=12)
plt.xlim(x_min, x_max)
plt.ylim(0, 7)

# Remove numbers
plt.xticks([])
plt.yticks([])

plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
plt.tight_layout()
plt.savefig('superimposed_landau_levels.png')
