import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from math import factorial

# Constants / domain
l = 1.0
x0 = 0.0
n_levels = 5
x_min, x_max = -5.0, 5.0
x = np.linspace(x_min, x_max, 500)

def harmonic_wavefunction(n, x, l=1.0):
    prefactor = 1.0 / np.sqrt(2**n * factorial(n) * np.sqrt(np.pi) * l)
    h_n = hermite(n)
    return prefactor * np.exp(-(x**2) / (2 * l**2)) * h_n(x/l)

# Shared plotting conventions (from qm1_hw4_ex1_b2.py)
scale = 0.8
colors = plt.cm.plasma(np.linspace(0, 0.8, n_levels))

# Plot 1: Standard Landau Levels
plt.figure(figsize=(12, 10))
U = 0.5 * (x - x0)**2

# Potential and appearance
plt.plot(x, U, 'k-', lw=1.5, alpha=0.5, label='Effective Potential $U(x)$')
plt.xticks([])
plt.yticks([])

for n in range(n_levels):
    En = n + 0.5
    psi_n = harmonic_wavefunction(n, x, l)
    prob_density = psi_n**2

    # Energy line and probability plotted with per-level colors
    plt.axhline(y=En, color='gray', linestyle=':', lw=1, alpha=0.3)
    plt.fill_between(x, En, En + prob_density * scale, color=colors[n], alpha=0.1)
    plt.plot(x, En + prob_density * scale, color=colors[n], label=f'n={n}')
    plt.text(x_min + 0.2, En + 0.1, f'$E_{n}$', verticalalignment='bottom')

plt.xlim(x_min, x_max)
plt.ylim(0, 6)
plt.xlabel('x')
plt.ylabel('Energy / Probability Density')
plt.title('Landau Levels (Schematically)')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
plt.grid(True, which='both', linestyle=':', alpha=0.3)
plt.tight_layout()
plt.savefig('landau_levels_standard.png')
plt.close()

# Plot 2: Landau Levels with a Potential Barrier (Wall)
wall_x = 2.3
plt.figure(figsize=(12, 10))
x_wall = np.linspace(x_min, wall_x, 500)
U_wall = 0.5 * (x_wall - x0)**2

plt.plot(x_wall, U_wall, 'k-', lw=1.5, alpha=0.5, label='Effective Potential $U(x)$')
plt.axvline(x=wall_x, color='red', lw=3, label='Potential Wall ($x=a$)')
plt.xticks([])
plt.yticks([])

for n in range(n_levels):
    En = n + 0.5
    psi_n = harmonic_wavefunction(n, x_wall, l)
    prob_density = psi_n**2

    plt.axhline(y=En, xmax=(wall_x - x_min)/(x_max - x_min), color='gray', linestyle=':', lw=1, alpha=0.3)
    plt.fill_between(x_wall, En, En + prob_density * scale, color=colors[n], alpha=0.3)
    plt.plot(x_wall, En + prob_density * scale, color=colors[n])
    plt.text(x_min + 0.2, En + 0.1, f'$E_{n}$', verticalalignment='bottom')

plt.xlim(x_min, x_max)
plt.ylim(0, 6)
plt.xlabel('x')
plt.ylabel('Energy / Probability Density')
plt.title('Landau Levels Near a Potential Wall (Schematically)')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
plt.grid(True, which='both', linestyle=':', alpha=0.3)
plt.tight_layout()
plt.savefig('landau_levels_wall.png')
plt.close()