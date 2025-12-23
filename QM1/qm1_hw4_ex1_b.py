import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from math import factorial

# Constants
l = 1.0
x0 = 0.0
n_levels = 5
x = np.linspace(-5, 5, 500)

def harmonic_wavefunction(n, x, l=1.0):
    prefactor = 1.0 / np.sqrt(2**n * factorial(n) * np.sqrt(np.pi) * l)
    h_n = hermite(n)
    return prefactor * np.exp(-(x**2) / (2 * l**2)) * h_n(x/l)

# Plot 1: Standard Landau Levels
plt.figure(figsize=(10, 8))
U = 0.5 * (x - x0)**2

# REMOVE AXES NUMBERS
plt.xticks([]) 
plt.yticks([])

plt.plot(x, U, 'k-', lw=2, label='Harmonic Oscillator Potential $U(x)$')

# Scale factor for plotting wavefunctions on the energy levels
scale = 0.8

for n in range(n_levels):
    En = n + 0.5
    psi_n = harmonic_wavefunction(n, x, l)
    prob_density = psi_n**2
    
    # Plot energy level line
    plt.axhline(y=En, color='gray', linestyle='--', alpha=0.5)
    
    # Plot probability density shifted by energy
    plt.fill_between(x, En, En + prob_density * scale, alpha=0.4)
    plt.plot(x, En + prob_density * scale, label=f'n={n}')
    plt.text(-4.8, En + 0.1, f'$E_{n}$', verticalalignment='bottom')

plt.ylim(0, 6)
plt.xlim(-5, 5)
plt.xlabel('x')
plt.ylabel('Energy / Probability Density')
plt.title('Landau Levels (Schematically)')
plt.legend(loc='upper right')
plt.grid(True, which='both', linestyle=':', alpha=0.3)
plt.savefig('landau_levels_standard.png')
plt.close()

# Plot 2: Landau Levels with a Potential Barrier (Wall)
# Wall at the "right edge" of the third level (n=2)
# E2 = 2.5. Classical turning point is sqrt(2*2.5) = sqrt(5) approx 2.23.
wall_x = 2.3 

plt.figure(figsize=(10, 8))
x_wall = np.linspace(-5, wall_x, 500)
U_wall = 0.5 * (x_wall - x0)**2

# REMOVE AXES NUMBERS
plt.xticks([]) 
plt.yticks([])

plt.plot(x_wall, U_wall, 'k-', lw=2, label='Harmonic Oscillator Potential $U(x)$')
# Add vertical wall
plt.axvline(x=wall_x, color='red', lw=4, label='Potential Wall $V(x) = \\infty$')

for n in range(n_levels):
    En = n + 0.5
    psi_n = harmonic_wavefunction(n, x_wall, l)
    prob_density = psi_n**2
    
    # For qualitative discussion, we show the "truncated" unperturbed states
    # Note: Real states would be zero at the wall and shift in energy, 
    # but the HW asks for the "shape of the effective potential" as a tool.
    plt.axhline(y=En, xmax=(wall_x + 5)/10, color='gray', linestyle='--', alpha=0.5)
    plt.fill_between(x_wall, En, En + prob_density * scale, alpha=0.4)
    plt.plot(x_wall, En + prob_density * scale)
    plt.text(-4.8, En + 0.1, f'$E_{n}$', verticalalignment='bottom')

plt.ylim(0, 6)
plt.xlim(-5, 5)
plt.xlabel('x')
plt.ylabel('Energy / Probability Density')
plt.title('Landau Levels Near a Potential Wall (Schematically)')
plt.legend(loc='upper left')
plt.grid(True, which='both', linestyle=':', alpha=0.3)
plt.savefig('landau_levels_wall.png')
plt.close()