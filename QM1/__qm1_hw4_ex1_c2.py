import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.special import hermite
from math import factorial

# Constants in dimensionless units (l_B = 1, hbar = 1, m = 1, q = -1)
def phi_bulk(x, x0, n):
    """Bulk Landau eigenfunction (Harmonic Oscillator)"""
    coeff = 1.0 / np.sqrt(2**n * factorial(n) * np.sqrt(np.pi))
    return coeff * np.exp(-(x - x0)**2 / 2.0) * hermite(n)(x - x0)

def current_density(x, x0, n, wall_pos=0):
    """Local current density jy(x) = q*omega*(x0 - x) * rho(x)"""
    # Force wavefunction to zero at and beyond the wall
    psi = phi_bulk(x, x0, n)
    psi[x > wall_pos] = 0 
    # Current density formula from notes
    return (-1.0) * (x0 - x) * (psi**2)

x = np.linspace(-10, 2, 500)

x0_values = np.linspace(-6, 2, 100)
total_currents = []

for x0 in x0_values:
    # Calculate local density across the sample
    jy = current_density(x, x0, n=0)
    # Integrate over x to get total current of the state
    Iy = simpson(jy, x)
    total_currents.append(Iy)

plt.figure(figsize=(10, 6))
plt.plot(x0_values, total_currents, color='purple', linewidth=2)
plt.axvline(x=0, color='r', linestyle='--', label='Wall Position')
plt.title("Total Current $I_y$ vs. Guiding Center $x_0$")
plt.xlabel("Guiding Center $x_0$ ($k_y l_B^2$)")
plt.ylabel("Integrated Current $I_y$")
plt.annotate('Bulk: $I_y \\approx 0$', xy=(-5, 0.05), weight='bold')
plt.annotate('Edge Channel: $I_y$ Rises', xy=(-0.5, 0.4), weight='bold')
plt.grid(True)
plt.show()
