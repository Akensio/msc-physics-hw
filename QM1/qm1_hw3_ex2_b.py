import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def V(x):
    # Gaussian potential barrier
    return 1.5 * np.exp(-x**2)

x = np.linspace(-2.5, 2.5, 2000)
E = 1.0  # Energy below V_max
m = 1.0

# Calculate momentum (complex)
# p = sqrt(2m(E - V))
p_complex = np.sqrt(2 * m * (E - V(x)).astype(complex))

# Separate Real and Imaginary parts
p_real = np.real(p_complex)
p_imag = np.imag(p_complex)

# Create 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 1. Plot Real Trajectories (Blue)
# Identify forbidden region indices where E < V(x)
forbidden_mask = E < V(x)

# Create arrays for plotting real trajectories
p_real_plot = p_real.copy()
# Insert NaNs in the forbidden region to break the blue line
p_real_plot[forbidden_mask] = np.nan 

# Plot Real parts (Blue)
ax.plot(x, p_real_plot, np.zeros_like(x), 'b', lw=2, label='Real Trajectories (Re(p))')
ax.plot(x, -p_real_plot, np.zeros_like(x), 'b', lw=2)

# Create arrays for plotting imaginary trajectories
p_imag_plot = p_imag.copy()
# Insert NaNs in the allowed region so the green line only appears in the tunnel
p_imag_plot[~forbidden_mask] = np.nan

ax.plot(x, np.zeros_like(x), p_imag_plot, 'r--', lw=2, label='Tunneling/Instanton (Im(p))')
ax.plot(x, np.zeros_like(x), -p_imag_plot, 'r--', lw=2)

# Labels
ax.set_xlabel('Position x')
ax.set_ylabel('Real Momentum Re(p)')
ax.set_zlabel('Imaginary Momentum Im(p)')
ax.set_title(f'3D Phase Space with Tunneling Path (E={E})')

# Mark the turning points
turning_indices = np.where(np.diff(forbidden_mask.astype(int)) != 0)[0]
for idx in turning_indices:
    ax.scatter(x[idx], 0, 0, color='k', s=50, label='Turning Points' if idx == turning_indices[0] else "")

ax.legend()
ax.view_init(elev=20, azim=-60)

plt.show()