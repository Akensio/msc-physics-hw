import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def V(x):
    # Gaussian potential barrier
    return 1.5 * np.exp(-x**2)

# Parameters
E = 2.0        # Energy > V_max
V_max = 1.5
m = 1.0

# ---------------------------------------------------------
# 1. Real Axis Trajectory
# ---------------------------------------------------------
x_real = np.linspace(-2.5, 2.5, 2000)
p_real_path = np.sqrt(2 * m * (E - V(x_real)))

# ---------------------------------------------------------
# 2. Imaginary Axis Trajectory (Full Symmetric Path)
# ---------------------------------------------------------
# Turning point magnitude on imaginary axis: y^2 = ln(E/V0)
y_max = np.sqrt(np.log(E / V_max))

# FIX: Scan from negative to positive turning point
y_imag = np.linspace(-y_max, y_max, 500)
x_imag_complex = 1j * y_imag

# Momentum is real along this entire imaginary line
# because V(iy) < E for the whole range [-y_max, +y_max]
p_imag_path = np.sqrt(2 * m * (E - V(x_imag_complex)))

# ---------------------------------------------------------
# 3. Plotting
# ---------------------------------------------------------
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# --- Plot 1: Real Axis Motion (Blue) ---
ax.plot(x_real, p_real_path, np.zeros_like(x_real), 
        color='blue', lw=2, label='Real Trajectory (Transmission)')
ax.plot(x_real, -p_real_path, np.zeros_like(x_real), 
        color='blue', lw=2)

# --- Plot 2: Imaginary Axis Motion (Red) ---
# Now plots both the upper and lower branches
ax.plot(np.zeros_like(y_imag), p_imag_path, y_imag, 
        color='blue', lw=3, ls='--', label='Imaginary Path (Reflection)')
ax.plot(np.zeros_like(y_imag), -p_imag_path, y_imag, 
        color='blue', lw=3, ls='--')

# --- Mark Points ---
# Barrier Top
ax.scatter(0, np.sqrt(2*m*(E-V_max)), 0, color='k', s=100, label='Barrier Top (x=0)')

# Complex Turning Points (Top and Bottom)
ax.scatter(0, 0, y_max, color='red', s=100, marker='x', label='Complex Turning Points')
ax.scatter(0, 0, -y_max, color='red', s=100, marker='x')

# --- Formatting ---
ax.set_xlabel('Real Position Re(x)')
ax.set_ylabel('Real Momentum Re(p)')
ax.set_zlabel('Imaginary Position Im(x)')
ax.set_title(f'Symmetric Reflection in Imaginary Time (E={E})')


# Add arrows for flow
# Real path arrows
mid_idx = len(x_real) // 4
ax.quiver(x_real[mid_idx], p_real_path[mid_idx], 0, 1, 0, 0, length=0.1, arrow_length_ratio=1, color='blue')
ax.quiver(x_real[mid_idx], -p_real_path[mid_idx], 0, -1, 0, 0, length=0.1, arrow_length_ratio=1, color='blue')
# Imaginary path arrows (going UP the imaginary axis)
mid_im = len(y_imag) // 2
ax.quiver(0, p_imag_path[mid_im], y_imag[mid_im], 0, 0, 1, length=0.3, arrow_length_ratio=0.5, color='blue')
# Imaginary path arrows (going DOWN the imaginary axis)
ax.quiver(0, -p_imag_path[mid_im], y_imag[mid_im], 0, 0, -1, length=0.3, arrow_length_ratio=0.5, color='blue')

# Adjust view to see the symmetry better
ax.legend()
ax.view_init(elev=20, azim=-45)

plt.show()