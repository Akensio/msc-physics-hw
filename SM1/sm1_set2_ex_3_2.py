import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# Set parameter a = 1 for scaling (results are in units of a)
a = 1.0

# Define the function for 2D Critical Temperature derived in the solution
# f(beta) = exp(2*beta*b) + sqrt(2)*exp(-beta*(a-b)) - (sqrt(2) + 1) = 0
def critical_condition(beta, b, a):
    return np.exp(2 * beta * b) + np.sqrt(2) * np.exp(-beta * (a - b)) - (np.sqrt(2) + 1)

# --- Calculation for 2D Phase Diagram ---

# The theoretical range for b where Tc > 0 is (0, a / (sqrt(2) + 1))
b_max = a / (np.sqrt(2) + 1)

# Generate b values within this range to solve for Tc
b_values = np.linspace(0.00001, b_max - 0.00001, 1000)
T_c_values = []

for b in b_values:
    # Solve for beta (inverse Temperature)
    try:
        beta_sol = brentq(critical_condition, 0.001, 100, args=(b, a))
        T_c_values.append(1.0 / beta_sol)
    except ValueError:
        T_c_values.append(0)

# Prepare data for plotting the curve
# We add 0 at the start (b=0) and end (b=b_max) to close the loop
b_plot = np.array([0] + list(b_values) + [b_max])
T_plot = np.array([0] + T_c_values + [0])


# --- Plotting 1D Phase Diagram ---
plt.figure(figsize=(6, 5))
plt.title('1D Phase Diagram')
plt.ylabel('$b/a$')
plt.xlabel('$k_B T / a$')
plt.ylim(-1.2, 1.2)
plt.xlim(0, 2)

# Stability limits
plt.axhline(y=-1, color='k', linestyle='--', label='Stability Limit ($|b|<a$)')
plt.axhline(y=1, color='k', linestyle='--')

# Shade the valid region (horizontal band)
plt.axhspan(-1, 1, color='lightgray', alpha=0.5)

plt.text(1, 0, 'Paramagnetic / Disordered\n(No Phase Transition)', 
         ha='center', va='center', fontsize=12)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()


# --- Plotting 2D Phase Diagram ---
plt.figure(figsize=(6, 5))
plt.title('2D Phase Diagram')
plt.ylabel('$b/a$')
plt.xlabel('$k_B T / a$')
plt.ylim(-1.2, 1.2)
# Set x-limit slightly larger than the max Tc found
plt.xlim(0, max(T_c_values) * 1.5 if T_c_values else 2)

# Stability limits
plt.axhline(y=-1, color='k', linestyle='--', label='Stability Limit ($|b|<a$)')
plt.axhline(y=1, color='k', linestyle='--')

# Plot Tc curve (x=T, y=b)
plt.plot(T_plot, b_plot, 'r-', linewidth=2, label='Phase Transition Curve')

# Fill ordered region (Ferromagnetic)
# We fill horizontally from x=0 to x=Tc(b) for the range of b
plt.fill_betweenx(b_plot, 0, T_plot, color='lightblue', alpha=0.5, label='Ferromagnetic (Ordered)')

# Label regions
# plt.text(0.2, 0.2, 'Ordered', ha='center', va='center', fontsize=10, color='blue', fontweight='bold')
# plt.text(1.0, -0.5, 'Paramagnetic / Disordered', ha='center', va='center', fontsize=10)
# plt.text(1.0, 0.8, 'Paramagnetic / Disordered', ha='center', va='center', fontsize=10)

plt.legend(loc='upper right')
plt.tight_layout()
plt.show()