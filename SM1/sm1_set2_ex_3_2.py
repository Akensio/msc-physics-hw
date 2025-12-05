import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# --- Plotting 1D Phase Diagram ---
plt.figure(figsize=(6, 5))
plt.xticks([])
plt.title('1D Phase Diagram')
plt.ylabel('$b/a$')
plt.xlabel('$T$')
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
# Set parameter a = 1 for scaling (results are in units of a)
a = 1.0

# The theoretical asymptote where Tc -> infinity
b_crit = a * (np.sqrt(2) - 1)

def critical_condition(beta, b, a):
    return np.exp(2 * beta * b) + np.sqrt(2) * np.exp(-beta * (a - b)) - (np.sqrt(2) + 1)

# Generate b values approaching b_crit from below
b_values = np.linspace(0.001, b_crit - 0.0001, 300)
T_c_values = []

for b in b_values:
    try:
        # Search for beta in a range allowing very small values (high T)
        beta_sol = brentq(critical_condition, 1e-9, 100, args=(b, a))
        T_c_values.append(1.0 / beta_sol)
    except ValueError:
        T_c_values.append(np.nan)

# Prepare plot data
b_plot = np.array([0] + list(b_values))
T_plot = np.array([0] + T_c_values)

plt.figure(figsize=(7, 5))
plt.xticks([])
plt.title('Corrected 2D Phase Diagram')
plt.ylabel('$b/a$')
plt.xlabel('$T$')
plt.ylim(-1.2, 1.2)
plt.xlim(0, 10) # Limit x-axis to show the divergence clearly

# Asymptote and Stability Lines
plt.axhline(y=b_crit, color='g', linestyle='--', label=r'Asymptote $b/a = \sqrt{2}-1$')
plt.axhline(y=-1, color='k', linestyle='--', label='Stability Limit')
plt.axhline(y=1, color='k', linestyle='--')

# Plot the diverging curve
plt.plot(T_plot, b_plot, 'r-', linewidth=2, label='Phase Boundary $T_c$')

# Fill the phases
# Everything "above/left" of the curve is Ordered
plt.fill_betweenx(b_plot, 0, T_plot, color='lightblue')
plt.fill_between([0, 20], b_crit, 1, color='lightblue', label='Ordered Phase')
# Everything "below/right" is Disordered (only valid for b < b_crit)
plt.fill_betweenx(b_plot, T_plot, 20, color='lightyellow', label='Disordered Phase')

plt.text(1, 0.8, 'Ordered\n(Even at T=$\infty$)', ha='center', fontsize=10, color='blue')
plt.text(6, 0.1, 'Disordered', ha='center', fontsize=12)

plt.legend(loc='lower right')
plt.tight_layout()
plt.show()