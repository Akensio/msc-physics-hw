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

# Prepare plot data for POSITIVE side
b_plot = np.array([0] + list(b_values))
T_plot = np.array([0] + T_c_values)

# Prepare plot data for NEGATIVE side (Symmetry)
b_plot_neg = -b_plot

plt.figure(figsize=(7, 6))
plt.xticks([])
plt.title('2D Phase Diagram')
plt.ylabel('$b/a$')
plt.xlabel('$T$')
plt.ylim(-1.2, 1.2)
plt.xlim(0, 10) 

# --- Asymptotes and Stability Lines ---
# Positive Asymptote
plt.axhline(y=b_crit, color='g', linestyle='--', label=r'Asymptotes $b/a = \pm(\sqrt{2}-1)$')
# Negative Asymptote
plt.axhline(y=-b_crit, color='g', linestyle='--')
# Stability Limits
plt.axhline(y=-1, color='k', linestyle='--', label='Stability Limit')
plt.axhline(y=1, color='k', linestyle='--')

# --- Plotting the Curves ---
# Positive Curve (Ferro)
plt.plot(T_plot, b_plot, 'r-', linewidth=2, label='Phase Boundary $T_c$')
# Negative Curve (Antiferro) - Symmetric
plt.plot(T_plot, b_plot_neg, 'r-', linewidth=2)

# --- Filling the Phases ---

# 1. Ferromagnetic Phase (Top Lobe)
# Fill under the curve
plt.fill_betweenx(b_plot, 0, T_plot, color='lightblue')
# Fill the "Always Ordered" strip above the asymptote
plt.fill_between([0, 20], b_crit, 1, color='lightblue', label='Ferromagnetic (Ordered)')

# 2. Antiferromagnetic Phase (Bottom Lobe)
# Fill "above" the negative curve (mathematically, between 0 and Tc)
plt.fill_betweenx(b_plot_neg, 0, T_plot, color='lightgreen')
# Fill the "Always Ordered" strip below the negative asymptote
plt.fill_between([0, 20], -1, -b_crit, color='lightgreen', label='Antiferromagnetic (Ordered)')

# 3. Disordered Phase (The middle channel)
# We fill everything else with a light color to show the "Disordered River"
plt.fill_betweenx(np.linspace(-b_crit, b_crit, 100), 20, 0, color='lightyellow', alpha=0.3, zorder=-1)


# --- Labels ---
plt.text(2, 0.7, 'Ferromagnetic', ha='center', fontsize=10, color='blue', fontweight='bold')
plt.text(2, -0.7, 'Antiferromagnetic', ha='center', fontsize=10, color='green', fontweight='bold')
plt.text(6, 0.0, 'Disordered', ha='center', fontsize=12)

plt.legend(loc='lower right')
plt.tight_layout()
plt.show()