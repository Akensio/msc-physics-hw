import numpy as np
import matplotlib.pyplot as plt

def V(x):
    # Gaussian potential barrier
    return 1.5 * np.exp(-x**2)

x = np.linspace(-4, 4, 1000)
V_x = V(x)
V_max = 1.5
m = 1.0

# Define energies
E_below = 1.0  # E < V_max
E_above = 2.0  # E > V_max

# Calculate momentum
def get_momentum(E, x_arr):
    # 2m(E - V)
    arg = 2 * m * (E - V(x_arr))
    # Filter for classically allowed regions
    # We return arrays with NaNs where forbidden to break the lines in plot
    p = np.sqrt(np.where(arg >= 0, arg, np.nan))
    return p

# Create Plot
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: E < V_max
p_below = get_momentum(E_below, x)
# Plot potential for reference (scaled)
axs[0].plot(x, V_x, 'k--', alpha=0.3, label='Potential V(x)')
axs[0].axhline(E_below, color='g', linestyle=':', label=f'Energy E={E_below}')
# Trajectories
# Left side (x < 0)
axs[0].plot(x[x<0], p_below[x<0], 'b', label='Left Trajectory (Reflected)')
axs[0].plot(x[x<0], -p_below[x<0], 'b')
# Right side (x > 0)
axs[0].plot(x[x>0], p_below[x>0], 'r', label='Right Trajectory (Reflected)')
axs[0].plot(x[x>0], -p_below[x>0], 'r')

axs[0].set_title(f'Phase Space: Below Barrier (E={E_below} < Vmax={V_max})')
axs[0].set_xlabel('Position x')
axs[0].set_ylabel('Momentum p')
axs[0].legend()
axs[0].grid(True)
axs[0].set_xlim(-4, 4)
axs[0].set_ylim(-3, 3)

# Add arrows to indicate direction
# Left trajectory: upper branch goes right, lower goes left
idx_arrow_l = np.searchsorted(x, -2.0)
if not np.isnan(p_below[idx_arrow_l]):
    axs[0].arrow(x[idx_arrow_l], p_below[idx_arrow_l], 0.1, 0, head_width=0.15, head_length=0.1, fc='b', ec='b')
    axs[0].arrow(x[idx_arrow_l], -p_below[idx_arrow_l], -0.1, 0, head_width=0.15, head_length=0.1, fc='b', ec='b')

# Right trajectory: upper branch goes right (out), lower goes left (in)
idx_arrow_r = np.searchsorted(x, 2.0)
if not np.isnan(p_below[idx_arrow_r]):
    axs[0].arrow(x[idx_arrow_r], -p_below[idx_arrow_r], -0.1, 0, head_width=0.15, head_length=0.1, fc='r', ec='r')
    axs[0].arrow(x[idx_arrow_r], p_below[idx_arrow_r], 0.1, 0, head_width=0.15, head_length=0.1, fc='r', ec='r')


# Plot 2: E > V_max
p_above = get_momentum(E_above, x)
axs[1].plot(x, V_x, 'k--', alpha=0.3, label='Potential V(x)')
axs[1].axhline(E_above, color='g', linestyle=':', label=f'Energy E={E_above}')
# Trajectories
axs[1].plot(x, p_above, 'b', label='Right-moving Particle (Transmitted)')
axs[1].plot(x, -p_above, 'r', label='Left-moving Particle (Transmitted)')

axs[1].set_title(f'Phase Space: Above Barrier (E={E_above} > Vmax={V_max})')
axs[1].set_xlabel('Position x')
axs[1].set_ylabel('Momentum p')
axs[1].legend()
axs[1].grid(True)
axs[1].set_xlim(-4, 4)
axs[1].set_ylim(-3, 3)

# Arrows
axs[1].arrow(0, p_above[500], 0.1, 0, head_width=0.15, head_length=0.1, fc='b', ec='b')
axs[1].arrow(0, -p_above[500], -0.1, 0, head_width=0.15, head_length=0.1, fc='r', ec='r')

plt.tight_layout()
plt.savefig('phase_space_barrier.png')
plt.show()