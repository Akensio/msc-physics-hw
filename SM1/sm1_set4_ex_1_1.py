import numpy as np
import matplotlib.pyplot as plt

def plot_ab_graph_fill():
    # Define range for plotting
    a_min, a_max = -2, 2
    b_min, b_max = -2, 2
    C = 1.0

    fig, ax = plt.subplots(figsize=(8, 6))

    # --- 1. Define the regions for fill_between ---
    a_neg = np.linspace(a_min, 0, 100)
    a_pos = np.linspace(0, a_max, 200)

    # Region: 3 Solutions (a < 0)
    ax.fill_between(a_neg, b_min, b_max, color=[0.6, 0.8, 1.0], label='3 Solutions')

    # Region: 5 Solutions (0 < a < b^2/3c and b < 0)
    # This is equivalent to b < -sqrt(3*C*a)
    # To use fill_between, we find the b-boundary: b = -sqrt(3*C*a)
    b_boundary = -np.sqrt(3 * C * a_pos)
    # We only fill where the boundary is within our plot limits
    ax.fill_between(a_pos, b_min, b_boundary, color=[1.0, 0.6, 0.6], label='5 Solutions')

    # Region: 1 Solution (The rest)
    # We fill from the b_boundary up to b_max
    ax.fill_between(a_pos, b_boundary, b_max, color=[0.9, 0.9, 0.9], label='1 Solution')

    # --- 2. Add Boundaries (Lines) ---
    # Vertical line at a=0 (split style as requested previously)
    ax.vlines(0, 0, b_max, colors='black', linestyles='solid', linewidth=1, label='2nd Order Transition')
    ax.vlines(0, b_min, 0, colors='black', linestyles='dashed', linewidth=1)

    # Spinodal Curve: a = b^2 / 3c  => b = -sqrt(3ca)
    # Note: we plot b as a function of a to match the fill
    ax.plot(a_pos, b_boundary, color='black', linewidth=1, linestyle=':', label='$a = b^2/3c$')

    # Transition Line: a = b^2 / 4c => b = -sqrt(4ca)
    b_trans = -np.sqrt(4 * C * a_pos)
    ax.plot(a_pos, b_trans, color='black', linewidth=2, linestyle='-', label='1st Order Transition')

    # --- 3. Aesthetics ---
    ax.set_xlim(a_min, a_max)
    ax.set_ylim(b_min, b_max)
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_title('Phase Diagram using fill_between')

    # Annotations
    ax.text(-1, 0, '3 Solutions', ha='center', fontweight='bold')
    ax.text(1, 1, '1 Solution', ha='center', fontweight='bold')
    ax.text(0.5, -1.5, '5 Solutions', ha='center', fontweight='bold', color='darkred')
    
    # Tricritical point
    ax.plot(0, 0, 'o', color='purple', markersize=8, zorder=5)

    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.savefig('ab_solution_regions.png')

plot_ab_graph_fill()