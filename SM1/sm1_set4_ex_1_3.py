import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_tk_rotated_final():
    # 1. Setup the T-K Grid
    T_min, T_max = -2, 4
    K_min, K_max = -2, 4
    resolution = 800  # Increased for smoother lines
    
    t_vals = np.linspace(T_min, T_max, resolution)
    k_vals = np.linspace(K_min, K_max, resolution)
    T, K = np.meshgrid(t_vals, k_vals)
    
    # 2. Transformation Equations (User's theta)
    theta = 0.8
    
    # Calculate a and b
    x_tricritical = 1
    y_tricritical = 1
    T_shifterd = T - x_tricritical
    K_shifterd = K - y_tricritical
    a_raw = T_shifterd * np.cos(theta) + K_shifterd * np.sin(theta)
    b_raw = -T_shifterd * np.sin(theta) + K_shifterd * np.cos(theta)
    
    # Apply reflection/constants
    a = a_raw
    b = -b_raw
    c = 1/10
    
    # 3. Define the Regions (Masks) for coloring
    regions = np.zeros_like(T)
    
    # Region 1: 3 Solutions (a < 0)
    mask_3sol = (a < 0)
    regions[mask_3sol] = 1 
    
    # Region 2: 5 Solutions (b < 0 AND 0 < a < b^2/3c)
    mask_5sol = (b < 0) & (a > 0) & (a < (b**2 / (3*c)))
    regions[mask_5sol] = 2
    
    # 4. Prepare Data for Lines (The "Split" trick)
    # We create copies of the function values and set them to NaN 
    # in the regions where we DON'T want the line drawn.

    # A) 2nd Order Line (Solid): a=0, but only where b > 0
    z_ising_solid = np.copy(a)
    z_ising_solid[b < 0] = np.nan 

    # B) Inner Spinodal (Dashed): a=0, but only where b < 0 
    # (This is the "other half" of the straight line)
    z_ising_dashed = np.copy(a)
    z_ising_dashed[b > 0] = np.nan

    # C) 1st Order Transition (Thick Solid): a = b^2/4c, only where b < 0
    func_first = a - (b**2)/(4*c)
    z_first_order = np.copy(func_first)
    z_first_order[b > 0] = np.nan # Hide in Ising region

    # D) Outer Spinodal (Dotted): a = b^2/3c, only where b < 0
    func_outer = a - (b**2)/(3*c)
    z_outer_spinodal = np.copy(func_outer)
    z_outer_spinodal[b > 0] = np.nan

    # 5. Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Colors
    cmap = mcolors.ListedColormap(['lightgreen', 'lightblue', 'lightcoral'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    # Draw Regions
    ax.contourf(T, K, regions, levels=[-0.5, 0.5, 1.5, 2.5], cmap=cmap, norm=norm)
    
    # Draw Lines using the masked arrays
    
    # 1. Solid Straight Line (Ising Phase Transition)
    ax.contour(T, K, z_ising_solid, levels=[0], colors='black', linewidths=1, linestyles='-')
    
    # 2. Dashed Straight Line (Inner Spinodal / m=0 stability limit)
    ax.contour(T, K, z_ising_dashed, levels=[0], colors='black', linewidths=1.5, linestyles='--')
    
    # 3. Thick Curve (1st Order Phase Transition)
    ax.contour(T, K, z_first_order, levels=[0], colors='black', linewidths=3, linestyles='-')
    
    # 4. Dotted Curve (Outer Spinodal)
    ax.contour(T, K, z_outer_spinodal, levels=[0], colors='black', linewidths=1, linestyles=':')

    # TCP Marker
    ax.plot(x_tricritical, y_tricritical, 'o', color='purple', markersize=10, zorder=10, label='TCP')

    # Make sure the aspect ratio is equal, so the parabola looks correct
    ax.set_aspect('equal')
    
    # Annotations
    ax.set_title('Phase Diagram with Split Lines')
    ax.set_xlabel('Temperature (T)')
    ax.set_ylabel('Interaction (K)')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper left')
    
    # Set limits to match previous view if needed
    ax.set_xlim(-2, 4)
    ax.set_ylim(-2, 4)
    
    plt.savefig('final_phase_diagram.png')
    plt.show()

plot_tk_rotated_final()