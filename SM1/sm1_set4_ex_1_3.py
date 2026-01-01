import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_tk_rotated_contours():
    # 1. Setup the T-K Grid
    T_min, T_max = -2, 4
    K_min, K_max = -2, 4
    resolution = 500
    
    t_vals = np.linspace(T_min, T_max, resolution)
    k_vals = np.linspace(K_min, K_max, resolution)
    T, K = np.meshgrid(t_vals, k_vals)
    
    # 2. Transformation Equations
    # We choose an angle 't' that orients the graph nicely
    theta = -0.5
    
    # Calculate a and b at every point in the grid
    a_raw = T * np.cos(theta) - K * np.sin(theta)
    b_raw = T * np.sin(theta) + K * np.cos(theta)
    
    # Apply reflection
    a = a_raw
    b = -b_raw
    c = 1.0
    
    # 3. Define the Regions (Masks)
    # Initialize grid to 0 (which will be '1 Solution')
    regions = np.zeros_like(T)
    
    # Region: 3 Solutions (Ordered)
    # Condition: a < 0
    mask_3sol = (a < 0)
    regions[mask_3sol] = 1 
    
    # Region: 5 Solutions (Metastable)
    # Condition: b < 0 AND 0 < a < b^2/3c
    mask_5sol = (b < 0) & (a > 0) & (a < (b**2 / (3*c)))
    regions[mask_5sol] = 2
    
    # Region: 1 Solution (Disordered) is the default (0)
    
    # 4. Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Define Colors: 0->Green, 1->Blue, 2->Red
    cmap = mcolors.ListedColormap(['lightgreen', 'lightblue', 'lightcoral'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    # Filled Contour Plot
    ax.contourf(T, K, regions, levels=[-0.5, 0.5, 1.5, 2.5], cmap=cmap, norm=norm)
    
    # 5. Add The Lines (Contours)
    # We plot the mathematical boundaries directly
    
    # A) 2nd Order Line (Ising): a = 0
    # We limit this to the region where it's actually 2nd order (b > 0)
    ax.contour(T, K, a, levels=[0], colors='black', linewidths=2, linestyles='solid')
    
    # B) Spinodal Line: a = b^2 / 3c
    # This defines the edge of the 5-solution region
    spinodal_func = a - (b**2)/(3*c)
    ax.contour(T, K, spinodal_func, levels=[0], colors='black', linewidths=1, linestyles='dotted')
    
    # C) 1st Order Transition: a = b^2 / 4c
    # This runs through the middle of the red region
    trans_func = a - (b**2)/(4*c)
    # We mask this to only show where b < 0 (inside the fork)
    # A simple way in contour is to just plot it, but let's be cleaner:
    # We can just plot the contour; it naturally curves correctly.
    # To differentiate from the Ising line, we can just let it flow.
    CS = ax.contour(T, K, trans_func, levels=[0], colors='black', linewidths=2.5, linestyles='solid')

    # 6. Tricritical Point (Where a=0 and b=0)
    # In this rotated frame, it's at T=0, K=0 (unless we shift T,K)
    ax.plot(0, 0, 'o', color='purple', markersize=10, zorder=10, label='TCP')
    
    # Annotations
    # ax.text(1.5, -1, '3 Solutions\n(Ordered)', ha='center', color='black', fontsize=10, fontweight='bold')
    # ax.text(-0.5, 3, '1 Solution\n(Disordered)', ha='center', color='black', fontsize=10, fontweight='bold')
    # ax.text(1.8, 2.0, '5 Solutions', ha='center', color='black', fontsize=9, fontweight='bold', rotation=45)
    
    ax.set_title('Phase Diagram (Rotated & Reflected)\n'
                 '$a = -(T \cos t - K \sin t)$\n'
                 '$b = (T \sin t + K \cos t)$')
    ax.set_xlabel('Temperature (T)')
    ax.set_ylabel('Interaction (K)')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper left')
    
    plt.savefig('rotated_phase_diagram.png')
    plt.show()

plot_tk_rotated_contours()