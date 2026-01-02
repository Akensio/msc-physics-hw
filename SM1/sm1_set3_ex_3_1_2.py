import numpy as np
import matplotlib.pyplot as plt

def solve_yang_lee_zeros(N_values, beta_J=1.0):
    """
    Solves and plots Yang-Lee zeros in the complex rho plane.
    """
    tau = np.exp(-2 * beta_J)
    
    plt.figure(figsize=(10, 10))
    colors = ['r', 'g', 'b']
    
    print(f"Solving for beta*J = {beta_J} (tau = {tau:.4f})")

    for idx, N in enumerate(N_values):
        zeros = []
        # Loop through all N roots of unity for -1
        for k in range(N):
            # u_k is the target ratio lambda_+ / lambda_-
            theta = np.pi * (2 * k + 1) / N
            u_k = np.exp(1j * theta)
            
            # Derived variable K = ((1+u)/(1-u))^2
            term = (1 + u_k) / (1 - u_k)
            K = term**2
            
            # Coefficients for Quadratic: A*rho^2 + B*rho + C = 0
            # A = 1 - K
            # B = 2 + 2K - 4*K*(tau^2)
            # C = 1 - K
            A = 1 - K
            B = 2 + 2*K - 4*K*(tau**2)
            C = 1 - K
            
            # Solve quadratic
            delta_quad = np.sqrt(B**2 - 4*A*C)
            rho1 = (-B + delta_quad) / (2*A)
            rho2 = (-B - delta_quad) / (2*A)
            
            zeros.append(rho1)
            zeros.append(rho2)
        
        zeros = np.array(zeros)
        
        plt.scatter(zeros.real, zeros.imag, label=f'N={N}', s=15, alpha=0.6, c=colors[idx])
        
    # Draw unit circle for reference
    t = np.linspace(0, 2*np.pi, 500)
    plt.plot(np.cos(t), np.sin(t), 'k--', alpha=0.3, label='Unit Circle')
    
    plt.title(f'Yang-Lee Zeros in the Complex $\\rho$ Plane ($\\beta J = {beta_J}$)')
    plt.xlabel('Re($\\rho$)')
    plt.ylabel('Im($\\rho$)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()

# Parameters from problem
# "Choose the value for beta J at your convenience" -> We choose 1.0
# Tweak this list for different amounts of roots
N_list = [10, 50, 500]
solve_yang_lee_zeros(N_list, beta_J=1.0)