import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def calculate_state_energy(spins: list, J: float, B: float) -> float:
    """
    Calculates the total energy of a single spin configuration
    using periodic boundary conditions and a magnetic field B.
    
    Args:
        spins (list): A list of +1 (up) or -1 (down) spins.
        J (float): The coupling constant.
        B (float): The external magnetic field.
        
    Returns:
        float: The total energy E = E_interaction + E_magnetic.
    """
    N = len(spins)
    E_interaction = 0
    for j in range(N):
        # (j + 1) % N handles the periodic boundary condition (ring)
        # It wraps the last spin (N-1) back to the first (0).
        s_i = spins[j]
        s_j = spins[(j + 1) % N]
        E_interaction += -J * s_i * s_j
        
    # Add the magnetic field energy: E_mag = -B * sum(s_i)
    E_magnetic = -B * sum(spins)
    
    return E_interaction + E_magnetic

def solve_by_brute_force(T_sym: sp.Symbol, N: int, J: float, k: float, B: float) -> sp.Expr:
    """
    Calculates the symbolic partition function Z(T) by explicitly
    enumerating all 2^N states, including magnetic field B.
    
    Args:
        T_sym (sp.Symbol): The symbolic variable for Temperature.
        N (int): Number of particles.
        J (float): Coupling constant.
        k (float): Boltzmann constant.
        B (float): The external magnetic field.
        
    Returns:
        sp.Expr: The symbolic partition function Z(T).
    """
    print(f"  Enumerating all 2^{N} = {2**N} states...")
    Z_brute = sp.sympify(0)  # <-- FIX 1: Initialize as a SymPy object
    
    num_states = 2**N
    
    # Loop from i = 0 to (2^N - 1)
    for i in range(num_states):
        # --- a. Create the spin configuration for state i ---
        binary_string = bin(i)[2:].zfill(N) 
        spins = [2 * int(bit) - 1 for bit in binary_string]
        
        # --- b. Calculate the energy E for this configuration (now with B) ---
        E_state = calculate_state_energy(spins, J, B)
        
        # --- c. Add this state's symbolic contribution to Z ---
        Z_brute += sp.exp(-E_state / (k * T_sym))
        
    # The result is a long sum of 64 sp.exp() terms
    return Z_brute

def solve_by_transfer_matrix(T_sym: sp.Symbol, N: int, J: float, k: float, B: float) -> sp.Expr:
    """
    Calculates the symbolic partition function Z(T) using the
    closed-form Transfer Matrix eigenvalues for B != 0.
    
    Args:
        T_sym (sp.Symbol): The symbolic variable for Temperature.
        N (int): Number of particles.
        J (float): Coupling constant.
        k (float): Boltzmann constant.
        B (float): The external magnetic field.
        
    Returns:
        sp.Expr: The symbolic partition function Z(T).
    """
    print("  Calculating eigenvalues for B != 0 (Exact)...")
    
    # --- This is the new, more general eigenvalue calculation ---
    # The B=0 case (cosh, sinh) was a simplification of this.
    
    # Helper variables for clarity
    J_T = J / (k * T_sym)
    B_T = B / (k * T_sym)
    
    # The two eigenvalues of the 2x2 transfer matrix are:
    term1 = sp.exp(J_T) * sp.cosh(B_T)
    term2_inside_sqrt = sp.exp(2 * J_T) * sp.sinh(B_T)**2 + sp.exp(-2 * J_T)
    term2 = sp.sqrt(term2_inside_sqrt)
    
    lambda_1 = term1 + term2
    lambda_2 = term1 - term2
    
    # The partition function is still the sum of eigenvalues to the Nth power
    Z_transfer = lambda_1**N + lambda_2**N
    
    return Z_transfer

def solve_by_lambda_approximation(T_sym: sp.Symbol, N: int, J: float, k: float, B: float) -> sp.Expr:
    """
    Calculates the symbolic partition function Z(T) using the
    lambda_1^N approximation. This is valid for large N.
    
    Args:
        T_sym (sp.Symbol): The symbolic variable for Temperature.
        N (int): Number of particles.
        J (float): Coupling constant.
        k (float): Boltzmann constant.
        B (float): The external magnetic field.
        
    Returns:
        sp.Expr: The symbolic partition function Z(T).
    """
    print("  Calculating eigenvalues for B != 0 (Approximation)...")
    
    # --- This is the new, more general eigenvalue calculation ---
    # The B=0 case (cosh, sinh) was a simplification of this.
    
    # Helper variables for clarity
    J_T = J / (k * T_sym)
    B_T = B / (k * T_sym)
    
    # The two eigenvalues of the 2x2 transfer matrix are:
    term1 = sp.exp(J_T) * sp.cosh(B_T)
    term2_inside_sqrt = sp.exp(2 * J_T) * sp.sinh(B_T)**2 + sp.exp(-2 * J_T)
    term2 = sp.sqrt(term2_inside_sqrt)
    
    lambda_1 = term1 + term2
    
    # The partition function is approximated by just the largest eigenvalue
    Z_approx = lambda_1**N
    
    return Z_approx


def derive_thermo_properties(Z_sym: sp.Expr, T_sym: sp.Symbol, k: float) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """
    Derives symbolic expressions for Energy (E) and Specific Heat (Cv)
    from a given partition function Z(T). (This function is unchanged)
    
    Args:
        Z_sym (sp.Expr): The symbolic partition function Z(T).
        T_sym (sp.Symbol): The symbolic variable for Temperature.
        k (float): Boltzmann constant.
        
    Returns:
        tuple (sp.Expr, sp.Expr): A tuple containing (E_sym, Cv_sym).
    """
    print("  Deriving symbolic E(T) and Cv(T)...")
    
    # Free Energy: F = -kT * ln(Z)
    F_sym = -k * T_sym * sp.ln(Z_sym)
    
    # Average Energy: E = kT^2 * (d(ln(Z))/dT)
    E_sym = k * T_sym**2 * (Z_sym.diff(T_sym) / Z_sym)
    
    # Specific Heat: Cv = dE / dT
    Cv_sym = E_sym.diff(T_sym)
    
    return F_sym, E_sym, Cv_sym

def plot_properties(F_brute: sp.Expr, F_transfer: sp.Expr, F_lambda: sp.Expr,
                    diff_abs_sym: sp.Expr, diff_rel_sym: sp.Expr,
                    T_sym: sp.Symbol, title: str):
    """
    Plots the given symbolic Z(T) functions and the approximation error.
    
    Args:
        F_brute (sp.Expr): Symbolic Partition Function (Brute Force).
        F_transfer (sp.Expr): Symbolic Partition Function (Transfer Matrix).
        F_lambda (sp.Expr): Symbolic Partition Function (Lambda Approx).
        diff_abs_sym (sp.Expr): Symbolic absolute difference (Brute - Lambda).
        diff_rel_sym (sp.Expr): Symbolic relative difference (Brute - Lambda) / Brute.
        T_sym (sp.Symbol): The symbolic variable T.
        title (str): Title for the plot.
    """
    print("  Generating numerical functions for plotting...")
    # Use lambdify to convert symbolic expressions to fast numpy functions
    F_brute_func = sp.lambdify(T_sym, F_brute, 'numpy')
    F_transfer_func = sp.lambdify(T_sym, F_transfer, 'numpy')
    F_lambda_func = sp.lambdify(T_sym, F_lambda, 'numpy')

    diff_rel_func = sp.lambdify(T_sym, diff_rel_sym, 'numpy')

    print("  Generating plot...")
    # Create a range of numerical T values
    T_values = np.linspace(0.1, 5, 1000) # Start from 0.1, not 0

    # Calculate the properties at F_brute_func T values
    F_brute_plot = F_brute_func(T_values)
    F_transfer_plot = F_transfer_func(T_values)
    F_lambda_plot = F_lambda_func(T_values)

    diff_rel_plot = diff_rel_func(T_values)

    # --- Figure 1: Z Comparison ---
    fig1, (ax1) = plt.subplots(1, 1, figsize=(10, 8))

    # Plot partition function - Brute Force
    ax1.plot(T_values, F_transfer_plot, label='F(T) Transfer (Exact)', color='blue')
    ax1.plot(T_values, F_brute_plot, label='F(T) Brute Force (Exact)', color='yellow', linestyle='--')
    
    # Plot partition function - Lambda Approximation
    ax1.plot(T_values, F_lambda_plot, label='F(T) Approx. ($\lambda_1^N$)', color='red', linestyle=':') # <-- FIX: Corrected label
    
    ax1.set_ylabel('F', fontsize=20)
    # ax1.set_yscale('log') # Z grows very fast, log scale is better
    ax1.set_title(title, fontsize=20)
    ax1.grid(True)
    ax1.legend()
    
    ax1.set_xlabel('Temperature [J/k]', fontsize=20) # Moved from bottom
    fig1.tight_layout()
    
    # --- Figure 2: Approximation Error (Brute vs Lambda) ---
    fig2, (ax2) = plt.subplots(1, 1, figsize=(10, 10), sharex=True)
    fig2.suptitle("Approximation Error (Brute vs. Lambda Approx.)", fontsize=20)
    
    # Plot Relative Difference
    ax2.plot(T_values, diff_rel_plot, label='(F_brute - F_lambda) / F_brute', color='orange')
    ax2.set_ylabel('Relative Difference (Error)', fontsize=20)
    ax2.set_xlabel('Temperature [J/K]', fontsize=20)
    ax2.grid(True)
    ax2.legend()
    
    fig2.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout for suptitle

    plt.show()


if __name__ == "__main__":
    # --- 1. Setup ---
    N = 12
    J = 1
    k = 1
    B = 0  # <-- Set magnetic field (try 0, 0.5, 1.0)
    T = sp.symbols('T') # Our main symbolic variable
    
    print(f"--- 1D Ising Model Comparison (N={N}, J={J}, k={k}, B={B}) ---")
    

    # --- 2. Run Method A: Brute Force ---
    print("\n[Method A: Brute Force Enumeration]")
    Z_brute = solve_by_brute_force(T, N, J, k, B)

    # Simplify the brute force result
    print("  Simplifying brute force result...")
    Z_brute_simplified = sp.simplify(Z_brute)
    
    print("Symbolic Z(T) (Brute Force, Simplified):")
    sp.pprint(Z_brute_simplified)
    # sp.preview(Z_brute_simplified)


    # --- 3. Run Method B: Transfer Matrix ---
    print("\n[Method B: Transfer Matrix (Exact)]")
    Z_transfer = solve_by_transfer_matrix(T, N, J, k, B)
    print("Symbolic Z(T) (Transfer Matrix):")
    sp.pprint(Z_transfer)
    # sp.preview(Z_transfer)


    # --- 4. Run Method C: Lambda Approximation ---
    print("\n[Method C: Lambda Approximation (N -> inf)]")
    Z_lambda = solve_by_lambda_approximation(T, N, J, k, B)
    print("Symbolic Z(T) (Lambda Approximation):")
    sp.pprint(Z_lambda)
    # sp.preview(Z_lambda) 
    
    # --- 5. Compare the Two Exact Methods ---
    print("\n[Comparison: Brute vs Transfer]")
    # We must .expand() the transfer matrix form to compare it
    # to the simplified sum-of-exponentials form.
    print("  Simplifying (Z_transfer.expand() - Z_brute_simplified)...")

    # Note: This simplification can be very slow with B!=0
    # We can test equality in a faster way
    
    # Test 2: Numerical test (faster)
    # If they are the same, their difference is 0.
    # We'll subtract, lambdify, and check if the result is ~zero.
    diff_func = sp.lambdify(T, Z_transfer - Z_brute_simplified, 'numpy')
    test_T = np.array([0.5, 1.0, 2.0])
    numeric_diff = diff_func(test_T)
    
    # The symbolic expand/simplify is cleaner if it finishes
    # Let's try it first.
    
    try:
        # Try the full symbolic simplification
        difference = sp.simplify(Z_transfer.expand() - Z_brute_simplified)
        
        if difference == 0:
            print("SUCCESS: Both methods yield the exact same symbolic function.")
        else:
            print(f"WARNING: Methods differ symbolically by: {difference}")

        if np.allclose(numeric_diff, 0):
             print("SUCCESS: Both methods are numerically identical at test points.")
        else:
            print(f"WARNING: Methods differ numerically: {numeric_diff}")

    except Exception as e:
        print(f"WARNING: Symbolic simplification failed or was too slow: {e}")


    # --- 5. Derive Thermodynamic Properties ---
    print("\n[Deriving Thermodynamic Properties]")

    F_brute, _, _ = derive_thermo_properties(Z_brute_simplified, T, k)
    sp.preview(F_brute)
    F_transfer, _ ,_ = derive_thermo_properties(Z_transfer, T, k)
    sp.preview(F_transfer)
    F_lambda, _ ,_ = derive_thermo_properties(Z_lambda, T, k)
    sp.preview(F_lambda)


    # --- 6. Calculate Differences for Plotting ---
    print("\n[Calculating Differences vs Approximation]")
    # F_brute is the exact "brute" result
    # F_lambda is the approximation
    diff_abs_sym = F_brute - F_lambda
    diff_rel_sym = (F_brute - F_lambda) / F_brute


    # --- 7. Derive Properties and Plot ---
    print("\n[Plotting]")
    
    plot_title = f"1D Ising Model (N={N}, J={J}, B={B}, k={k}) - F(T) Comparison"
    plot_properties(
        F_brute,
        F_transfer,
        F_lambda,
        diff_abs_sym,
        diff_rel_sym,
        T, 
        title=plot_title
    )
    
    print("\n--- Done ---")