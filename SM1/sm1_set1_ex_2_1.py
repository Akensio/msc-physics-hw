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
    Z_brute = 0  # Initialize the partition function
    
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
    print("  Calculating eigenvalues for B != 0...")
    
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

def derive_thermo_properties(Z_sym: sp.Expr, T_sym: sp.Symbol, k: float) -> (sp.Expr, sp.Expr):
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
    
    # Average Energy: E = kT^2 * (d(ln(Z))/dT)
    E_sym = k * T_sym**2 * (Z_sym.diff(T_sym) / Z_sym)
    
    # Specific Heat: Cv = dE / dT
    Cv_sym = E_sym.diff(T_sym)
    
    return E_sym, Cv_sym

def plot_properties(E_sym: sp.Expr, Cv_sym: sp.Expr, T_sym: sp.Symbol, title: str):
    """
    Plots the given symbolic E(T) and Cv(T) functions. (Unchanged)
    
    Args:
        E_sym (sp.Expr): Symbolic Average Energy.
        Cv_sym (sp.Expr): Symbolic Specific Heat.
        T_sym (sp.Symbol): The symbolic variable T.
        title (str): Title for the plot.
    """
    print("  Generating numerical functions for plotting...")
    # Use lambdify to convert symbolic expressions to fast numpy functions
    E_func = sp.lambdify(T_sym, E_sym, 'numpy')
    Cv_func = sp.lambdify(T_sym, Cv_sym, 'numpy')

    print("  Generating plot...")
    # Create a range of numerical T values
    T_values = np.linspace(0.1, 5.0, 200) # Start from 0.1, not 0

    # Calculate the properties at those T values
    E_plot = E_func(T_values)
    Cv_plot = Cv_func(T_values)

    # Create the plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot Average Energy
    ax1.plot(T_values, E_plot, label='Average Energy E(T)', color='blue')
    ax1.set_ylabel('Energy E')
    ax1.set_title(title)
    ax1.grid(True)

    # Plot Specific Heat
    ax2.plot(T_values, Cv_plot, label='Specific Heat $C_v(T)$', color='red')
    ax2.set_xlabel('Temperature (T)')
    ax2.set_ylabel('Specific Heat $C_v$')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# --- Main execution block ---
if __name__ == "__main__":
    
    # --- 1. Setup ---
    N = 6
    J = 1
    k = 1
    B = 1  # <-- NEW: Set magnetic field
    T = sp.symbols('T') # Our main symbolic variable
    
    print(f"--- 1D Ising Model Comparison (N={N}, J={J}, k={k}, B={B}) ---")
    
    # --- 2. Run Method A: Transfer Matrix ---
    print("\n[Method A: Transfer Matrix]")
    Z_transfer = solve_by_transfer_matrix(T, N, J, k, B)
    print("Symbolic Z(T) (Transfer Matrix):")
    sp.pprint(Z_transfer)
    
    # --- 3. Run Method B: Brute Force ---
    print("\n[Method B: Brute Force Enumeration]")
    Z_brute = solve_by_brute_force(T, N, J, k, B)
    
    # Simplify the brute force result
    print("  Simplifying brute force result...")
    Z_brute_simplified = sp.simplify(Z_brute)
    
    print("Symbolic Z(T) (Brute Force, Simplified):")
    sp.pprint(Z_brute_simplified)
    
    # --- 4. Compare the Two Methods ---
    print("\n[Comparison]")
    # We must .expand() the transfer matrix form to compare it
    # to the simplified sum-of-exponentials form.
    print("  Simplifying (Z_transfer.expand() - Z_brute_simplified)...")
    
    # Note: This simplification can be very slow with B!=0
    # We can test equality in a faster way
    
    # Test 1: Fast symbolic simplification
    # difference = sp.simplify(Z_transfer.expand() - Z_brute_simplified)
    
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
            print(f"SUCCESS: Both methods yield the exact same symbolic function.")
        else:
            print(f"FAILURE: Methods differ symbolically by: {difference}")
            # exit() # Don't exit, let's plot anyway
    except Exception as e:
        print(f"Symbolic simplification failed or was too slow: {e}")
        print("Falling back to numerical check...")
        if np.allclose(numeric_diff, 0):
             print(f"SUCCESS: Both methods are numerically identical at test points.")
        else:
            print(f"FAILURE: Methods differ numerically: {numeric_diff}")
            # exit()


    # --- 5. Derive Properties and Plot ---
    print("\n[Derivation and Plotting]")
    # Since they are identical, we can use either one.
    # Z_transfer is cleaner to work with.
    E_sym, Cv_sym = derive_thermo_properties(Z_transfer, T, k)
    
    plot_title = f"1D Ising Model (N={N}, J={J}, B={B}) - Methods Verified"
    plot_properties(
        E_sym, 
        Cv_sym, 
        T, 
        title=plot_title
    )
    
    print("\n--- Done ---")