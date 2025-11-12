import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define Symbolic Variables and Constants ---
T = sp.symbols('T')
N = 6
J = 1
k = 1
num_states = 2**N  # This is 2^6 = 64

print(f"Explicitly enumerating all {num_states} states for N={N}...")

# --- 2. Build Z(T) by Brute Force Enumeration ---
Z_bruteforce = 0  # Start our partition function at 0

# Loop from i = 0 to 63
for i in range(num_states):
    
    # --- a. Create the spin configuration for state i ---
    # We use the binary representation of 'i' to get the spins
    # 0 -> '000000' -> [-1, -1, -1, -1, -1, -1] (all down)
    # 1 -> '000001' -> [-1, -1, -1, -1, -1, +1]
    # ...
    # 63 -> '111111' -> [+1, +1, +1, +1, +1, +1] (all up)
    
    # Get binary string (e.g., '1011'), remove '0b', pad with '0's to length N
    binary_string = bin(i)[2:].zfill(N) 
    
    # Map '0' -> -1 (spin down) and '1' -> +1 (spin up)
    spins = [2 * int(bit) - 1 for bit in binary_string]
    
    # --- b. Calculate the energy E for this configuration ---
    E_state = 0
    for j in range(N):
        # The (j + 1) % N handles the periodic boundary condition!
        # When j=5, (j+1)%N = 0, so it calculates spins[5] * spins[0]
        s_i = spins[j]
        s_j = spins[(j + 1) % N]
        E_state += -J * s_i * s_j
        
    # --- c. Add this state's contribution to Z ---
    # This is the symbolic part! We are adding an expression of T.
    Z_bruteforce += sp.exp(-E_state / (k * T))

print("\n--- Brute Force Partition Function Z(T) (before simplification) ---")
# This will be a very long sum of 64 exp() terms
# sp.pprint(Z_bruteforce) # Uncomment this if you want to see the giant mess

# --- 3. Simplify the Result ---
print("\n--- Brute Force Z(T) (after simplification) ---")
# SymPy is smart enough to find all the terms with the same energy
# (degeneracy) and group them together.
Z_simplified = sp.simplify(Z_bruteforce)
sp.pprint(Z_simplified)


# --- 4. (Optional) Prove it's the same as the Transfer Matrix ---
print("\n--- Comparing to Transfer Matrix Method ---")
lambda_1 = 2 * sp.cosh(J / (k * T))
lambda_2 = 2 * sp.sinh(J / (k * T))
Z_transfer = lambda_1**N + lambda_2**N

# This will expand the (cosh+sinh) formula and show it's
# identical to our simplified sum.
difference = sp.simplify(Z_simplified - Z_transfer.expand())
print(f"Difference between methods (should be 0): {difference}")


# --- 5. Now, we proceed exactly as before ---
# The rest of the code is the same, just using Z_simplified

print("\n--- Now Plotting... ---")

# Derive other properties from our simplified Z
F = -k * T * sp.log(Z_simplified)
E = k * T**2 * (Z_simplified.diff(T) / Z_simplified)
Cv = E.diff(T)

# Turn them into plottable functions
E_func = sp.lambdify(T, E, 'numpy')
Cv_func = sp.lambdify(T, Cv, 'numpy')

# Create a range of numerical T values to plot
T_values = np.linspace(0.1, 5.0, 200)

# Calculate the properties at those T values
E_plot = E_func(T_values)
Cv_plot = Cv_func(T_values)

# Create the plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot Average Energy
ax1.plot(T_values, E_plot, label='Average Energy E(T)', color='blue')
ax1.set_ylabel('Energy E')
ax1.set_title(f'1D Ising Model (N={N}, Brute Force Method)')
ax1.grid(True)

# Plot Specific Heat
ax2.plot(T_values, Cv_plot, label='Specific Heat $C_v(T)$', color='red')
ax2.set_xlabel('Temperature (T)')
ax2.set_ylabel('Specific Heat $C_v$')
ax2.grid(True)



plt.tight_layout()
plt.show()

# --- 1. Define Symbolic Variables ---
# T is our symbolic variable (the "param T")
T = sp.symbols('T')
N = 6
J = 1
k = 1  # You set k=1

print(f"Solving for N={N}, J={J}, k={k}")

# --- 2. Build the Symbolic Partition Function Z(T) ---
# Using the Transfer Matrix method
lambda_1 = 2 * sp.cosh(J / (k * T))
lambda_2 = 2 * sp.sinh(J / (k * T))

Z = lambda_1**N + lambda_2**N

print("--- Partition Function Z(T) ---")
sp.pprint(Z)

# --- 3. Derive Other Symbolic Properties ---

# Helmholtz Free Energy: F = -kT * ln(Z)
F = -k * T * sp.log(Z)

# Average Energy: E = -d(ln(Z)) / d(beta) = kT^2 * d(ln(Z)) / dT
# E = T**2 * (1/Z) * Z.diff(T)
E = k * T**2 * (Z.diff(T) / Z)

# Specific Heat: Cv = dE / dT
Cv = E.diff(T)

print("\n--- Symbolic Average Energy E(T) ---")
# E.simplify() can be slow, so we'll just print
sp.pprint(E) 

print("\n--- Symbolic Specific Heat Cv(T) ---")
# This formula will be very large!
# sp.pprint(Cv.simplify()) # .simplify() is very slow here

# --- 4. Turn Symbolic Formulas into Plottable Functions ---
# This is the magic bridge from SymPy to NumPy
# 'numpy' means the created function will use numpy's versions
# of cosh, sinh, exp, etc.
Z_func = sp.lambdify(T, Z, 'numpy')
E_func = sp.lambdify(T, E, 'numpy')
Cv_func = sp.lambdify(T, Cv, 'numpy')

print("\n--- Now Plotting... ---")

# --- 5. Plot the Functions Using NumPy and Matplotlib ---
# Create a range of numerical T values to plot
T_values = np.linspace(0.1, 5.0, 200) # Start from 0.1, not 0

# Calculate the properties at those T values
E_plot = E_func(T_values)
Cv_plot = Cv_func(T_values)

# Create the plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot Average Energy
ax1.plot(T_values, E_plot, label='Average Energy E(T)', color='blue')
ax1.set_ylabel('Energy E')
ax1.set_title(f'1D Ising Model (N={N}, J={J}, B=0)')
ax1.grid(True)

# Plot Specific Heat
ax2.plot(T_values, Cv_plot, label='Specific Heat $C_v(T)$', color='red')
ax2.set_xlabel('Temperature (T)')
ax2.set_ylabel('Specific Heat $C_v$')
ax2.grid(True)



plt.tight_layout()
plt.show()