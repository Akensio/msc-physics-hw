import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

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