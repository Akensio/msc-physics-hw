mydict = {1: (1, 1), 2: (1, -1), 3: (-1, 1), 4: (-1, -1)}

import sympy

# Define , J_1, J_2, and B as symbolic variables
beta, J_1, J_2, B = sympy.symbols("beta J_1 J_2 B")

# This mapping is from Image 1
# (1,0,0,0) -> s_i=1, s_i+1=1
# (0,1,0,0) -> s_i=1, s_i+1=-1
# (0,0,1,0) -> s_i=-1, s_i+1=1
# (0,0,0,1) -> s_i=-1, s_i+1=-1
SIGMA_MAP = {
    (1, 0, 0, 0): (1, 1),
    (0, 1, 0, 0): (1, -1),
    (0, 0, 1, 0): (-1, 1),
    (0, 0, 0, 1): (-1, -1),
}


def get_sigma_pair(vec):
    """Converts a one-hot vector into its (s_i, s_i+1) pair."""
    if vec in SIGMA_MAP:
        return SIGMA_MAP[vec]
    else:
        raise ValueError(f"Vector {vec} is not a valid sigma state.")


def calculate_symbolic_dot_product(vec_j, vec_j_plus_1):
    """
    Calculates the symbolic "dot product" f_tilde between two sigma-tilde vectors.

    This function assumes the "dot product" is 0 if the overlapping spins
    are not consistent.

    It also assumes the formula for f_tilde has a typo, and
    f_tilde(s_j, s_j+1) = f(s_i, s_i+1, s_i+2) + f(s_i+1, s_i+2, s_i)
    """

    # Get the sigma values from the input vectors
    try:
        s_i, s_i_plus_1 = get_sigma_pair(vec_j)
        s_i_plus_2, s_i_plus_3 = get_sigma_pair(vec_j_plus_1)
    except ValueError as e:
        return f"Error: {e}"

    # --- Calculate the symbolic expression ---
    # Based on our assumption:
    # f_tilde = f(s_i, s_i+1, s_i+2) + f(s_i+1, s_i+2, s_i)
    #
    # f(a, b, c) = - J_1*a*b - J_2*a*c - B*a

    # f(s_i, s_i+1, s_i+2)
    f_1 = -J_1 * s_i * s_i_plus_1 - J_2 * s_i * s_i_plus_2 - B * s_i

    # f(s_i+1, s_i+2, s_i_3)
    f_2 = (
        -J_1 * s_i_plus_1 * s_i_plus_2 - J_2 * s_i_plus_1 * s_i_plus_3 - B * s_i_plus_1
    )

    # Sum them up and simplify
    f_tilde = sympy.simplify(sympy.E ** (-beta * (f_1 + f_2)))

    return f_tilde


# Example 1: vec_j = (1,0,0,0) and vec_j+1 = (1,0,0,0)
# s_j:     (s_i=1, s_i+1=1)
# s_j+1:   (s_i+1=1, s_i+2=1)
# Overlap is consistent (s_i+1 = 1 for both)
# # We have: s_i=1, s_i+1=1, s_i+2=1
# v1 = (1, 0, 0, 0)
# v2 = (1, 0, 0, 0)
# result_1 = calculate_symbolic_dot_product(v1, v2)
# print(f"f_tilde({v1}, {v2}) = {result_1}")

basis_vectors = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]

print("Calculating f_tilde for all 16 combinations (vec_j, vec_j_plus_1):")
print("-" * 60)

# Iterate over all combinations
rows: list[list[sympy.Expr]] = []
for vec_j in basis_vectors:
    rows.append([])
    for vec_j_plus_1 in basis_vectors:
        result = calculate_symbolic_dot_product(vec_j, vec_j_plus_1)
        # Format the output for clarity. Using \t for tab alignment.
        rows[-1].append(result)
    print("-" * 60)  # Add a separator for each vec_j

sympy.init_printing()

transfer_matrix = sympy.Matrix(rows)
# sympy.preview(transfer_matrix)

transfer_matrix_2_2_J_negative_1: sympy.Matrix = transfer_matrix.copy().subs({J_1: -1, J_2: -1, B: 0})
# sympy.preview(transfer_matrix_2_2_J_negative_1)
print("\n\nLaTeX format of the transfer matrix for J=-1:")
print(sympy.latex(transfer_matrix_2_2_J_negative_1))
print("\nEigenvalues (J=-1):")
eigenvals_2_2_J_negative_1 = transfer_matrix_2_2_J_negative_1.eigenvals()
# print(sympy.simplify(eigenvals_2_2_J_negative_1))
print(sympy.latex(eigenvals_2_2_J_negative_1))

transfer_matrix_2_2_J_plus_1: sympy.Matrix = transfer_matrix.copy().subs({J_1: 1, J_2: 1, B: 0})
# sympy.preview(transfer_matrix_2_2_J_plus_1)
print("\n\nLaTeX format of the transfer matrix for J=+1:")
print(sympy.latex(transfer_matrix_2_2_J_plus_1))
print("\nEigenvalues (J=+1):")
eigenvals_2_2_J_plus_1 = transfer_matrix_2_2_J_plus_1.eigenvals()
# print(sympy.simplify(eigenvals_2_2_J_plus_1))
print(sympy.latex(sympy.simplify(eigenvals_2_2_J_plus_1)))