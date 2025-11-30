import sympy as sp
from pathlib import Path
sp.init_printing()

# Make sure output directory exists
OUT_DIR = Path(__file__).parent / "out_2_2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Symbols
BETA: sp.Expr
X: sp.Expr
Q: sp.Expr
R: sp.Expr
BETA, X, Q, R = sp.symbols("beta x Q R")

# Matrix form
M = sp.Matrix(
    [
        [X**4   , X**2  , 1     , X**-2 ],
        [X**-2  , 1     , X**-2 , 1     ],
        [1      , X**-2 , 1     , X**-2 ],
        [X**-2  , 1     , X**2  , X**4  ],
    ]
)

# Calculate eigenvalues, eigenvectors, and diagonalization
eigenvals: dict[sp.Matrix, int] = M.eigenvals()
P_D_pair: tuple[sp.Matrix, sp.Matrix] = M.diagonalize()
P, D = P_D_pair
P: sp.Matrix = sp.simplify(P)
P_inv: sp.Matrix = sp.simplify(P.inv())
sp.preview(P, viewer="file", filename=(OUT_DIR / "eigenvector_matrix_P.png").resolve())

# Make sure diagonalization works
D_verif: sp.Matrix = sp.simplify(P_inv * M * P)
M_verif: sp.Matrix = sp.simplify(P * D * P_inv)
sp.preview(D_verif, viewer="file", filename=(OUT_DIR / "diagonal_matrix_verification.png").resolve())
sp.preview(M_verif, viewer="file", filename=(OUT_DIR / "matrix_reconstruction_verification.png").resolve())

# Get the B matrix
A = sp.Matrix([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0,-1, 0],
    [0, 0, 0,-1],
])
B: sp.Matrix = sp.simplify(P_inv * A * P)
sp.preview(B, viewer="file", filename=(OUT_DIR / "B_matrix.png").resolve())

L_11 = B[0, 0] * B[0, 0]
L_12 = B[0, 1] * B[1, 0]
L_13 = B[0, 2] * B[2, 0]
L_14 = B[0, 3] * B[3, 0]

sp.preview(L_11, viewer="file", filename=(OUT_DIR / "L_11.png").resolve())
sp.preview(L_12, viewer="file", filename=(OUT_DIR / "L_12.png").resolve())
sp.preview(L_13, viewer="file", filename=(OUT_DIR / "L_13.png").resolve())
sp.preview(L_14, viewer="file", filename=(OUT_DIR / "L_14.png").resolve())

# Substitute for J=1
# Matrix
M_J_pos_1 = M.subs(X, sp.exp(BETA))
print("Matrix M (J=1):")
sp.preview(M_J_pos_1, viewer="file", filename=(OUT_DIR / "matrix_M_J_pos_1.png").resolve())
print(sp.latex(sp.simplify(M_J_pos_1)))
# Eigenvalues from diagonal matrix
D_J_pos_1 = D.subs(X, sp.exp(BETA))
sp.preview(D_J_pos_1, viewer="file", filename=(OUT_DIR / "diagonal_matrix_D_J_pos_1.png").resolve())
print("\neigenval1 (J=1):")
print(sp.latex(sp.expand(D_J_pos_1[0, 0])))
print("\neigenval2 (J=1):")
print(sp.latex(sp.expand(D_J_pos_1[1, 1])))
print("\neigenval3 (J=1):")
print(sp.latex(sp.expand(D_J_pos_1[2, 2])))
print("\neigenval4 (J=1):")
print(sp.latex(sp.expand(D_J_pos_1[3, 3])))
# From manual testing, our Perron-Frobenius eigenvalue is 2
L_21_J_pos_1 = B[1, 0] * B[0, 1]
print("\nL_21 (J=1):")
print(sp.latex(sp.simplify(L_21_J_pos_1.subs(X, sp.exp(BETA)))))
L_22_J_pos_1 = B[1, 1] * B[1, 1]
print("\nL_22 (J=1):")
print(sp.latex(sp.simplify(L_22_J_pos_1.subs(X, sp.exp(BETA)))))
L_23_J_pos_1 = B[1, 2] * B[2, 1]
print("\nL_23 (J=1):")
print(sp.latex(sp.simplify(L_23_J_pos_1.subs(X, sp.exp(BETA)))))
L_24_J_pos_1 = B[1, 3] * B[3, 1]
print("\nL_24 (J=1):")
print(sp.latex(sp.simplify(L_24_J_pos_1.subs(X, sp.exp(BETA)))))


# Substitute for J=-1
# Matrix
M_J_neg_1 = M.subs(X, sp.exp(-BETA))
print("Matrix M (J=-1):")
sp.preview(M_J_neg_1, viewer="file", filename=(OUT_DIR / "matrix_M_J_neg_1.png").resolve())
print(sp.latex(sp.simplify(M_J_neg_1)))
# Eigenvalues from diagonal matrix
D_J_neg_1 = D.subs(X, sp.exp(-BETA))
sp.preview(D_J_neg_1, viewer="file", filename=(OUT_DIR / "diagonal_matrix_D_J_neg_1.png").resolve())
print("\neigenval1 (J=-1):")
print(sp.latex(sp.expand(D_J_neg_1[0, 0])))
print("\neigenval2 (J=-1):")
print(sp.latex(sp.expand(D_J_neg_1[1, 1])))
print("\neigenval3 (J=-1):")
print(sp.latex(sp.expand(D_J_neg_1[2, 2])))
print("\neigenval4 (J=-1):")
print(sp.latex(sp.expand(D_J_neg_1[3, 3])))
# From manual testing, our Perron-Frobenius eigenvalue is 2
L_21_J_neg_1 = B[1, 0] * B[0, 1]
print("\nL_21 (J=-1):")
print(sp.latex(sp.simplify(L_21_J_neg_1.subs(X, sp.exp(-BETA)))))
L_22_J_neg_1 = B[1, 1] * B[1, 1]
print("\nL_22 (J=-1):")
print(sp.latex(sp.simplify(L_22_J_neg_1.subs(X, sp.exp(-BETA)))))
L_23_J_neg_1 = B[1, 2] * B[2, 1]
print("\nL_23 (J=-1):")
print(sp.latex(sp.simplify(L_23_J_neg_1.subs(X, sp.exp(-BETA)))))
L_24_J_neg_1 = B[1, 3] * B[3, 1]
print("\nL_24 (J=-1):")
print(sp.latex(sp.simplify(L_24_J_neg_1.subs(X, sp.exp(-BETA)))))