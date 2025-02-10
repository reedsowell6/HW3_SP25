def is_symmetric(A): # Checking if matrix A is symmetric
    for i in range(len(A)):
        for j in range(len(A)):
            if A[i][j] != A[j][i]:
                return False
    return True

def is_positive_definite(A): # Checking to see if matrix A is positive
    try:
        n = len(A)
        for i in range(n):
            sub_matrix = [[A[x][y] for y in range(i+1)] for x in range(i+1)]
            det = determinant(sub_matrix)
            if det <= 0:
                return False
        return True
    except:
        return False

def determinant(A): # Calculating the determinant of matrix A using Laplace
    n = len(A)
    if n == 1:
        return A[0][0]
    if n == 2:
        return A[0][0] * A[1][1] - A[0][1] * A[1][0]
    det = 0
    for i in range(n):
        sub_matrix = [row[:i] + row[i+1:] for row in A[1:]]
        det += ((-1) ** i) * A[0][i] * determinant(sub_matrix)
    return det

def cholesky_decomposition(A): # Performing Cholesky decomposition of matrix A
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            sum_ = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = (A[i][i] - sum_) ** 0.5  # Diagonal elements
            else:
                L[i][j] = (A[i][j] - sum_) / L[j][j]  # Lower triangular elements
    return L

def forward_substitution(L, b): # Solving Ly = b using forward substitution
    n = len(L)
    y = [0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]
    return y

def backward_substitution(U, y): # Solving Ux = y using backward substitution
    n = len(U)
    x = [0] * n
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i+1, n))) / U[i][i]
    return x

def cholesky_solve(A, b): # Solve Ax = b using Cholesky decomposition
    L = cholesky_decomposition(A)
    y = forward_substitution(L, b)
    L_T = [[L[j][i] for j in range(len(L))] for i in range(len(L))]
    x = backward_substitution(L_T, y)
    return x

def doolittle_lu_decomposition(A): # Perform LU decomposition using Doolittle's method
    n = len(A)
    L = [[0] * n for _ in range(n)]
    U = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
        for j in range(i, n):
            if i == j:
                L[i][j] = 1  # Diagonal elements of L are 1
            else:
                L[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]
    return L, U

def doolittle_solve(A, b): # Solve Ax = b using Doolittle decomposition
    L, U = doolittle_lu_decomposition(A)
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    return x

# Define matrices A and vectors b for different systems
systems = [
    ([[1, -1, 3, 2],
      [-1, 5, -5, -2],
      [3, -5, 19, 3],
      [2, -2, 3, 21]],
     [15, -35, 94, 1]),

    ([[4, 2, 4, 0],
      [2, 2, 3, 2],
      [4, 3, 6, 3],
      [0, 2, 3, 9]],
     [20, 36, 60, 122])
]

# Solve each system using appropriate method
for A, b in systems:
    print("\nSolving system:")
    for row, val in zip(A, b):
        print(row, "=", val)
    if is_symmetric(A) and is_positive_definite(A):
        print("Matrix is symmetric and positive definite. Using Cholesky method.")
        x = cholesky_solve(A, b)
    else:
        print("Matrix is not symmetric positive definite. Using Doolittle method.")
        x = doolittle_solve(A, b)
    print("Solution vector x:")
    print(x)

    # Final Edit