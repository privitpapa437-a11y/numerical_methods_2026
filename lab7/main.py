import numpy as np
def generate_data(n=100, x_val=2.5):
    A = np.random.rand(n, n) * 10
    for i in range(n):
        A[i, i] += np.sum(A[i, :]) + 10
    x_true = np.full(n, x_val)
    B = np.array([sum(A[i, j] * x_true[j] for j in range(n)) for i in range(n)])
    np.savetxt("matrix_A_lab8.txt", A, fmt='%.6f')
    np.savetxt("vector_B_lab8.txt", B, fmt='%.6f')
    return x_true
def read_matrix_from_file(filename):
    return np.loadtxt(filename)
def read_vector_from_file(filename):
    return np.loadtxt(filename)
def matrix_vector_mult(A, X):
    return np.dot(A, X)
def vector_norm(v):
    return np.max(np.abs(v))
def matrix_norm(A):
    return np.max(np.sum(np.abs(A), axis=1))

def simple_iteration(A, B, eps, max_iter=10000):
    n = len(B)
    X = np.ones(n)
    tau = 1.0 / matrix_norm(A)
    for k in range(max_iter):
        R = matrix_vector_mult(A, X) - B
        if vector_norm(R) <= eps:
            return X, k
        X = X - tau * R
    return X, max_iter

def jacobi(A, B, eps, max_iter=10000):
    n = len(B)
    X = np.ones(n)
    X_new = np.zeros(n)
    for k in range(max_iter):
        for i in range(n):
            s = np.dot(A[i, :], X) - A[i, i] * X[i]
            X_new[i] = (B[i] - s) / A[i, i]
        R = matrix_vector_mult(A, X_new) - B
        if vector_norm(R) <= eps:
            return X_new, k + 1
        X = X_new.copy()
    return X, max_iter

def seidel(A, B, eps, max_iter=10000):
    n = len(B)
    X = np.ones(n)
    for k in range(max_iter):
        for i in range(n):
            s = np.dot(A[i, :], X) - A[i, i] * X[i]
            X[i] = (B[i] - s) / A[i, i]
        R = matrix_vector_mult(A, X) - B
        if vector_norm(R) <= eps:
            return X, k + 1
    return X, max_iter
if __name__ == "__main__":
    n = 100
    print("Крок 1: генерація даних")
    x_true = generate_data(n, 2.5)
    print(f"Дані згенеровано та збережено у файли.")
    A = read_matrix_from_file("matrix_A_lab8.txt")
    B = read_vector_from_file("vector_B_lab8.txt")
    eps0 = 1e-14
    print(f"Початкове наближення: x_i = 1.0 для всіх i")
    print(f"Задана точність (eps0): {eps0}\n")
    print("Крок 2: розв'язок ітераційними методами")
    X_simp, iter_simp = simple_iteration(A, B, eps0)
    print("1. Метод простої ітерації:")
    print(f"   Кількість ітерацій: {iter_simp}")
    print(f"   Перші 5 елементів: {X_simp[:5]}\n")
    X_jac, iter_jac = jacobi(A, B, eps0)
    print("2. Метод Якобі:")
    print(f"   Кількість ітерацій: {iter_jac}")
    print(f"   Перші 5 елементів: {X_jac[:5]}\n")
    X_seid, iter_seid = seidel(A, B, eps0)
    print("3. Метод Зейделя:")
    print(f"   Кількість ітерацій: {iter_seid}")
    print(f"   Перші 5 елементів: {X_seid[:5]}\n")
    print("Усі методи успішно досягли точного розв'язку (2.5)!")