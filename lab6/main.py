import numpy as np
import os
def generate_and_save_data(n=100, x_val=2.5):
    A = np.random.rand(n, n) * 10
    for i in range(n):
        A[i, i] += n * 10
    x_true = np.full(n, x_val)
    B = mat_vec_mult(A, x_true)
    np.savetxt("matrix_A.txt", A, fmt='%.6f')
    np.savetxt("vector_B.txt", B, fmt='%.6f')
    return A, B, x_true

def mat_vec_mult(A, X):
    n = A.shape[0]
    res = np.zeros(n)
    for i in range(n):
        res[i] = sum(A[i, j] * X[j] for j in range(n))
    return res

# максимальна похибка
def vector_norm(V):
    return np.max(np.abs(V))

def lu_decomposition(A):
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    # Задаємо діагональні елементи матриці U рівними 1
    for i in range(n):
        U[i, i] = 1.0
    for k in range(n):
        #Знаходимо k-й стовпець матриці L
        for i in range(k, n):
            sum_L = sum(L[i, j] * U[j, k] for j in range(k))
            L[i, k] = A[i, k] - sum_L
        #Знаходимо k-й рядок матриці U
        for j in range(k + 1, n):
            sum_U = sum(L[k, idx] * U[idx, j] for idx in range(k))
            U[k, j] = (A[k, j] - sum_U) / L[k, k]
    return L, U

def solve_lu(L, U, B):
    n = len(B)
    Z = np.zeros(n)
    X = np.zeros(n)
    # Прямий хід (LZ = B)
    for i in range(n):
        sum_Z = sum(L[i, j] * Z[j] for j in range(i))
        Z[i] = (B[i] - sum_Z) / L[i, i]
    # Зворотний хід (UX = Z)
    for i in range(n - 1, -1, -1):
        sum_X = sum(U[i, j] * X[j] for j in range(i + 1, n))
        X[i] = Z[i] - sum_X
    return X

if __name__ == "__main__":
    n = 100
    print("1.Генерація даних")
    A, B, x_true = generate_and_save_data(n, 2.5)
    print(f"Матриця A ({n}x{n}) та вектор B згенеровані і збережені у файли.")

    print("\n2. LU-розклад")
    L, U = lu_decomposition(A)
    np.savetxt("matrix_LU.txt", L + U, fmt='%.6f')  # Зберігаємо в один файл для зручності
    print("Матриці L та U успішно обчислені та збережені у файл 'matrix_LU.txt'.")

    print("\n3. Розв'язок системи рівнянь")
    X0 = solve_lu(L, U, B)
    print(f"Перші 5 елементів знайденого розв'язку: {X0[:5]}")

    print("\n4. Оцінка точності")
    B_calc = mat_vec_mult(A, X0)
    R0 = B - B_calc
    eps = vector_norm(R0)
    print(f"Початкова похибка (нев'язка) eps: {eps:.2e}")

    print("\n5. Ітераційне уточнення розв'язку")
    eps0_target = 1e-14
    iteration = 0
    X_current = X0.copy()
    current_eps = eps

    while current_eps > eps0_target:
        iteration += 1
        B_curr = mat_vec_mult(A, X_current)
        R = B - B_curr
        current_eps = vector_norm(R)

        if current_eps <= eps0_target:
            break
        dX = solve_lu(L, U, R)
        X_current = X_current + dX
        print(f"Ітерація {iteration}: поточна похибка = {current_eps:.2e}")
        if iteration > 15:
            print("Перевищено ліміт ітерацій!")
            break

    print(f"\n Досягнуто заданої точності!")
    print(f"Кількість проведених ітерацій: {iteration}")
    print(f"Фінальна похибка: {current_eps:.2e}")
    print(f"Перші 5 елементів ідеального розв'язку: {X_current[:5]}")