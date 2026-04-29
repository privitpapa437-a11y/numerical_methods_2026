import numpy as np
def write_coeffs(filename, coeffs):
    np.savetxt(filename, coeffs, fmt='%.1f')
def read_coeffs(filename):
    return np.loadtxt(filename)
def horner_val(a, x):
    n = len(a) - 1
    res = a[n]
    for i in range(n - 1, -1, -1):
        res = res * x + a[i]
    return res
def newton_horner(a, x0, eps):
    m = len(a) - 1
    x = x0
    for iter_count in range(1, 1000):
        b = np.zeros(m + 1)
        b[m] = a[m]
        for i in range(m - 1, -1, -1):
            b[i] = a[i] + x * b[i + 1]
        c = np.zeros(m + 1)
        c[m] = b[m]
        for i in range(m - 1, 0, -1):
            c[i] = b[i] + x * c[i + 1]
        fx = b[0]
        dfx = c[1]
        x_new = x - fx / dfx
        if abs(x_new - x) < eps and abs(horner_val(a, x_new)) < eps:
            return x_new, iter_count
        x = x_new
    return x, "-"
def lin_method(a, p0, q0, eps):
    p, q = p0, q0
    for iter_count in range(1, 1000):
        b3 = a[3]
        b2 = a[2] - p * b3
        q_new = a[0] / b2
        p_new = (a[1] * b2 - a[0] * b3) / (b2 ** 2)
        if abs(p_new - p) < eps and abs(q_new - q) < eps:
            alpha = -p_new / 2
            beta = np.sqrt(abs(q_new - alpha ** 2))
            return alpha, beta, iter_count
        p, q = p_new, q_new
    return None, None, "-"
if __name__ == "__main__":
    coeffs_true = np.array([-20.0, 22.0, -12.0, 1.0])
    write_coeffs("poly_coeffs.txt", coeffs_true)
    print("1. Коефіцієнти рівняння збережено у файл 'poly_coeffs.txt'")
    a = read_coeffs("poly_coeffs.txt")
    eps = 1e-10
    print(f"\n2. Аналізуємо рівняння: F(x) = {a[3]}*x^3 + {a[2]}*x^2 + {a[1]}*x + {a[0]} = 0")
    x_real, it_real = newton_horner(a, 11.0, eps)
    print(f"Метод Ньютона (схема Горнера) для дійсного кореня:")
    print(f"Корінь: x = {x_real:.10f}")
    print(f"Кількість ітерацій: {it_real}")
    alpha, beta, it_comp = lin_method(a, -1.5, 1.5, eps)
    if alpha is not None:
        print(f"Метод Ліна для комплексних коренів:")
        print(f"Корені: x = {alpha:.10f} +- {beta:.10f} * i")
        print(f"Кількість ітерацій: {it_comp}")
    else:
        print("Метод Ліна не зійшовся.")
