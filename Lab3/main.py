import csv
import numpy as np
import matplotlib.pyplot as plt


def read_data(filename):
    # Зчитування даних з файлу
    x, y = [], []
    with open(filename, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['Month']))
            y.append(float(row['Temp']))
    return np.array(x), np.array(y)


def form_matrix(x, m):
    # Формування матриці системи рівнянь для МНК (нормальні рівняння)
    A = np.zeros((m + 1, m + 1))
    for i in range(m + 1):
        for j in range(m + 1):
            A[i, j] = np.sum(x ** (i + j))
    return A


def form_vector(x, y, m):
    # Формування вектора вільних членів (права частина)
    b = np.zeros(m + 1)
    for i in range(m + 1):
        b[i] = np.sum(y * (x ** i))
    return b


def gauss_solve(A, b):
    # Метод Гауса з вибором головного елемента по стовпцю
    n = len(b)
    A = A.copy()
    b = b.copy()

    # Прямий хід Гауса
    for k in range(n - 1):
        # Пошук рядка з найбільшим елементом для зменшення похибки округлення
        max_row = np.argmax(np.abs(A[k:n, k])) + k
        A[[k, max_row]] = A[[max_row, k]]
        b[[k, max_row]] = b[[max_row, k]]

        # Обнулення елементів під головною діагоналлю
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # Зворотний хід Гауса (знаходження самих коефіцієнтів)
    x_sol = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x_sol[i] = (b[i] - np.sum(A[i, i + 1:] * x_sol[i + 1:])) / A[i, i]
    return x_sol


def polynomial(x, coef):
    # Обчислення значень полінома за знайденими коефіцієнтами
    y_poly = np.zeros(len(x))
    for i in range(len(coef)):
        y_poly += coef[i] * (x ** i)
    return y_poly


def variance(y_true, y_approx):
    # Обчислення дисперсії (середнього квадрата похибки)
    return np.mean((y_true - y_approx) ** 2)

x_data, y_data = read_data("data.csv")

max_degree = 10
variances = []
degrees = list(range(1, max_degree + 1))

#  Пошук оптимального степеня полінома
print(" ПОШУК ОПТИМАЛЬНОГО СТЕПЕНЯ")
for m in degrees:
    A = form_matrix(x_data, m)
    b_vec = form_vector(x_data, y_data, m)
    coef = gauss_solve(A, b_vec)
    y_approx = polynomial(x_data, coef)
    var = variance(y_data, y_approx)
    variances.append(var)
    print(f"Степінь m={m}: Дисперсія = {var:.2f}")

optimal_m = np.argmin(variances) + 1
print(f"\nНайкращий степінь полінома: m = {optimal_m}")

# Розрахунок фінальної моделі з оптимальним степенем
A_opt = form_matrix(x_data, optimal_m)
b_opt = form_vector(x_data, y_data, optimal_m)
coef_opt = gauss_solve(A_opt, b_opt)
y_approx_opt = polynomial(x_data, coef_opt)

# Екстраполяція (прогноз на 3 місяці вперед)
x_future = np.array([25, 26, 27])
y_future = polynomial(x_future, coef_opt)

print("\n ПРОГНОЗ ТЕМПЕРАТУРИ")
for i in range(3):
    print(f"Місяць {x_future[i]}: очікувана температура {y_future[i]:.2f} °C")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Графік 1: Залежність дисперсії від степеня m
ax1.plot(degrees, variances, marker='o', color='purple')
ax1.set_title("Залежність дисперсії від степеня m")
ax1.set_xlabel("Степінь полінома (m)")
ax1.set_ylabel("Дисперсія (похибка)")
ax1.axvline(x=optimal_m, color='red', linestyle='--', label=f'Найкращий m={optimal_m}')
ax1.grid(True, linestyle='--')
ax1.legend()

# Графік 2: Апроксимація та прогноз (Основний)
x_smooth = np.linspace(min(x_data), max(x_future), 100)
y_smooth = polynomial(x_smooth, coef_opt)

ax2.scatter(x_data, y_data, color='red', label='Фактичні дані', zorder=5)
ax2.plot(x_smooth, y_smooth, color='blue', linewidth=2, label=f'МНК (m={optimal_m})')
ax2.scatter(x_future, y_future, color='green', marker='*', s=150, label='Прогноз (25-27)', zorder=6)
ax2.set_title("Апроксимація та прогноз температур")
ax2.set_xlabel("Місяць")
ax2.set_ylabel("Температура (°C)")
ax2.grid(True, linestyle='--')
ax2.legend()

# Графік 3: Похибка апроксимації (відхилення)
error = y_data - y_approx_opt
ax3.bar(x_data, error, color='orange', alpha=0.7)
ax3.axhline(0, color='black', linewidth=1)
ax3.set_title("Похибка апроксимації")
ax3.set_xlabel("Місяць")
ax3.set_ylabel("Похибка (Відхилення)")
ax3.grid(True, linestyle='--')

plt.tight_layout()
plt.show()