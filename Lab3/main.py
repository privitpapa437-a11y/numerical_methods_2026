import csv
import numpy as np
import matplotlib.pyplot as plt

def read_data(filename):
    x, y = [], []  # Створюємо два порожні списки: для місяців (x) і температур (y)
    with open(filename, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:  # Проходимося по кожному рядку таблиці
            x.append(float(row['Month']))  # Беремо значення місяця і додаємо в список x
            y.append(float(row['Temp']))  # Беремо значення температури і додаємо в список y
    return np.array(x), np.array(y)  # Перетворюємо списки у масиви numpy і повертаємо їх

def form_matrix(x, m):  # Функція для створення матриці системи рівнянь
    A = np.zeros((m + 1, m + 1))
    for i in range(m + 1):  # Запускаємо цикл по рядках матриці
        for j in range(m + 1):  # Запускаємо цикл по стовпцях матриці
            A[i, j] = np.sum(x ** (i + j))  # Рахуємо суму іксів у відповідному степені і записуємо в клітинку
    return A

def form_vector(x, y, m):  # Функція для створення вектора вільних членів (права частина рівняння)
    b = np.zeros(m + 1)
    for i in range(m + 1):  # Проходимося по кожному елементу цього вектора
        b[i] = np.sum(y * (x ** i))  # Рахуємо суму добутків ігріків на ікси у степені і записуємо
    return b


def gauss_solve(A, b):
    n = len(b)  # Дізнаємося розмір нашої системи (кількість рівнянь)
    A = A.copy()  # Робимо копію матриці, щоб не зіпсувати оригінал
    b = b.copy()  # Робимо копію вектора

    for k in range(n - 1):  # Йдемо по діагоналі матриці
        max_row = np.argmax(np.abs(A[k:n, k])) + k  # Шукаємо рядок з найбільшим числом у поточному стовпці
        A[[k, max_row]] = A[[max_row, k]]  # Міняємо місцями поточний рядок з знайденим максимальним (у матриці)
        b[[k, max_row]] = b[[max_row, k]]  # Робимо таку ж перестановку у векторі відповідей
        for i in range(k + 1, n):  # Йдемо по рядках нижче головного елемента
            factor = A[i, k] / A[k, k]  # Вираховуємо множник, щоб обнулити елемент під діагоналлю
            A[i, k:] -= factor * A[k, k:]  # Віднімаємо від поточного рядка верхній, помножений на множник
            b[i] -= factor * b[k]  # Те саме робимо для вектора відповідей


    x_sol = np.zeros(n)
    for i in range(n - 1, -1, -1):  # Йдемо знизу вгору (від останнього рівняння до першого)
        x_sol[i] = (b[i] - np.sum(A[i, i + 1:] * x_sol[i + 1:])) / A[i, i]  # Вираховуємо невідому змінну за формулою
    return x_sol  # Повертаємо знайдені коефіцієнти нашого полінома

def polynomial(x, coef):  # Функція, яка будує лінію за знайденими коефіцієнтами
    y_poly = np.zeros(len(x))
    for i in range(len(coef)):  # Проходимося по кожному коефіцієнту
        y_poly += coef[i] * (x ** i)  # Додаємо до загальної суми значення цього шматочка полінома
    return y_poly  # Повертаємо готові точки для нашої плавної лінії


def variance(y_true, y_approx):  # Функція для перевірки похибки (наскільки наша лінія крива)
    return np.mean((y_true - y_approx) ** 2)  # Рахуємо середній квадрат відхилення нашої лінії від реальних точок

x_data, y_data = read_data("data.csv")  # Зчитуємо наші місяці і температури з файлу

max_degree = 4
variances = []  # Створюємо кошик, куди будемо записувати похибки кожного полінома

for m in range(1, max_degree + 1):  # Запускаємо цикл для перевірки кожного степеня по черзі
    A = form_matrix(x_data, m)  # Будуємо матрицю для поточного степеня
    b_vec = form_vector(x_data, y_data, m)  # Будуємо вектор для поточного степеня
    coef = gauss_solve(A, b_vec)  # Розв'язуємо це все Гаусом і отримуємо коефіцієнти
    y_approx = polynomial(x_data, coef)
    var = variance(y_data, y_approx)  # Рахуємо, наскільки сильно ця лінія відхиляється від реальних точок
    variances.append(var)
    print(f"Степінь m={m}: Дисперсія (похибка) = {var:.2f}")

optimal_m = np.argmin(variances) + 1  # Знаходимо, який степінь дав найменшу похибку (і додаємо 1, бо індекси з нуля)

A_opt = form_matrix(x_data, optimal_m)  # Створюємо фінальну матрицю для найкращого степеня
b_opt = form_vector(x_data, y_data, optimal_m)  # Створюємо фінальний вектор для найкращого степеня
coef_opt = gauss_solve(A_opt, b_opt)  # Знаходимо ідеальні коефіцієнти
y_approx_opt = polynomial(x_data, coef_opt)  # Рахуємо висоти нашої ідеальної лінії для існуючих 24 місяців


x_future = np.array([25, 26, 27])  # Задаємо нові місяці, яких не було в базі (25, 26, 27)
y_future = polynomial(x_future, coef_opt)  # Просимо нашу ідеальну лінію передбачити температуру для цих місяців

for i in range(3):  # Проходимося по трьох нових місяцях
    print(f"Місяць {x_future[i]}: очікувана температура {y_future[i]:.2f} °C")  # Виводимо передбачену температуру


x_smooth = np.linspace(min(x_data), max(x_future),100)
y_smooth = polynomial(x_smooth, coef_opt)  # Рахуємо ідеальні висоти для цих 100 точок

plt.figure(figsize=(12, 6))  # Створюємо велике полотно для малювання
plt.scatter(x_data, y_data, color='red', label='Фактичні дані (з файлу)',zorder=5)  # Ставимо червоні крапки на реальні температури
plt.plot(x_smooth, y_smooth, color='blue', linewidth=2,label=f'МНК (Степінь m={optimal_m})')  # Малюємо нашу плавну синю лінію
plt.scatter(x_future, y_future, color='green', marker='*', s=200, label='Прогноз (Місяці 25-27)',zorder=6)  # Малюємо зелені зірочки-прогнози

plt.title("Апроксимація температур методом найменших квадратів (МНК)")
plt.xlabel("Місяць")
plt.ylabel("Температура (°C)")
plt.grid(True, linestyle='--')
plt.legend()
plt.show()