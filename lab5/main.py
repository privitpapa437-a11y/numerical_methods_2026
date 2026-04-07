import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi


def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12)**2)

a, b = 0, 24

I0, _ = spi.quad(f, a, b)
print(f"Пошук оптимального N ")
print(f"Точне значення інтегралу I0: {I0:.6f}")

# Складова формула Сімпсона
def simpson(N):
    if N % 2 != 0:
        N += 1  # Для методу Сімпсона N обов'язково має бути парним
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    # Формула Сімпсона
    I = (h / 3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])
    return I

#Дослідження залежності похибки від N (від 10 до 1000)
N_values = np.arange(10, 1002, 2)
errors = []

for N in N_values:
    err = np.abs(simpson(N) - I0)
    errors.append(err)

# Шукаємо Nopt, де похибка стає меншою або рівною 1e-12
target_eps = 1e-12
N_opt = None
eps_opt = None

for i, err in enumerate(errors):
    if err <= target_eps:
        N_opt = N_values[i]
        eps_opt = err
        break
print(f"Оптимальне розбиття N_opt (для точності 1e-12): {N_opt}")
if N_opt:
    print(f"Досягнута точність при N_opt: {eps_opt:.2e}")

plt.figure(figsize=(10, 6))
plt.plot(N_values, errors, color='blue', linewidth=2)
plt.yscale('log')
if N_opt:
    plt.scatter(N_opt, eps_opt, color='red', s=100, zorder=5, label=f'Оптимум N={N_opt}')
plt.title("Залежність похибки інтегрування Сімпсона від кількості розбиттів N")
plt.xlabel("Кількість розбиттів (N)")
plt.ylabel("Похибка |I(N) - I0|")
plt.grid(True, which="both", linestyle='--', alpha=0.7)
plt.legend()
plt.show()