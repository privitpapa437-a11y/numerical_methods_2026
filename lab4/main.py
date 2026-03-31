import numpy as np
import matplotlib.pyplot as plt
# Функція вологості ґрунту
def M(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)
# Точна аналітична похідна
def exact_dM(t):
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)
# Формула чисельного диференціювання
def num_dM(t, h):
    return (M(t + h) - M(t - h)) / (2 * h)
t0 = 1.0
exact_val = exact_dM(t0)

print("ПОШУК ОПТИМАЛЬНОГО КРОКУ")
print(f"Точне значення похідної: {exact_val:.6f}")

h_values = np.logspace(-20, 3, 1000)
errors = np.abs(num_dM(t0, h_values) - exact_val)

best_idx = np.argmin(errors)
h_opt = h_values[best_idx]
err_opt = errors[best_idx]

print(f"Оптимальний крок h0: {h_opt:.2e}")
print(f"Мінімальна похибка R0: {err_opt:.2e}\n")

print(" МЕТОДИ ПОКРАЩЕННЯ ТОЧНОСТІ ")
h = 1e-3

D_h = num_dM(t0, h)
D_2h = num_dM(t0, 2 * h)
D_4h = num_dM(t0, 4 * h)

R1 = np.abs(D_h - exact_val)
print("1. Звичайне чисельне диференціювання (h = 0.001):")
print(f"   Значення: {D_h:.6f}, Похибка R1: {R1:.2e}\n")

# Метод Рунге-Ромберга
D_R = D_h + (D_h - D_2h) / 3
R2 = np.abs(D_R - exact_val)
print("2. Метод Рунге-Ромберга:")
print(f"   Уточнене значення: {D_R:.6f}")
print(f"   Похибка R2: {R2:.2e}")
print(f"   (Похибка зменшилась у {R1/R2:.1f} разів)\n")

# Метод Ейткена
D_E = (D_2h**2 - D_4h * D_h) / (2 * D_2h - (D_4h + D_h))
p = (1 / np.log(2)) * np.log(np.abs((D_4h - D_2h) / (D_2h - D_h)))
R3 = np.abs(D_E - exact_val)

print("3. Метод Ейткена:")
print(f"   Уточнене значення: {D_E:.6f}")
print(f"   Похибка R3: {R3:.2e}")
print(f"   Оцінка порядку точності p: {p:.2f}")

plt.figure(figsize=(10, 6))
plt.loglog(h_values, errors, color='blue', linewidth=2)
plt.scatter(h_opt, err_opt, color='red', s=100, zorder=5, label=f'Оптимум: h={h_opt:.1e}')
plt.title("Залежність похибки чисельного диференціювання від кроку h")
plt.xlabel("Крок h")
plt.ylabel("Похибка R")
plt.grid(True, which="both", linestyle='--', alpha=0.7)
plt.legend()
plt.show()