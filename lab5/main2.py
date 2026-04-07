import numpy as np
import scipy.integrate as spi

def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12)**2)

a, b = 0, 24
I0, _ = spi.quad(f, a, b)

def simpson(N):
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    return (h / 3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])

#Вибір базового розбиття N0
# N0 = N_opt / 10 (кратне 8) беремо N0 = 8.
N0 = 8

# Рахуємо інтеграли для трьох різних розбиттів (N0, N0/2, N0/4)
I_N0   = simpson(N0)       # N = 8
I_N0_2 = simpson(N0 // 2)  # N = 4
I_N0_4 = simpson(N0 // 4)  # N = 2

eps0 = np.abs(I_N0 - I0)

print(f"методи Рунге-Ромберга та Ейткена")
print(f"Точне значення I0: {I0:.6f}")
print(f"Базове розбиття N0 = {N0}")
print(f"Значення за Сімпсоном I(N0): {I_N0:.6f}, Похибка eps0: {eps0:.2e}\n")

#Метод Рунге-Ромберга

I_R = I_N0 + (I_N0 - I_N0_2) / 15
epsR = np.abs(I_R - I0)

print(" Метод Рунге-Ромберга:")
print(f"   Уточнене значення I_R: {I_R:.6f}")
print(f"   Похибка epsR: {epsR:.2e}")
print(f"   (Похибка зменшилась у {eps0/epsR:.1f} разів)\n")

# Метод Ейткена
num = I_N0_2**2 - I_N0 * I_N0_4
den = 2 * I_N0_2 - (I_N0 + I_N0_4)
I_E = num / den
epsE = np.abs(I_E - I0)

# Оцінка порядку точності
p = (1 / np.log(2)) * np.log(np.abs((I_N0_4 - I_N0_2) / (I_N0_2 - I_N0)))

print("Метод Ейткена:")
print(f"Уточнене значення I_E: {I_E:.6f}")
print(f"Похибка epsE: {epsE:.2e}")
print(f"Порядок методу p: {p:.2f}")
print(f"(Похибка зменшилась у {eps0/epsE:.1f} разів)")