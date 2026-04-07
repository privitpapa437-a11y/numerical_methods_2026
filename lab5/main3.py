import numpy as np
import scipy.integrate as spi
f_evals = 0
def f(x):
    global f_evals
    f_evals += 1
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12) ** 2)


#  функція рахує інтеграл Сімпсона для одного маленького відрізка
def simpson_step(h, fa, fm, fb):
    return (h / 6) * (fa + 4 * fm + fb)
# Основна рекурсивна функція Адаптивного Сімпсона
def adaptive_simpson(a, b, eps, fa, fm, fb):
    m = (a + b) / 2
    h = b - a
    # Нові середні точки для лівої та правої половини
    m_left = (a + m) / 2
    m_right = (m + b) / 2

    fm_left = f(m_left)
    fm_right = f(m_right)

    # Рахуємо інтеграли
    I1 = simpson_step(h, fa, fm, fb)
    I2_left = simpson_step(h / 2, fa, fm_left, fm)
    I2_right = simpson_step(h / 2, fm, fm_right, fb)
    I2 = I2_left + I2_right

    # Умова збіжності  |I1 - I2| <= eps
    if np.abs(I1 - I2) <= eps:
        return I2
    else:
        # Якщо точності не вистачає, ділимо далі (eps теж ділимо для точності)
        left_int = adaptive_simpson(a, m, eps / 2, fa, fm_left, fm)
        right_int = adaptive_simpson(m, b, eps / 2, fm, fm_right, fb)
        return left_int + right_int

a, b = 0, 24

def f_exact(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12) ** 2)
I0, _ = spi.quad(f_exact, a, b)

print("Адаптивний алгоритм ")
print(f"Точне значення I0: {I0:.6f}\n")
print(f"{'Заданий епсилон (eps)':<25} -   {'Реальна похибка':<20} -   {'Кількість викликів f(x)'}")


# Тестуємо алгоритм з різними вимогами до точності
eps_values = [1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11]

for eps in eps_values:
    # Обнуляємо лічильник перед кожним тестом
    f_evals = 0

    fa = f(a)
    fb = f(b)
    fm = f((a + b) / 2)

    I_adapt = adaptive_simpson(a, b, eps, fa, fm, fb)
    actual_error = np.abs(I_adapt - I0)

    print(f"{eps:<25.0e}  {actual_error:<20.2e}  {f_evals}")