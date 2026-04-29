import numpy as np

def F(x):
    return x * np.sin(x) - 1
def dF(x):
    return np.sin(x) + x * np.cos(x)
def ddF(x):
    return 2 * np.cos(x) - x * np.sin(x)
def check_stop(x_new, x_old, eps):
    return abs(F(x_new)) < eps and abs(x_new - x_old) < eps
def simple_iteration(x0, eps, tau):
    x = x0
    for i in range(1, 1000):
        x_new = x + tau * F(x)
        if check_stop(x_new, x, eps):
            return x_new, i
        x = x_new
    return x, "-"
def newton(x0, eps):
    x = x0
    for i in range(1, 1000):
        x_new = x - F(x) / dF(x)
        if check_stop(x_new, x, eps):
            return x_new, i
        x = x_new
    return x, "-"
def chebyshev(x0, eps):
    x = x0
    for i in range(1, 1000):
        fx = F(x)
        dfx = dF(x)
        ddfx = ddF(x)
        x_new = x - fx / dfx - 0.5 * (fx ** 2 * ddfx) / (dfx ** 3)
        if check_stop(x_new, x, eps):
            return x_new, i
        x = x_new
    return x, "-"
def secant(x0, x1, eps):
    xm1, x = x0, x1
    for i in range(1, 1000):
        fx = F(x)
        fxm1 = F(xm1)
        x_new = x - fx * (x - xm1) / (fx - fxm1)
        if check_stop(x_new, x, eps):
            return x_new, i
        xm1, x = x, x_new
    return x, "-"
def parabola(x0, x1, x2, eps):
    xm2, xm1, x = x0, x1, x2
    for i in range(1, 1000):
        fx = F(x)
        f_x_xm1 = (F(x) - F(xm1)) / (x - xm1)
        f_xm1_xm2 = (F(xm1) - F(xm2)) / (xm1 - xm2)
        f_x_xm1_xm2 = (f_x_xm1 - f_xm1_xm2) / (x - xm2)
        A = f_x_xm1_xm2
        B = f_x_xm1 + (x - xm1) * f_x_xm1_xm2
        C = fx
        D = B ** 2 - 4 * A * C
        sign_B = 1 if B >= 0 else -1
        dx = -2 * C / (B + sign_B * np.sqrt(D))
        x_new = x + dx

        if check_stop(x_new, x, eps):
            return x_new, i
        xm2, xm1, x = xm1, x, x_new
    return x, "-"
def inv_interp(x0, x1, x2, eps):
    xm2, xm1, x = x0, x1, x2
    for i in range(1, 1000):
        y0, y1, y2 = F(xm2), F(xm1), F(x)
        t1 = xm2 * (y1 * y2) / ((y0 - y1) * (y0 - y2))
        t2 = xm1 * (y0 * y2) / ((y1 - y0) * (y1 - y2))
        t3 = x * (y0 * y1) / ((y2 - y0) * (y2 - y1))
        x_new = t1 + t2 + t3
        if check_stop(x_new, x, eps):
            return x_new, i
        xm2, xm1, x = xm1, x, x_new
    return x, "-"
if __name__ == "__main__":
    eps = 1e-10
    x_tab = np.arange(0, 4.1, 0.1)
    y_tab = F(x_tab)
    np.savetxt("tabulation.txt", np.column_stack((x_tab, y_tab)), fmt='%.4f')
    print("1. Табуляцію функції збережено у файл 'tabulation.txt'")
    print(f"\nТочність: eps = {eps}")
    print(f"{'Метод':<25} | {'Корінь 1 (зростання)':<20} | {'Корінь 2 (спадання)':<20}")
    r1_simp, i1_simp = simple_iteration(1.0, eps, tau=-0.5)
    r2_simp, i2_simp = simple_iteration(3.0, eps, tau=0.5)
    print(f"{'Простої ітерації':<25} | {r1_simp:.10f} ({i1_simp} іт.) | {r2_simp:.10f} ({i2_simp} іт.)")
    r1_newt, i1_newt = newton(1.0, eps)
    r2_newt, i2_newt = newton(3.0, eps)
    print(f"{'Ньютона':<25} | {r1_newt:.10f} ({i1_newt} іт.)  | {r2_newt:.10f} ({i2_newt} іт.)")
    r1_cheb, i1_cheb = chebyshev(1.0, eps)
    r2_cheb, i2_cheb = chebyshev(3.0, eps)
    print(f"{'Чебишева':<25} | {r1_cheb:.10f} ({i1_cheb} іт.)   | {r2_cheb:.10f} ({i2_cheb} іт.)")
    r1_sec, i1_sec = secant(0.9, 1.0, eps)
    r2_sec, i2_sec = secant(2.9, 3.0, eps)
    print(f"{'Хорд':<25} | {r1_sec:.10f} ({i1_sec} іт.)  | {r2_sec:.10f} ({i2_sec} іт.)")
    r1_par, i1_par = parabola(0.9, 1.0, 1.1, eps)
    r2_par, i2_par = parabola(2.8, 2.9, 3.0, eps)
    print(f"{'Парабол':<25} | {r1_par:.10f} ({i1_par} іт.)   | {r2_par:.10f} ({i2_par} іт.)")
    r1_inv, i1_inv = inv_interp(0.9, 1.0, 1.1, eps)
    r2_inv, i2_inv = inv_interp(2.8, 2.9, 3.0, eps)
    print(f"{'Звор. інтерполяції':<25} | {r1_inv:.10f} ({i1_inv} іт.)   | {r2_inv:.10f} ({i2_inv} іт.)")