import numpy as np

def rosenbrock(x):
    return 100 * (x[0] ** 2 - x[1]) ** 2 + (x[0] - 1) ** 2
def system_target_function(x):
    f1 = x[0] ** 2 + x[1] ** 2 - 4
    f2 = x[0] - x[1]
    return f1 ** 2 + f2 ** 2
def hooke_jeeves(func, x0, delta, eps1=1e-6, eps2=1e-6, q=2.0, p=2.0):
    x_base = np.array(x0, dtype=float)
    delta = np.array(delta, dtype=float)
    trajectory = [x_base.copy()]
    steps = 0
    def explore(x_start, current_delta):
        x = x_start.copy()
        f_start = func(x)
        for i in range(len(x)):
            x[i] += current_delta[i]
            if func(x) < f_start:
                f_start = func(x)
            else:
                x[i] -= 2 * current_delta[i]
                if func(x) < f_start:
                    f_start = func(x)
                else:
                    x[i] += current_delta[i]
        return x

    while np.max(delta) > eps1:
        x_new = explore(x_base, delta)
        steps += 1
        if func(x_new) < func(x_base) and abs(func(x_new) - func(x_base)) > eps2:
            while True:
                x_p = x_base + p * (x_new - x_base)
                x_base = x_new.copy()
                trajectory.append(x_base.copy())
                x_new_explore = explore(x_p, delta)
                steps += 1
                if func(x_new_explore) < func(x_base):
                    x_new = x_new_explore.copy()
                else:
                    break
        else:
            delta = delta / q
    return x_base, steps, trajectory


if __name__ == "__main__":
    print("Метод Хука-Дживса \n")
    print("1. Тестування на функції Розенброка:")
    x0_rosen = [-1.2, 0.0]
    delta0 = [0.5, 0.5]
    res_rosen, steps_rosen, traj_rosen = hooke_jeeves(rosenbrock, x0_rosen, delta0)
    print(f"Початкова точка: {x0_rosen}")
    print(f"Знайдений мінімум: x1 = {res_rosen[0]:.6f}, x2 = {res_rosen[1]:.6f}")
    print(f"Значення функції: {rosenbrock(res_rosen):.6e}")
    print(f"Кількість ітерацій: {steps_rosen}\n")
    print("2. Розв'язок системи нелінійних рівнянь:")
    x0_sys = [1.0, 0.5]
    res_sys, steps_sys, traj_sys = hooke_jeeves(system_target_function, x0_sys, delta0)
    print(f"Початкове наближення: {x0_sys}")
    print(f"Знайдений розв'язок: x1 = {res_sys[0]:.6f}, x2 = {res_sys[1]:.6f}")
    print(f"Нев'язка цільової функції: {system_target_function(res_sys):.6e}")
    print(f"Кількість ітерацій: {steps_sys}")
    np.savetxt("trajectory_sys.txt", traj_sys, fmt='%.6f', header='x1 x2')
    print("\nТраєкторію спуску успішно збережено у файл 'trajectory_sys.txt'")