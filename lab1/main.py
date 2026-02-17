import requests
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Отримання даних (GPS координати маршруту на Говерлу) ---
# Список координат з методички (API іноді глючить, тому надійніше задати їх списком,
# але спробуємо спочатку через API, як просить методичка)

url = "https://api.open-elevation.com/api/v1/lookup"
locations = (
    "48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|"
    "48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|"
    "48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|"
    "48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|"
    "48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|"
    "48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|"
    "48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"
)

print("Завантаження даних...")
try:
    response = requests.get(f"{url}?locations={locations}")
    data = response.json()
    results = data["results"]
except Exception as e:
    print(f"Помилка API: {e}. Використовую резервні дані.")
    # Тут можна було б додати хардкод координат, якщо API не відповість
    results = []

if not results:
    print("Не вдалося отримати дані.")
    exit()

# Виводимо таблицю отриманих точок (Пункт 3 методички)
print(f"\nКількість вузлів: {len(results)}")
print(f"{'No':<3} | {'Latitude':<10} | {'Longitude':<10} | {'Elevation':<10}")
for i, point in enumerate(results):
    print(f"{i:<3} | {point['latitude']:.6f} | {point['longitude']:.6f} | {point['elevation']:.2f}")


# --- 2. Розрахунок відстані (Haversine Formula) ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Радіус Землі в метрах
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# Підготовка масивів X (відстань) та Y (висота)
coords = [(p['latitude'], p['longitude']) for p in results]
elevations = [p['elevation'] for p in results]  # Це наші Y
distances = [0.0]  # Це наші X (перша точка - 0 метрів)

for i in range(1, len(coords)):
    d = haversine(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1])
    distances.append(distances[-1] + d)  # Накопичуємо відстань (кумулятивна)

# Виводимо таблицю відстаней (Пункт 4 методички)
print("\nТабуляція (відстань, висота):")
print(f"{'Distance (m)':<15} | {'Elevation (m)':<15}")
for i in range(len(distances)):
    print(f"{distances[i]:<15.2f} | {elevations[i]:<15.2f}")

# --- 3. Попередній графік (точки) ---
plt.figure(figsize=(10, 6))
plt.scatter(distances, elevations, color='red', label='GPS Points')
plt.plot(distances, elevations, '--', alpha=0.5, label='Linear connect')  # Просто з'єднати лінією для наочності
plt.xlabel("Відстань (м)")
plt.ylabel("Висота (м)")
plt.title("Профіль висоти (Дискретні точки)")
plt.legend()
plt.grid(True)
plt.show()


# --- 4. Реалізація Кубічного Сплайна (Математична частина) ---
class CubicSpline:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.n = len(x) - 1
        self.h = np.diff(self.x)

        # Обчислюємо коефіцієнти (згідно методички)
        self.c = self.calculate_c_matrix()
        self.a = self.y[:-1]  # a_i = y_i
        self.b = self.calculate_b()
        self.d = self.calculate_d()

    def calculate_c_matrix(self):
        # Метод прогонки (Thomas algorithm) для системи лінійних рівнянь
        n = self.n
        h = self.h

        # Матриця A (трьохдіагональна) та вектор F (права частина)
        alpha = np.zeros(n)
        beta = np.zeros(n)

        # Прямий хід прогонки
        # c0 = 0 (гранична умова вільного сплайна)
        alpha[0] = 0
        beta[0] = 0

        for i in range(1, n):
            # Коефіцієнти рівняння (зі стор. 2-3 методички)
            A_i = h[i - 1]
            C_i = 2 * (h[i - 1] + h[i])
            B_i = h[i]
            F_i = 3 * ((self.y[i + 1] - self.y[i]) / h[i] - (self.y[i] - self.y[i - 1]) / h[i - 1])

            z = A_i * alpha[i - 1] + C_i
            alpha[i] = -B_i / z
            beta[i] = (F_i - A_i * beta[i - 1]) / z

        # Зворотний хід прогонки
        c = np.zeros(n + 1)
        c[n] = 0  # Гранична умова

        for i in range(n - 1, 0, -1):
            c[i] = alpha[i] * c[i + 1] + beta[i]

        return c

    def calculate_b(self):
        b = np.zeros(self.n)
        for i in range(self.n):
            b[i] = (self.y[i + 1] - self.y[i]) / self.h[i] - self.h[i] * (self.c[i + 1] + 2 * self.c[i]) / 3
        return b

    def calculate_d(self):
        d = np.zeros(self.n)
        for i in range(self.n):
            d[i] = (self.c[i + 1] - self.c[i]) / (3 * self.h[i])
        return d

    def interpolate(self, x_val):
        # Знаходимо потрібний інтервал
        if x_val < self.x[0] or x_val > self.x[-1]:
            return None  # За межами діапазону

        # Знаходимо i таке, що x_i <= x_val < x_{i+1}
        i = np.searchsorted(self.x, x_val, side='right') - 1
        i = np.clip(i, 0, self.n - 1)

        dx = x_val - self.x[i]
        # Формула сплайна: S(x) = a + b*dx + c*dx^2 + d*dx^3
        return self.a[i] + self.b[i] * dx + self.c[i] * (dx ** 2) + self.d[i] * (dx ** 3)


# --- 5. Використання сплайна та побудова фінального графіку ---

# Створюємо сплайн на основі наших даних
spline = CubicSpline(distances, elevations)

# Генеруємо багато точок для гладкої лінії (наприклад, кожні 10 метрів)
x_smooth = np.linspace(distances[0], distances[-1], 500)
y_smooth = [spline.interpolate(xi) for xi in x_smooth]

# Малюємо графік
plt.figure(figsize=(12, 7))
plt.scatter(distances, elevations, color='red', label='GPS Вузли (Points)', zorder=5)
plt.plot(distances, elevations, '--', color='gray', alpha=0.5, label='Лінійне з\'єднання')
plt.plot(x_smooth, y_smooth, '-', color='blue', linewidth=2, label='Кубічний сплайн (Smooth)')

plt.xlabel("Відстань (м)")
plt.ylabel("Висота (м)")
plt.title("Профіль висоти маршруту на Говерлу (Інтерполяція)")
plt.legend()
plt.grid(True)
plt.show()

# Вивід коефіцієнтів (Пункт 8-9 методички)
print("\nКоефіцієнти сплайнів (перші 5):")
print(f"{'i':<3} | {'a':<10} | {'b':<10} | {'c':<10} | {'d':<10}")
for i in range(min(5, spline.n)):
    print(f"{i:<3} | {spline.a[i]:<10.2f} | {spline.b[i]:<10.2f} | {spline.c[i]:<10.4f} | {spline.d[i]:<10.6f}")