import numpy as np
import matplotlib.pyplot as plt


# 1. ПАРАМЕТРЫ СЕТКИ
nx, nz = 40, 20   # количество ячеек
dx, dz = 1.0, 1.0

cells = []
for ix in range(nx):
    for iz in range(nz):
        x = ix * dx
        z = -iz * dz  # вниз
        cells.append((x, z))

n_cells = len(cells)


# 2. ПРИЕМНИКИ (профиль)
n_receivers = 100
receivers = [(x, 1.0) for x in np.linspace(0, nx*dx, n_receivers)]


# 3. ПРЯМАЯ ЗАДАЧА
def compute_g(cell, receiver):
    x, z = cell
    xr, zr = receiver

    dx = xr - x
    dz = zr - z

    r = np.sqrt(dx**2 + dz**2) + 1e-8

    return dz / (r**3)  # вертикальная компонента


# 4. МАТРИЦА L
def build_L(cells, receivers):
    L = np.zeros((len(receivers), len(cells)))

    for i, r in enumerate(receivers):
        for j, c in enumerate(cells):
            L[i, j] = compute_g(c, r)

    return L


# 5. СОСЕДИ (для γ)
def get_neighbors(nx, nz):
    neighbors = [[] for _ in range(nx * nz)]

    def index(ix, iz):
        return iz + ix * nz

    for ix in range(nx):
        for iz in range(nz):
            i = index(ix, iz)

            if ix > 0:
                neighbors[i].append(index(ix - 1, iz))
            if ix < nx - 1:
                neighbors[i].append(index(ix + 1, iz))
            if iz > 0:
                neighbors[i].append(index(ix, iz - 1))
            if iz < nz - 1:
                neighbors[i].append(index(ix, iz + 1))

    return neighbors


# 6. МАТРИЦА РЕГУЛЯРИЗАЦИИ
def build_C(n_cells, neighbors, gamma):
    C = np.zeros((n_cells, n_cells))

    for i in range(n_cells):
        for j in neighbors[i]:
            C[i, i] += gamma
            C[i, j] -= gamma

    return C


# 7. ОБРАТНАЯ ЗАДАЧА
def solve_inverse(L, g, C):
    A = L.T @ L
    b = L.T @ g

    return np.linalg.solve(A + C, b)


# 8. СОЗДАНИЕ ИСТИННОЙ МОДЕЛИ
true_rho = np.zeros(n_cells)

# задаем аномалию (прямоугольник)
for ix in range(8, 12):
    for iz in range(3, 6):
        idx = iz + ix * nz
        true_rho[idx] = 1.0


# 9. РАСЧЕТ ДАННЫХ
L = build_L(cells, receivers)
g_true = L @ true_rho

# добавим небольшой шум
noise = 0.01 * np.random.randn(len(g_true))
g_obs = g_true + noise
# g_obs = g_true  # без шума для наглядности


# 10. РЕГУЛЯРИЗАЦИЯ γ
neighbors = get_neighbors(nx, nz)
gamma = 1e-1

C = build_C(n_cells, neighbors, gamma)


# 11. РЕШЕНИЕ
rho_est = solve_inverse(L, g_obs, C)


# 12. РАСЧЕТ МОДЕЛЬНОГО СИГНАЛА
g_calc = L @ rho_est

# 13. НЕВЯЗКА
misfit = np.linalg.norm(g_obs - g_calc)
print("Невязка:", misfit)


# 14. ВИЗУАЛИЗАЦИЯ
# --- поле ---
plt.figure(figsize=(10, 4))
plt.plot(g_obs, label="Наблюденные")
plt.plot(g_calc, '--', label="Рассчитанные")
plt.legend()
plt.title("Сигналы вдоль профиля")
plt.xlabel("Приемник")
plt.ylabel("Δg")
plt.grid()

# --- невязка ---
plt.figure(figsize=(10, 4))
plt.plot(g_obs - g_calc)
plt.title("Невязка")
plt.grid()

# --- распределение плотности ---
def plot_model(rho, title):
    model = rho.reshape((nx, nz)).T

    plt.figure(figsize=(6, 4))
    plt.imshow(model, origin='lower', aspect='auto')
    plt.colorbar(label="Плотность")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Z")

plot_model(true_rho, "Истинная модель")
plot_model(rho_est, "Восстановленная модель")

plt.show()