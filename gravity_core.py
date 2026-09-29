import numpy as np


class GravitySolver:
    def __init__(self, nx=40, nz=20, dx=1.0, dz=1.0, n_receivers=100, receiver_z=1.0):
        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dz = dz
        self.n_cells = nx * nz
        self.n_receivers = n_receivers
        self.receiver_z = receiver_z

        # 1. Построение координат ячеек (как в отчете: x=ix*dx, z=-iz*dz)
        self.cells = []
        for ix in range(nx):
            for iz in range(nz):
                x = ix * dx
                z = -iz * dz
                self.cells.append((x, z))
        self.cells = np.array(self.cells)

        # 2. Построение профиля приемников (z = receiver_z)
        self.receivers = np.column_stack([
            np.linspace(0, nx * dx, n_receivers),
            np.full(n_receivers, receiver_z)
        ])

        # 3. Вычисление матрицы прямого оператора L (векторизованно)
        self.L = self._build_L()

        # 4. Список соседей и матрица регуляризации C
        self.neighbors = self._get_neighbors()

    def _build_L(self):
        # rx: (n_receivers, 1), cx: (1, n_cells)
        dx = self.receivers[:, 0:1] - self.cells[:, 0:1].T
        dz = self.receivers[:, 1:2] - self.cells[:, 1:2].T
        r = np.sqrt(dx**2 + dz**2) + 1e-8
        return dz / (r**3)

    def _get_neighbors(self):
        neighbors = [[] for _ in range(self.n_cells)]

        def index(ix, iz):
            return iz + ix * self.nz

        for ix in range(self.nx):
            for iz in range(self.nz):
                i = index(ix, iz)
                if ix > 0:
                    neighbors[i].append(index(ix - 1, iz))
                if ix < self.nx - 1:
                    neighbors[i].append(index(ix + 1, iz))
                if iz > 0:
                    neighbors[i].append(index(ix, iz - 1))
                if iz < self.nz - 1:
                    neighbors[i].append(index(ix, iz + 1))
        return neighbors

    def build_C(self, gamma):
        C = np.zeros((self.n_cells, self.n_cells))
        for i in range(self.n_cells):
            for j in self.neighbors[i]:
                C[i, i] += gamma
                C[i, j] -= gamma
        return C

    def forward(self, rho):
        """Прямая задача: g = L @ rho"""
        return self.L @ rho

    def solve_classical_inverse(self, g_obs, gamma=1e-1):
        """Решение обратной задачи классическим методом с гамма-регуляризацией"""
        C = self.build_C(gamma)
        A = self.L.T @ self.L
        b = self.L.T @ g_obs
        return np.linalg.solve(A + C, b)

    def get_standard_test_model(self):
        """1. Стандартная аномалия"""
        rho = np.zeros(self.n_cells)
        for ix in range(8, 12):
            for iz in range(3, 6):
                idx = iz + ix * self.nz
                rho[idx] = 1.0
        return rho

    def get_medium_anomaly_model(self):
        """2. Аномалия среднего размера"""
        rho = np.zeros((self.nx, self.nz))
        # Крупное тело в центре сетки
        rho[14:26, 6:14] = 1.0
        return rho.flatten()

    def get_inclined_anomaly_model(self):
        """3. Наклонная аномалия / дайка"""
        rho = np.zeros((self.nx, self.nz))
        # Наклонный пласт от малой глубины к большой
        for i in range(12):
            x_idx = 10 + i
            z_top = 2 + i
            z_bot = min(self.nz, z_top + 4)
            if x_idx < self.nx and z_top < self.nz:
                rho[x_idx, z_top:z_bot] = 1.0
        return rho.flatten()

    def get_complex_shape_model(self):
        """4. Аномалия сложной формы / пласт с разломом"""
        rho = np.zeros((self.nx, self.nz))
        # Горизонтальное тело
        rho[8:28, 9:14] = 0.9
        # Вертикальный подъем / выступ (ступень)
        rho[22:28, 3:10] = 1.2
        rho[13:17, 7:10] = 0.8
        return rho.flatten()

    def get_two_lateral_objects_model(self):
        """5. Два объекта, разнесенные по латерали"""
        rho = np.zeros((self.nx, self.nz))
        # Левый объект
        rho[6:12, 6:12] = 1.0
        # Правый объект
        rho[28:34, 6:12] = 1.0
        return rho.flatten()

    def get_overlapping_objects_model(self):
        """6. Перекрывающие объекты: экранирование"""
        rho = np.zeros((self.nx, self.nz))
        # Верхний тонкий приповерхностный пласт
        rho[15:25, 2:4] = 1.0
        # Нижнее массивное глубокое тело
        rho[11:29, 12:16] = 1.0
        return rho.flatten()

    def get_deep_object_model(self):
        """7. Глубокозалегающий объект"""
        rho = np.zeros((self.nx, self.nz))
        rho[15:25, 14:18] = 1.2
        return rho.flatten()

    def get_fault_step_model(self):
        """8. Ступенчатый сброс"""
        rho = np.zeros((self.nx, self.nz))
        # Поднятое левое крыло
        rho[5:20, 4:7] = 1.0
        # Опущенное правое крыло
        rho[20:35, 10:13] = 1.0
        return rho.flatten()

    def generate_random_model(self, max_anomalies=3):
        """Генерация случайной геологической модели (1-3 аномалии)"""
        rho = np.zeros((self.nx, self.nz))
        num_bodies = np.random.randint(1, max_anomalies + 1)

        for _ in range(num_bodies):
            # Ширина и глубина аномалии
            w = np.random.randint(2, 9)
            h = np.random.randint(2, 6)
            
            # Позиция
            x0 = np.random.randint(1, max(2, self.nx - w - 1))
            z0 = np.random.randint(1, max(2, self.nz - h - 1))
            
            # Плотность (от 0.5 до 2.0)
            density = np.random.uniform(0.6, 1.5)

            # Форма: прямоугольная или слегка сглаженная/ступенчатая
            shape_type = np.random.choice(['rect', 'trapezoid', 'inclined'])
            if shape_type == 'rect':
                rho[x0:x0+w, z0:z0+h] = np.maximum(rho[x0:x0+w, z0:z0+h], density)
            elif shape_type == 'trapezoid':
                for step, dx_i in enumerate(range(w)):
                    top_z = max(0, z0 - (step % 2))
                    rho[x0+dx_i, top_z:z0+h] = np.maximum(rho[x0+dx_i, top_z:z0+h], density)
            else:
                # Наклонный пласт
                for dx_i in range(w):
                    shift = int(dx_i * 0.5)
                    z_start = min(self.nz - 1, z0 + shift)
                    z_end = min(self.nz, z_start + h)
                    rho[x0+dx_i, z_start:z_end] = np.maximum(rho[x0+dx_i, z_start:z_end], density)

        return rho.flatten()

    def generate_dataset(self, n_samples=2500, noise_std=0.01):
        """
        Генерация датасета пар (g_obs, rho) согласно заданию.
        Рекомендуемый объем: >= 2000 моделей.
        """
        X = np.zeros((n_samples, self.n_receivers), dtype=np.float32)
        Y = np.zeros((n_samples, self.n_cells), dtype=np.float32)

        for i in range(n_samples):
            rho = self.generate_random_model()
            g_true = self.forward(rho)
            noise = noise_std * np.random.randn(self.n_receivers)
            g_obs = g_true + noise

            X[i] = g_obs
            Y[i] = rho

        return X, Y
