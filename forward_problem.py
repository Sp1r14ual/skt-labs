"""
Модуль решения прямой задачи магниторазведки.
Реализует аналитическое решение для 2D ячеистой среды в дипольном приближении
"""

import numpy as np


class MagneticSurvey2D:
    def __init__(
        self,
        nx: int = 20,
        nz: int = 10,
        dx: float = 100.0,
        dz: float = 100.0,
        dy: float = 200.0,
        x_min: float = 0.0,
        z_top: float = 0.0,
        n_receivers: int = 40,
        receiver_z: float = 0.0,
        receiver_y: float = 500.0,
        component: str = "Bx",
        p0: tuple[float, float, float] = (1.0, 0.0, 0.0),
    ):
        """
        Инициализация параметров съемки и сетки расчетной области.

        :param nx: Количество ячеек по оси Ox
        :param nz: Количество ячеек по оси Oz
        :param dx: Шаг по оси Ox (метры)
        :param dz: Шаг по оси Oz (метры)
        :param dy: Эффективная толщина ячеек по оси Oy (метры)
        :param x_min: Начальная координата x расчетной области
        :param z_top: Верхняя отметка z расчетной области (дневная поверхность)
        :param n_receivers: Базовое количество приемников на профиле
        :param receiver_z: Высота/глубина расположения приемников
        :param receiver_y: Координата y профиля приемников
        :param component: Измеряемая компонента вектора индукции ('Bx' или 'Bz')
        :param p0: Вектор направления фонового намагничивания (p_x, p_y, p_z)
        """
        self.nx = nx
        self.nz = nz
        self.n_cells = nx * nz
        self.dx = dx
        self.dz = dz
        self.dy = dy
        self.cell_volume = dx * dy * dz
        self.x_min = x_min
        self.z_top = z_top

        self.n_receivers = n_receivers
        self.receiver_z = receiver_z
        self.receiver_y = receiver_y
        self.component = component
        self.p0 = np.array(p0, dtype=np.float64)

        # Вычисление центров ячеек
        # Ячейки нумеруются: i = iz + ix * nz (как в коде прошлой работы)
        self.cells = []
        for ix in range(self.nx):
            xc = self.x_min + (ix + 0.5) * self.dx
            for iz in range(self.nz):
                # z направлена вверх, в глубину z отрицательна
                zc = self.z_top - (iz + 0.5) * self.dz
                self.cells.append((xc, zc))
        self.cells = np.array(self.cells, dtype=np.float64)

        # Стандартный равномерный профиль приемников
        self.receiver_x = np.linspace(
            self.x_min, self.x_min + self.nx * self.dx, self.n_receivers
        )
        self.receivers = np.column_stack(
            [
                self.receiver_x,
                np.full(self.n_receivers, self.receiver_y),
                np.full(self.n_receivers, self.receiver_z),
            ]
        )

        # Построение матрицы прямого оператора L для базового профиля
        self.L = self.build_L(self.receivers)

    def cell_index(self, ix: int, iz: int) -> int:
        """Индекс ячейки в одномерном векторе параметров (iz + ix * nz)."""
        return iz + ix * self.nz

    def index_to_grid(self, idx: int) -> tuple[int, int]:
        """Преобразование одномерного индекса в (ix, iz)."""
        ix = idx // self.nz
        iz = idx % self.nz
        return ix, iz

    def compute_kernel(
        self, cell_coords: np.ndarray, rec_coords: np.ndarray
    ) -> tuple[float, float]:
        """
        Вычисление дипольного отклика ячейки в точке приема по формулам (7)-(9) пособия:
        r = sqrt(dx^2 + dy^2 + dz^2)
        B_x = mes(Omega) / (4*pi*r^3) * [ p_x*(3*x~^2/r^2 - 1) + p_y*(3*x~*y~/r^2) + p_z*(3*x~*z~/r^2) ]
        B_z = mes(Omega) / (4*pi*r^3) * [ p_x*(3*x~*z~/r^2) + p_y*(3*y~*z~/r^2) + p_z*(3*z~^2/r^2 - 1) ]
        """
        xc, zc = cell_coords
        xr, yr, zr = rec_coords

        dx = xr - xc
        dy = yr  # центр ячейки при y=0
        dz = zr - zc

        r2 = dx * dx + dy * dy + dz * dz + 1e-12
        r = np.sqrt(r2)
        r3 = r2 * r
        r5 = r2 * r3

        # Геометрический множитель (в СИ mu_0 / 4pi = 1e-7, в пособии используется нормированная шкала)
        # Масштаб объема ячейки
        c0 = self.cell_volume / (4.0 * np.pi * r3)

        px, py, pz = self.p0

        # Вклады от трех компонент намагниченности
        bx = c0 * (
            px * (3.0 * dx * dx / r2 - 1.0)
            + py * (3.0 * dx * dy / r2)
            + pz * (3.0 * dx * dz / r2)
        )

        bz = c0 * (
            px * (3.0 * dx * dz / r2)
            + py * (3.0 * dy * dz / r2)
            + pz * (3.0 * dz * dz / r2 - 1.0)
        )

        return bx, bz

    def build_L(
        self, receivers: np.ndarray, component: str = None
    ) -> np.ndarray:
        """
        Построение матрицы линейного прямого оператора L размера (n_rec, n_cells).
        g = L @ lambda
        """
        if component is None:
            component = self.component

        n_rec = len(receivers)
        L = np.zeros((n_rec, self.n_cells), dtype=np.float64)

        for i in range(n_rec):
            rec = receivers[i]
            for j in range(self.n_cells):
                cell = self.cells[j]
                bx, bz = self.compute_kernel(cell, rec)
                L[i, j] = bx if component == "Bx" else bz

        return L

    def forward_solve(
        self, lambda_vec: np.ndarray, L: np.ndarray = None
    ) -> np.ndarray:
        """
        Решение прямой задачи для заданного вектора намагниченности ячеек.
        :param lambda_vec: Вектор значений намагниченности ячеек длины n_cells
        :param L: Матрица прямого оператора (если None, используется стандартная self.L)
        :return: Вектор сигнала в приемниках
        """
        if L is None:
            L = self.L
        return L @ lambda_vec

    def add_noise(
        self,
        signal: np.ndarray,
        noise_level: float = 0.01,
        seed: int = None,
    ) -> np.ndarray:
        """
        Добавление относительного/абсолютного гауссова шума к сигналу.
        noise_level: доля от максимальной амплитуды сигнала (например, 0.01 = 1%)
        """
        if noise_level <= 0:
            return signal.copy()
        if seed is not None:
            np.random.seed(seed)

        amplitude = np.max(np.abs(signal))
        if amplitude < 1e-12:
            amplitude = 1.0
        noise = np.random.normal(0.0, noise_level * amplitude, size=signal.shape)
        return signal + noise
