"""
Модуль предобработки сигналов магниторазведки.
Реализует сглаживающий кубический сплайн с базисными функциями Эрмита.
Позволяет фильтровать шум и интерполировать данные с произвольного/неравномерного
профиля приемников на фиксированную сетку входов нейронной сети.
"""

import numpy as np
from scipy.interpolate import make_smoothing_spline


class HermiteSmoothingSpline:
    def __init__(
        self,
        n_elements: int = 20,
        alpha: float = 1e-4,
        weight: float = 1.0,
    ):
        """
        :param n_elements: Количество конечных элементов вдоль профиля (m в пособии)
        :param alpha: Параметр регуляризации гладкости сплайна (по первой производной)
        :param weight: Вес достоверности измерений omega_j
        """
        self.n_elements = n_elements
        self.alpha = alpha
        self.weight = weight
        self.q = None
        self.node_coords = None

    def _eval_basis(self, x: float, elem_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Вычисление локальных базисных функций Эрмита и их производных
        для точки x внутри элемента elem_idx = [x_i, x_{i+1}].
        Формулы (55) пособия:
          psi_1(xi) = 1 - 3*xi^2 + 2*xi^3
          psi_2(xi) = h_i * (xi - 2*xi^2 + xi^3)
          psi_3(xi) = 3*xi^2 - 2*xi^3
          psi_4(xi) = h_i * (-xi^2 + xi^3)
        """
        x_i = self.node_coords[elem_idx]
        x_next = self.node_coords[elem_idx + 1]
        h_i = x_next - x_i
        xi = (x - x_i) / h_i
        xi = np.clip(xi, 0.0, 1.0)

        # Значения функций
        psi = np.array(
            [
                1.0 - 3.0 * xi**2 + 2.0 * xi**3,
                h_i * (xi - 2.0 * xi**2 + xi**3),
                3.0 * xi**2 - 2.0 * xi**3,
                h_i * (-xi**2 + xi**3),
            ],
            dtype=np.float64,
        )

        # Производные d(psi)/dx = (d(psi)/dxi) * (1 / h_i)
        dpsi = np.array(
            [
                (-6.0 * xi + 6.0 * xi**2) / h_i,
                1.0 - 4.0 * xi + 3.0 * xi**2,
                (6.0 * xi - 6.0 * xi**2) / h_i,
                -2.0 * xi + 3.0 * xi**2,
            ],
            dtype=np.float64,
        )

        return psi, dpsi

    def fit(self, x_obs: np.ndarray, y_obs: np.ndarray):
        """
        Построение сплайна по наблюдениям (x_obs, y_obs).
        Формирование СЛАУ Aq = b.
        Размерность вектора q равна 2 * (m + 1), где m = n_elements.
        В каждом узле 2 степени свободы: значение функции и ее производная.
        """
        x_min, x_max = np.min(x_obs), np.max(x_obs)
        self.node_coords = np.linspace(x_min, x_max, self.n_elements + 1)
        m = self.n_elements
        n_dof = 2 * (m + 1)

        A = np.zeros((n_dof, n_dof), dtype=np.float64)
        b = np.zeros(n_dof, dtype=np.float64)

        # 1. Вклад наблюдений: sum_j omega_j * Psi(x_j) * Psi(x_j)^T
        for xj, yj in zip(x_obs, y_obs):
            # Определяем, в какой элемент попадает xj
            elem_idx = int(np.floor((xj - x_min) / (x_max - x_min + 1e-12) * m))
            elem_idx = min(max(elem_idx, 0), m - 1)

            psi, _ = self._eval_basis(xj, elem_idx)
            # Индексы глобальных степеней свободы для элемента:
            # узел i: 2*elem_idx, 2*elem_idx + 1; узел i+1: 2*elem_idx + 2, 2*elem_idx + 3
            dof_indices = [
                2 * elem_idx,
                2 * elem_idx + 1,
                2 * elem_idx + 2,
                2 * elem_idx + 3,
            ]

            for ii in range(4):
                gi = dof_indices[ii]
                b[gi] += self.weight * psi[ii] * yj
                for jj in range(4):
                    gj = dof_indices[jj]
                    A[gi, gj] += self.weight * psi[ii] * psi[jj]

        # 2. Вклад регуляризации производной: int alpha * (dPsi/dx)(dPsi/dx)^T dx
        # Аналитическое интегрирование матрицы жесткости на каждом элементе методом Гаусса (3 точки)
        gauss_xi = np.array([0.1127016654, 0.5, 0.8872983346])
        gauss_w = np.array([5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0])

        for e in range(m):
            h_i = self.node_coords[e + 1] - self.node_coords[e]
            dof_indices = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]

            Ke = np.zeros((4, 4), dtype=np.float64)
            for g_xi, gw in zip(gauss_xi, gauss_w):
                # dpsi/dx в локальных координатах
                dpsi = np.array(
                    [
                        (-6.0 * g_xi + 6.0 * g_xi**2) / h_i,
                        1.0 - 4.0 * g_xi + 3.0 * g_xi**2,
                        (6.0 * g_xi - 6.0 * g_xi**2) / h_i,
                        -2.0 * g_xi + 3.0 * g_xi**2,
                    ]
                )
                Ke += self.alpha * np.outer(dpsi, dpsi) * (gw * h_i)

            for ii in range(4):
                for jj in range(4):
                    A[dof_indices[ii], dof_indices[jj]] += Ke[ii, jj]

        # Добавим малую диагональную добавку для обусловленности
        A += 1e-10 * np.eye(n_dof)

        self.q = np.linalg.solve(A, b)
        return self

    def predict(self, x_eval: np.ndarray) -> np.ndarray:
        """
        Вычисление значений сплайна в точках x_eval.
        """
        if self.q is None:
            raise ValueError("Сплайн еще не обучен (вызовите fit)")

        m = self.n_elements
        x_min = self.node_coords[0]
        x_max = self.node_coords[-1]
        y_eval = np.zeros_like(x_eval, dtype=np.float64)

        for idx, x in enumerate(x_eval):
            elem_idx = int(np.floor((x - x_min) / (x_max - x_min + 1e-12) * m))
            elem_idx = min(max(elem_idx, 0), m - 1)

            psi, _ = self._eval_basis(x, elem_idx)
            dof_indices = [
                2 * elem_idx,
                2 * elem_idx + 1,
                2 * elem_idx + 2,
                2 * elem_idx + 3,
            ]
            y_eval[idx] = np.sum(psi * self.q[dof_indices])

        return y_eval


class SignalPreprocessor:
    """
    Препроцессор входных геофизических сигналов.
    Выполняет:
    1. Интерполяцию произвольного профиля наблюдений на фиксированные входные узлы нейросети.
    2. Фильтрацию высокочастотного шума и случайных выбросов с помощью сглаживающего сплайна.
    3. Нормализацию данных.
    """

    def __init__(
        self,
        target_x: np.ndarray,
        use_hermite: bool = True,
        smoothing_factor: float = 1e-3,
        n_elements: int = 20,
    ):
        """
        :param target_x: Фиксированные координаты приемников для подачи в нейросеть
        :param use_hermite: Использовать ли эрмитов сплайн из пособия (True) или адаптивный сплайн SciPy
        :param smoothing_factor: Коэффициент регуляризации/сглаживания
        :param n_elements: Количество элементов для сплайна Эрмита
        """
        self.target_x = np.array(target_x, dtype=np.float64)
        self.n_target = len(self.target_x)
        self.use_hermite = use_hermite
        self.smoothing_factor = smoothing_factor
        self.n_elements = n_elements

    def process(self, x_obs: np.ndarray, signal_obs: np.ndarray) -> np.ndarray:
        """
        Предобработка сигнала: сглаживание и интерполяция в целевые точки target_x.
        """
        # Если наблюдений достаточно мало или требуется SciPy сплайн
        if not self.use_hermite or len(x_obs) < self.n_elements + 2:
            try:
                # make_smoothing_spline из scipy (lam подбирается от smoothing_factor)
                spl = make_smoothing_spline(
                    x_obs, signal_obs, lam=self.smoothing_factor
                )
                return spl(self.target_x)
            except Exception:
                # Линейная интерполяция как безопасный fallback
                return np.interp(self.target_x, x_obs, signal_obs)

        # Полная реализация по пособию НГТУ
        try:
            spline = HermiteSmoothingSpline(
                n_elements=self.n_elements,
                alpha=self.smoothing_factor,
                weight=1.0,
            )
            spline.fit(x_obs, signal_obs)
            return spline.predict(self.target_x)
        except Exception:
            return np.interp(self.target_x, x_obs, signal_obs)
