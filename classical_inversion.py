"""
Модуль классического решения обратной задачи (в соответствии с первой лабораторной работой).
Реализует регуляризацию Тихонова и сглаживающую регуляризацию по градиенту (gamma)
на 4-связной сетке ячеек.
"""

import numpy as np
from forward_problem import MagneticSurvey2D


class ClassicalInversion:
    def __init__(self, survey: MagneticSurvey2D):
        self.survey = survey
        self.nx = survey.nx
        self.nz = survey.nz
        self.n_cells = survey.n_cells
        self.neighbors = self._build_neighbors()

    def _build_neighbors(self) -> list[list[int]]:
        """
        Построение списка соседних ячеек
        """
        neighbors = [[] for _ in range(self.n_cells)]

        for ix in range(self.nx):
            for iz in range(self.nz):
                i = self.survey.cell_index(ix, iz)

                if ix > 0:
                    neighbors[i].append(self.survey.cell_index(ix - 1, iz))
                if ix < self.nx - 1:
                    neighbors[i].append(self.survey.cell_index(ix + 1, iz))
                if iz > 0:
                    neighbors[i].append(self.survey.cell_index(ix, iz - 1))
                if iz < self.nz - 1:
                    neighbors[i].append(self.survey.cell_index(ix, iz + 1))

        return neighbors

    def build_C(self, gamma: float = 0.1) -> np.ndarray:
        """
        Построение матрицы регуляризации C по формулам:
        C_{ij} = -(gamma_i + gamma_j) для соседей,
        C_{ii} = sum_m (gamma_i + gamma_m).
        """
        C = np.zeros((self.n_cells, self.n_cells), dtype=np.float64)

        for i in range(self.n_cells):
            for j in self.neighbors[i]:
                C[i, i] += gamma
                C[i, j] -= gamma

        return C

    def solve(
        self,
        g_obs: np.ndarray,
        gamma: float = 1e-2,
        alpha: float = 1e-4,
        auto_scale: bool = True,
        non_negative: bool = True,
    ) -> np.ndarray:
        """
        Решение СЛАУ (L^T L + C + alpha * I) lambda = L^T g_obs.
        :param g_obs: Наблюденный сигнал в приемниках
        :param gamma: Коэффициент регуляризации сглаживания по соседям
        :param alpha: Коэффициент регуляризации Тихонова (по норме решения)
        :param auto_scale: Автоматическое масштабирование gamma и alpha относительно нормы матрицы A
                           (критически важно для перехода от безразмерных задач гравиразведки
                           к магниторазведке в реальных метрах, где ||A|| ~ 10^-6)
        :param non_negative: Ограничивать ли результат снизу нулем (lambda >= 0)
        :return: Восстановленный вектор намагниченности lambda
        """
        L = self.survey.L
        A = L.T @ L
        b = L.T @ g_obs

        # Определение масштаба матрицы A
        if auto_scale:
            diag_mean = float(np.mean(np.diag(A)))
            scale = diag_mean if diag_mean > 0 else 1.0
            eff_gamma = gamma * scale
            eff_alpha = alpha * scale
        else:
            eff_gamma = gamma
            eff_alpha = alpha

        # Регуляризующие добавки
        reg_matrix = np.zeros((self.n_cells, self.n_cells), dtype=np.float64)
        if eff_gamma > 0:
            reg_matrix += self.build_C(eff_gamma)
        if eff_alpha > 0:
            reg_matrix += eff_alpha * np.eye(self.n_cells)

        # Решение СЛАУ
        if eff_gamma == 0 and eff_alpha == 0:
            reg_matrix += 1e-12 * np.eye(self.n_cells)

        rho_est = np.linalg.solve(A + reg_matrix, b)

        if non_negative:
            rho_est = np.clip(rho_est, 0.0, None)

        return rho_est

    def compute_metrics(
        self,
        g_obs: np.ndarray,
        lambda_est: np.ndarray,
        lambda_true: np.ndarray = None,
    ) -> dict:
        """
        Вычисление невязок и метрик качества инверсии.
        """
        g_calc = self.survey.forward_solve(lambda_est)
        diff = g_obs - g_calc

        # Норма невязки
        misfit_l2 = np.linalg.norm(diff)
        rel_misfit = misfit_l2 / (np.linalg.norm(g_obs) + 1e-12) * 100.0

        # R_sq по формуле (39) пособия:
        denom = np.maximum(np.abs(g_obs), np.abs(g_calc)) + 1e-12
        r_sq = 100.0 * np.sqrt(np.mean((diff / denom) ** 2))

        metrics = {
            "g_calc": g_calc,
            "residual": diff,
            "misfit_norm": misfit_l2,
            "rel_misfit_percent": rel_misfit,
            "R_sq": r_sq,
        }

        if lambda_true is not None:
            metrics["model_mae"] = np.mean(np.abs(lambda_est - lambda_true))
            metrics["model_mse"] = np.mean((lambda_est - lambda_true) ** 2)

        return metrics
