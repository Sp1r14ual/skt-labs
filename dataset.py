"""
Модуль генерации датасета синтетических моделей и сигналов для обучения нейронной сети.
Реализует Способ 1 (случайные изолированные тела/разломы) и
Способ 2 (сбалансированное разбиение на блоки через случайные разрезы по осям Ox и Oz
"""

import os
import numpy as np
from forward_problem import MagneticSurvey2D


class DatasetGenerator:
    def __init__(self, survey: MagneticSurvey2D = None):
        """
        :param survey: Экземпляр MagneticSurvey2D с параметрами геометрии и матрицей L.
        """
        if survey is None:
            self.survey = MagneticSurvey2D(nx=20, nz=10, n_receivers=40)
        else:
            self.survey = survey

        self.nx = self.survey.nx
        self.nz = self.survey.nz
        self.n_cells = self.survey.n_cells

    def generate_balanced_model(
        self,
        max_cuts_x: int = 4,
        max_cuts_z: int = 4,
        anomaly_prob: float = 0.35,
        multi_level: bool = True,
    ) -> np.ndarray:
        """
        Способ 2:
        1. Выбирается случайное число вертикальных разрезов M_x в [0, max_cuts_x].
        2. Выбираются M_x уникальных индексов вертикальных линий [1, nx-1].
        3. Для каждой вертикальной области выбирается M_z^i горизонтальных разрезов [1, nz-1].
        4. Полученным блокам случайным образом присваивается намагниченность lambda в [0, 1].
        """
        grid = np.zeros((self.nz, self.nx), dtype=np.float64)

        # Вертикальные разрезы
        n_cuts_x = np.random.randint(0, min(max_cuts_x + 1, self.nx))
        if n_cuts_x > 0:
            x_splits = np.sort(
                np.random.choice(range(1, self.nx), size=n_cuts_x, replace=False)
            )
        else:
            x_splits = np.array([], dtype=int)
        x_bounds = [0] + list(x_splits) + [self.nx]

        # Для каждой вертикальной полосы выполняем горизонтальные разрезы
        for i in range(len(x_bounds) - 1):
            x_start = x_bounds[i]
            x_end = x_bounds[i + 1]

            n_cuts_z = np.random.randint(0, min(max_cuts_z + 1, self.nz))
            if n_cuts_z > 0:
                z_splits = np.sort(
                    np.random.choice(
                        range(1, self.nz), size=n_cuts_z, replace=False
                    )
                )
            else:
                z_splits = np.array([], dtype=int)
            z_bounds = [0] + list(z_splits) + [self.nz]

            for j in range(len(z_bounds) - 1):
                z_start = z_bounds[j]
                z_end = z_bounds[j + 1]

                # Решаем, будет ли данный блок аномальным
                if np.random.rand() < anomaly_prob:
                    if multi_level:
                        val = np.random.uniform(0.3, 1.0)
                    else:
                        val = 1.0
                else:
                    val = 0.0

                grid[z_start:z_end, x_start:x_end] = val

        # Разворачиваем в 1D вектор с нумерацией: idx = iz + ix * nz
        vector = np.zeros(self.n_cells, dtype=np.float64)
        for ix in range(self.nx):
            for iz in range(self.nz):
                vector[self.survey.cell_index(ix, iz)] = grid[iz, ix]

        return vector

    def generate_isolated_bodies_model(
        self,
        num_bodies: int = None,
        multi_level: bool = True,
    ) -> np.ndarray:
        """
        Способ 1: случайное размещение прямоугольных или наклонных тел.
        """
        grid = np.zeros((self.nz, self.nx), dtype=np.float64)

        if num_bodies is None:
            num_bodies = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])

        for _ in range(num_bodies):
            body_type = np.random.choice(["rect", "tilted"])
            val = np.random.uniform(0.4, 1.0) if multi_level else 1.0

            if body_type == "rect":
                w = np.random.randint(2, max(3, self.nx // 3))
                h = np.random.randint(1, max(2, self.nz // 2))
                x0 = np.random.randint(0, self.nx - w + 1)
                z0 = np.random.randint(0, self.nz - h + 1)
                grid[z0 : z0 + h, x0 : x0 + w] = val
            else:
                # Наклонная дайка
                length = np.random.randint(3, max(4, min(self.nx, self.nz)))
                start_x = np.random.randint(0, self.nx - length)
                start_z = np.random.randint(0, self.nz - length)
                direction = np.random.choice([-1, 1])
                for step in range(length):
                    cx = start_x + step
                    cz = (
                        start_z + step
                        if direction == 1
                        else start_z + (length - 1 - step)
                    )
                    cz = min(max(cz, 0), self.nz - 1)
                    grid[cz, cx] = val
                    if cz + 1 < self.nz:
                        grid[cz + 1, cx] = val

        vector = np.zeros(self.n_cells, dtype=np.float64)
        for ix in range(self.nx):
            for iz in range(self.nz):
                vector[self.survey.cell_index(ix, iz)] = grid[iz, ix]

        return vector

    def get_benchmark_model(self, name: str = "test_model") -> np.ndarray:
        """
        Тестовые модели.
        """
        vector = np.zeros(self.n_cells, dtype=np.float64)

        if name in ["test_model", "base_model", "lab1"]:
            # тело плотности rho=1.0 (в координатах сетки)
            if self.nx == 40 and self.nz == 20:
                # Точные индексы: ix in [8, 12), iz in [3, 6)
                for ix in range(8, 12):
                    for iz in range(3, 6):
                        vector[self.survey.cell_index(ix, iz)] = 1.0
            else:
                # Пропорциональное масштабирование на сетку 20x10
                # ix от 4 до 6, iz от 1 до 3 (прямоугольник 2x2 или 3x2)
                for ix in range(4, 7):
                    for iz in range(1, 4):
                        vector[self.survey.cell_index(ix, iz)] = 1.0

        elif name == "medium":
            # Аномалия среднего размера в центре
            mid_x = self.nx // 2
            mid_z = self.nz // 2
            for ix in range(mid_x - 3, mid_x + 3):
                for iz in range(mid_z - 2, mid_z + 2):
                    if 0 <= ix < self.nx and 0 <= iz < self.nz:
                        vector[self.survey.cell_index(ix, iz)] = 1.0

        elif name == "tilted":
            # Наклонное тело
            for step in range(min(self.nx // 2, self.nz)):
                ix = 4 + step
                iz = 1 + step
                if 0 <= ix < self.nx and 0 <= iz < self.nz:
                    vector[self.survey.cell_index(ix, iz)] = 1.0
                    if iz + 1 < self.nz:
                        vector[self.survey.cell_index(ix, iz + 1)] = 1.0

        elif name == "two_separated":
            # Два разделенных тела
            # Тело 1
            for ix in range(2, 6):
                for iz in range(2, 5):
                    if 0 <= ix < self.nx and 0 <= iz < self.nz:
                        vector[self.survey.cell_index(ix, iz)] = 1.0
            # Тело 2
            for ix in range(self.nx - 6, self.nx - 2):
                for iz in range(2, 5):
                    if 0 <= ix < self.nx and 0 <= iz < self.nz:
                        vector[self.survey.cell_index(ix, iz)] = 1.0

        elif name == "two_stacked":
            # Два тела одно над другим
            mid_x = self.nx // 2
            for ix in range(mid_x - 3, mid_x + 3):
                # Верхнее
                if 1 < self.nz:
                    vector[self.survey.cell_index(ix, 1)] = 1.0
                # Нижнее
                for iz in range(max(2, self.nz - 3), self.nz):
                    vector[self.survey.cell_index(ix, iz)] = 1.0

        return vector

    def generate_dataset(
        self,
        n_samples: int = 2500,
        noise_level: float = 0.01,
        save_path: str = None,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Генерация полного датасета (модели среды и сигналы).
        Разбиение 90% обучающая / 10% валидационная/тестовая выборка (в соответствии с заданием).
        :return: X_train, y_train, X_val, y_val
        """
        np.random.seed(seed)
        models = np.zeros((n_samples, self.n_cells), dtype=np.float32)

        # 60% моделей - способ 2 (сбалансированные разрезы)
        # 40% моделей - способ 1 (изолированные тела)
        n_balanced = int(n_samples * 0.6)
        n_isolated = n_samples - n_balanced

        for i in range(n_balanced):
            models[i] = self.generate_balanced_model()

        for i in range(n_isolated):
            models[n_balanced + i] = self.generate_isolated_bodies_model()

        # Также гарантированно добавляем тестовую модель в датасет
        models[-1] = self.get_benchmark_model("test_model")

        # Быстрый векторизованный расчет сигналов: S = models @ L.T
        # signals shape: (n_samples, n_receivers)
        signals = (models @ self.survey.L.T).astype(np.float32)

        # Добавление шума
        if noise_level > 0:
            noise = np.random.normal(
                0.0,
                noise_level * np.max(np.abs(signals), axis=1, keepdims=True),
                size=signals.shape,
            ).astype(np.float32)
            signals += noise

        # Разделение 90% / 10%
        n_train = int(n_samples * 0.9)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        X_train = signals[train_idx]
        y_train = models[train_idx]
        X_val = signals[val_idx]
        y_val = models[val_idx]

        if save_path:
            np.savez_compressed(
                save_path,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
            )
            print(f"Датасет успешно сохранен в: {save_path}")

        return X_train, y_train, X_val, y_val

    @staticmethod
    def load_dataset(
        path: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Загрузка сохраненного датасета из .npz файла."""
        data = np.load(path)
        return (
            data["X_train"],
            data["y_train"],
            data["X_val"],
            data["y_val"],
        )
