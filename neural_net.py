"""
Модуль архитектуры нейронной сети для решения обратной задачи магниторазведки.
Полносвязная архитектура (MLP / DNN)
(1-3 скрытых слоя, Batch Normalization, Dropout, функции активации ReLU/Tanh, Adam).
"""

import torch
import torch.nn as nn


class MagneticInversionNet(nn.Module):
    def __init__(
        self,
        n_inputs: int = 40,
        n_outputs: int = 200,
        hidden_dim: int = 400,
        num_hidden_layers: int = 2,
        dropout_rate: float = 0.15,
        activation: str = "relu",
        use_batch_norm: bool = True,
    ):
        """
        :param n_inputs: Размерность входного вектора (количество точек профиля/приемников)
        :param n_outputs: Размерность выходного вектора (количество ячеек сетки Nx * Nz)
        :param hidden_dim: Количество нейронов в скрытом слое (по умолчанию 400 по пособию)
        :param num_hidden_layers: Количество скрытых слоев (от 1 до 3)
        :param dropout_rate: Процент прореживания (Dropout) для предотвращения переобучения
        :param activation: Функция активации ('relu' или 'tanh')
        :param use_batch_norm: Использовать ли пакетную нормализацию (BatchNorm1d)
        """
        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        # Входная нормализация (пособие, стр. 58: "Для повышения производительности...
        # возможно использование во входном слое пакетной нормализации")
        self.input_norm = (
            nn.BatchNorm1d(n_inputs, affine=True)
            if use_batch_norm
            else nn.Identity()
        )

        act_cls = nn.ReLU if activation.lower() == "relu" else nn.Tanh

        layers = []
        current_dim = n_inputs

        for i in range(num_hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_cls())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim

        # Выходной слой
        layers.append(nn.Linear(current_dim, n_outputs))
        # Ограничение выхода в диапазон [0, 1] для физически корректных значений lambda
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход:
        x: Tensor формы (batch_size, n_inputs)
        return: Tensor формы (batch_size, n_outputs) со значениями lambda_i
        """
        x_norm = self.input_norm(x)
        return self.network(x_norm)
