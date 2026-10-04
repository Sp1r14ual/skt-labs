import torch
import torch.nn as nn


class GravityNN(nn.Module):
    """
    Полносвязная нейронная сеть для решения обратной задачи гравиразведки.
    Входной слой: количество приемников (100).
    Скрытые слои: 2 скрытых слоя (512 и 1024 нейрона) с функциями активации ReLU,
                 батч-нормализацией и дропаутом для регуляризации.
    Выходной слой: количество ячеек сетки (800 = 40x20).
    """
    def __init__(self, in_features=100, out_features=800, hidden_dims=(512, 1024), dropout=0.1):
        super(GravityNN, self).__init__()

        layers = []
        prev_dim = in_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # Финальный полносвязный слой к 800 ячейкам
        layers.append(nn.Linear(prev_dim, out_features))
        # ReLU на выходе, так как избыточная плотность аномалии неотрицательна (rho >= 0)
        layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
