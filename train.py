"""
Модуль обучения нейронной сети для решения обратной задачи магниторазведки.
Отслеживает функцию потерь (MSE) и метрику (MAE) на обучающей и валидационной выборках,
поддерживает адаптивный ранний останов (EarlyStopping) и сохранение лучшей модели.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from neural_net import MagneticInversionNet
from dataset import DatasetGenerator
from forward_problem import MagneticSurvey2D


class ModelTrainer:
    def __init__(
        self,
        model: MagneticInversionNet,
        learning_rate: float = 0.002,
        weight_decay: float = 1e-5,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()  # Формула (61) пособия: MSE
        self.metric_fn = nn.L1Loss()  # Формула (62) пособия: MAE

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_mae": [],
            "val_mae": [],
        }

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 150,
        batch_size: int = 64,
        patience: int = 25,
        save_path: str = "best_model.pt",
        callback: callable = None,
    ) -> dict:
        """
        Запуск цикла адаптивного обучения.
        :param callback: Опциональная функция callback(epoch, total_epochs, train_loss, val_loss, train_mae, val_mae)
        """
        train_dataset = TensorDataset(
            torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        best_val_loss = float("inf")
        patience_counter = 0

        # Планировщик скорости обучения (адаптивное уменьшение шага eta при выходе на плато)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5
        )

        start_time = time.time()
        print(
            f"Начало обучения: {len(X_train)} обучающих, {len(X_val)} валидационных примеров..."
        )

        for epoch in range(1, epochs + 1):
            # 1. Обучение
            self.model.train()
            total_train_loss = 0.0
            total_train_mae = 0.0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = self.criterion(pred, batch_y)
                loss.backward()
                self.optimizer.step()

                total_train_loss += loss.item() * len(batch_x)
                total_train_mae += self.metric_fn(
                    pred, batch_y
                ).item() * len(batch_x)

            epoch_train_loss = total_train_loss / len(X_train)
            epoch_train_mae = total_train_mae / len(X_train)

            # 2. Валидация
            self.model.eval()
            total_val_loss = 0.0
            total_val_mae = 0.0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    pred = self.model(batch_x)
                    val_loss = self.criterion(pred, batch_y)
                    total_val_loss += val_loss.item() * len(batch_x)
                    total_val_mae += self.metric_fn(
                        pred, batch_y
                    ).item() * len(batch_x)

            epoch_val_loss = total_val_loss / len(X_val)
            epoch_val_mae = total_val_mae / len(X_val)

            scheduler.step(epoch_val_loss)

            self.history["train_loss"].append(epoch_train_loss)
            self.history["val_loss"].append(epoch_val_loss)
            self.history["train_mae"].append(epoch_train_mae)
            self.history["val_mae"].append(epoch_val_mae)

            # Проверка Early Stopping и сохранение
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1

            if callback is not None:
                callback(
                    epoch,
                    epochs,
                    epoch_train_loss,
                    epoch_val_loss,
                    epoch_train_mae,
                    epoch_val_mae,
                )

            if epoch % 10 == 0 or epoch == epochs or patience_counter == 0:
                print(
                    f"Эпоха {epoch:3d}/{epochs} | Train MSE: {epoch_train_loss:.6f} | "
                    f"Val MSE: {epoch_val_loss:.6f} | Val MAE: {epoch_val_mae:.4f}"
                )

            if patience_counter >= patience:
                print(
                    f"Адаптивный ранний останов на эпохе {epoch} (валидационный лосс не улучшался {patience} эпох)."
                )
                break

        elapsed = time.time() - start_time
        print(
            f"Обучение завершено за {elapsed:.2f} сек. Лучший Val MSE: {best_val_loss:.6f}"
        )

        # Загружаем лучшие сохраненные веса
        if save_path and os.path.exists(save_path):
            self.model.load_state_dict(
                torch.load(save_path, weights_only=True)
            )

        return self.history

    def plot_history(self, save_fig: str = "loss_history.png"):
        """Построение графиков функции потерь и метрик в логарифмическом масштабе (как на Рис. 43 пособия)."""
        epochs = range(1, len(self.history["train_loss"]) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        ax1.plot(epochs, self.history["train_loss"], label="Обучающая (Train MSE)")
        ax1.plot(
            epochs,
            self.history["val_loss"],
            "--",
            label="Валидационная (Val MSE)",
        )
        ax1.set_yscale("log")
        ax1.set_title("Функция потерь (MSE Loss)")
        ax1.set_xlabel("Эпоха")
        ax1.set_ylabel("MSE (log scale)")
        ax1.grid(True, which="both", ls=":")
        ax1.legend()

        ax2.plot(epochs, self.history["train_mae"], label="Train MAE")
        ax2.plot(epochs, self.history["val_mae"], "--", label="Val MAE")
        ax2.set_title("Метрика абсолютной ошибки (MAE)")
        ax2.set_xlabel("Эпоха")
        ax2.set_ylabel("MAE")
        ax2.grid(True, which="both", ls=":")
        ax2.legend()

        plt.tight_layout()
        if save_fig:
            plt.savefig(save_fig, dpi=200)
            print(f"График обучения сохранен в: {save_fig}")
        plt.close()


def run_full_training(
    n_samples: int = 3000,
    epochs: int = 120,
    model_save_path: str = "best_model.pt",
    dataset_path: str = "dataset.npz",
):
    """Функция автоматического запуска генерации датасета и полного обучения."""
    survey = MagneticSurvey2D(nx=20, nz=10, n_receivers=40)
    gen = DatasetGenerator(survey)

    if os.path.exists(dataset_path):
        print(f"Загрузка существующего датасета из {dataset_path}...")
        X_tr, y_tr, X_val, y_val = gen.load_dataset(dataset_path)
    else:
        print(f"Генерация датасета из {n_samples} моделей...")
        X_tr, y_tr, X_val, y_val = gen.generate_dataset(
            n_samples=n_samples, save_path=dataset_path
        )

    model = MagneticInversionNet(
        n_inputs=survey.n_receivers,
        n_outputs=survey.n_cells,
        hidden_dim=400,
        num_hidden_layers=2,
        dropout_rate=0.15,
        activation="relu",
    )

    trainer = ModelTrainer(model, learning_rate=0.002)
    history = trainer.train(
        X_tr,
        y_tr,
        X_val,
        y_val,
        epochs=epochs,
        batch_size=64,
        save_path=model_save_path,
    )
    trainer.plot_history("loss_history.png")
    return model, trainer, history


if __name__ == "__main__":
    run_full_training()
