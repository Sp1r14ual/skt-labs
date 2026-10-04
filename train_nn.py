import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from gravity_core import GravitySolver
from model import GravityNN


def train():
    print("=" * 60)
    print("НАЧАЛО ОБУЧЕНИЯ НЕЙРОННОЙ СЕТИ ДЛЯ ГРАВИРАЗВЕДКИ")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    # 1. Инициализация солвера и генерация обучающего датасета
    solver = GravitySolver()
    n_samples = 3000
    print(f"\n1. Генерация синтетического датасета ({n_samples} геологических моделей)...")
    start_gen = time.time()
    X, Y = solver.generate_dataset(n_samples=n_samples, noise_std=0.01)
    print(f"   Датасет успешно сгенерирован за {time.time() - start_gen:.2f} сек.")
    print(f"   Форма X (сигналы на 100 приемниках): {X.shape}")
    print(f"   Форма Y (плотности на 800 ячейках):  {Y.shape}")

    # 2. Разделение датасета на train/val в соотношении 90/10 (согласно заданию)
    split_idx = int(0.9 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    Y_train, Y_val = Y[:split_idx], Y[split_idx:]

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. Создание модели, функции потерь и оптимизатора
    model = GravityNN(in_features=solver.n_receivers, out_features=solver.n_cells).to(device)
    criterion_loss = nn.MSELoss()  # Функция потерь MSE (согласно методичке, формула 61)
    criterion_mae = nn.L1Loss()    # Метрика качества MAE (согласно методичке, формула 62)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 4. Цикл обучения с адаптивным количеством эпох (отслеживание val_loss)
    n_epochs = 60
    train_losses = []
    val_losses = []
    val_maes = []

    best_val_loss = float('inf')
    best_weights_path = os.path.join(os.path.dirname(__file__), "gravity_nn.pth")

    print("\n2. Старт обучения нейросети...")
    start_train = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        running_train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion_loss(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * len(batch_x)

        epoch_train_loss = running_train_loss / len(train_dataset)

        # Валидация
        model.eval()
        running_val_loss = 0.0
        running_val_mae = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion_loss(outputs, batch_y)
                mae = criterion_mae(outputs, batch_y)

                running_val_loss += loss.item() * len(batch_x)
                running_val_mae += mae.item() * len(batch_x)

        epoch_val_loss = running_val_loss / len(val_dataset)
        epoch_val_mae = running_val_mae / len(val_dataset)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        val_maes.append(epoch_val_mae)

        scheduler.step(epoch_val_loss)

        # Сохранение лучшей модели
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_weights_path)
            saved_mark = "(*сохранена лучшая)"
        else:
            saved_mark = ""

        if epoch % 5 == 0 or epoch == 1 or epoch == n_epochs:
            print(f"Эпоха [{epoch:02d}/{n_epochs:02d}] | "
                  f"Train Loss (MSE): {epoch_train_loss:.6f} | "
                  f"Val Loss (MSE): {epoch_val_loss:.6f} | "
                  f"Val MAE: {epoch_val_mae:.6f} {saved_mark}")

    total_time = time.time() - start_train
    print(f"\n3. Обучение успешно завершено за {total_time:.2f} сек!")
    print(f"   Лучший Val Loss (MSE): {best_val_loss:.6f}")
    print(f"   Файл весов сохранен: {best_weights_path}")

    # 5. Построение и сохранение графиков функции потерь (для отчета)
    plot_path = os.path.join(os.path.dirname(__file__), "loss_history.png")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss (MSE)", color="blue", lw=2)
    plt.plot(val_losses, label="Validation Loss (MSE)", color="orange", lw=2, linestyle="--")
    plt.title("Динамика функции потерь в процессе обучения нейросети", fontsize=12)
    plt.xlabel("Эпоха", fontsize=11)
    plt.ylabel("Функция потерь (MSE)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.yscale("log")
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"   График функции потерь сохранен: {plot_path}")


if __name__ == "__main__":
    train()
