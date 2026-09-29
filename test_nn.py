import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

from gravity_core import GravitySolver
from model import GravityNN


def test_comparison():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    solver = GravitySolver()

    # 1. Загрузка обученной нейросети
    weights_path = os.path.join(os.path.dirname(__file__), "gravity_nn.pth")
    if not os.path.exists(weights_path):
        print(f"Ошибка: файл весов {weights_path} не найден! Сначала запустите train_nn.py")
        return

    model = GravityNN(in_features=solver.n_receivers, out_features=solver.n_cells).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    # 2. Формирование тестовой модели из первой лабораторной (прямоугольная аномалия)
    true_rho = solver.get_standard_test_model()

    # Прямая задача: расчет наблюдаемого поля с шумом
    g_true = solver.forward(true_rho)
    noise = 0.01 * np.random.randn(len(g_true))
    g_obs = g_true + noise

    # 3. Классическое решение обратной задачи с гамма-регуляризацией
    start_classic = time.time()
    gamma = 1e-1
    rho_classic = solver.solve_classical_inverse(g_obs, gamma=gamma)
    time_classic = time.time() - start_classic
    g_classic = solver.forward(rho_classic)
    misfit_classic = np.linalg.norm(g_obs - g_classic)

    # 4. Решение обратной задачи с помощью обученной нейросети
    start_nn = time.time()
    with torch.no_grad():
        x_tensor = torch.from_numpy(g_obs.astype(np.float32)).unsqueeze(0).to(device)
        pred_tensor = model(x_tensor)
        rho_nn = pred_tensor.squeeze(0).cpu().numpy()
    time_nn = time.time() - start_nn
    g_nn = solver.forward(rho_nn)
    misfit_nn = np.linalg.norm(g_obs - g_nn)

    # 5. Метрики расхождения с истинной плотностью
    mse_density_classic = np.mean((true_rho - rho_classic)**2)
    mse_density_nn = np.mean((true_rho - rho_nn)**2)

    print("=" * 65)
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ МЕТОДОВ ДЛЯ ТЕСТОВОЙ МОДЕЛИ")
    print("=" * 65)
    print(f"Классическая инверсия (gamma={gamma}):")
    print(f"  - Время решения:       {time_classic * 1000:.2f} мс")
    print(f"  - Невязка поля:        {misfit_classic:.5f}")
    print(f"  - MSE ошибки плотности:{mse_density_classic:.5f}")
    print("\nИнверсия с помощью Нейросети:")
    print(f"  - Время решения:       {time_nn * 1000:.2f} мс (в {time_classic / max(time_nn, 1e-5):.1f}x быстрее!)")
    print(f"  - Невязка поля:        {misfit_nn:.5f}")
    print(f"  - MSE ошибки плотности:{mse_density_nn:.5f}")
    print("=" * 65)

    # 6. Визуализация и сохранение сводного рисунка для отчета
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # --- График сигналов вдоль профиля ---
    ax_sig = axes[0, 0]
    rec_x = solver.receivers[:, 0]
    ax_sig.plot(rec_x, g_obs, 'k-', lw=1.5, label='Наблюденные (g_obs)')
    ax_sig.plot(rec_x, g_classic, 'g--', lw=1.8, label=f'Классич. регуляриз. (невязка={misfit_classic:.3f})')
    ax_sig.plot(rec_x, g_nn, 'r:', lw=2.2, label=f'Нейросеть (невязка={misfit_nn:.3f})')
    ax_sig.set_title("Сигналы вертикальной компоненты Δg вдоль профиля", fontsize=11, fontweight='bold')
    ax_sig.set_xlabel("X (м)")
    ax_sig.set_ylabel("Δg")
    ax_sig.grid(True, linestyle=":", alpha=0.6)
    ax_sig.legend(fontsize=9)

    # Функция отрисовки 2D матрицы плотности
    def draw_density(ax, rho_vec, title):
        mat = rho_vec.reshape((solver.nx, solver.nz)).T
        im = ax.imshow(mat, origin='lower', aspect='auto', cmap='viridis',
                       extent=[0, solver.nx * solver.dx, -solver.nz * solver.dz, 0])
        fig.colorbar(im, ax=ax, label="Плотность ρ")
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel("X")
        ax.set_ylabel("Z")

    # --- Истинная модель ---
    draw_density(axes[0, 1], true_rho, "Истинная модель (1-й семестр)")

    # --- Классическая восстановленная модель ---
    draw_density(axes[1, 0], rho_classic, f"Классич. инверсия (MSE плотности={mse_density_classic:.4f})")

    # --- Восстановленная нейросетью модель ---
    draw_density(axes[1, 1], rho_nn, f"Нейросеть (MSE плотности={mse_density_nn:.4f})")

    plt.tight_layout()
    result_img = os.path.join(os.path.dirname(__file__), "nn_inversion_result.png")
    plt.savefig(result_img, dpi=300)
    plt.close()
    print(f"Сводный график результатов сохранен: {result_img}")


if __name__ == "__main__":
    test_comparison()
