"""
Скрипт автоматического проведения серии вычислительных экспериментов
для проверки качества обученной нейронной сети, ее сравнения с классической инверсией
и оценки работы препроцессора при шумах и нерегулярных приемниках.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from forward_problem import MagneticSurvey2D
from preprocessor import SignalPreprocessor
from dataset import DatasetGenerator
from neural_net import MagneticInversionNet
from classical_inversion import ClassicalInversion


def setup_style():
    """Настройка эстетики графиков matplotlib."""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#d0d0d0"
    plt.rcParams["grid.linestyle"] = ":"


def plot_2d_model(
    ax,
    vector_rho: np.ndarray,
    survey: MagneticSurvey2D,
    title: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "viridis",
):
    """Отрисовка двумерного разреза распределения намагниченности среды."""
    grid = np.zeros((survey.nz, survey.nx))
    for ix in range(survey.nx):
        for iz in range(survey.nz):
            grid[iz, ix] = vector_rho[survey.cell_index(ix, iz)]

    extent = [
        survey.x_min,
        survey.x_min + survey.nx * survey.dx,
        survey.z_top - survey.nz * survey.dz,
        survey.z_top,
    ]
    im = ax.imshow(
        grid,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("X (м)", fontsize=9)
    ax.set_ylabel("Z (м, глубина)", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)
    return im


def run_all_experiments(
    weights_path: str = "best_model.pt", results_dir: str = "results"
):
    os.makedirs(results_dir, exist_ok=True)
    setup_style()

    print("=== ЗАПУСК ВЫЧИСЛИТЕЛЬНЫХ ЭКСПЕРИМЕНТОВ ===")
    survey = MagneticSurvey2D(nx=20, nz=10, n_receivers=40)
    gen = DatasetGenerator(survey)
    classical = ClassicalInversion(survey)

    # Инициализация и загрузка обученной нейросети
    model = MagneticInversionNet(
        n_inputs=survey.n_receivers,
        n_outputs=survey.n_cells,
        hidden_dim=400,
        num_hidden_layers=2,
    )
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        print(f"Успешно загружены веса нейросети из: {weights_path}")
    else:
        print(f"Предупреждение: файл весов {weights_path} не найден! Запустите train.py.")
        return

    model.eval()

    def predict_nn(sig):
        with torch.no_grad():
            inp = torch.from_numpy(sig).float().unsqueeze(0)
            out = model(inp).squeeze(0).numpy()
        return out

    # ЭКСПЕРИМЕНТ 1: Тестовая модель
    print(
        "\n[1/5] Эксперимент 1: Восстановление тестовой модели..."
    )
    true_lab1 = gen.get_benchmark_model("test_model")
    sig_lab1 = survey.forward_solve(true_lab1)
    # Добавим шум 1%
    np.random.seed(42)
    sig_lab1_obs = survey.add_noise(sig_lab1, noise_level=0.01)

    # Решение классической инверсией: без регуляризации и с регуляризацией gamma
    rec_class_noreg = classical.solve(sig_lab1_obs, gamma=0.0)
    rec_class_reg = classical.solve(sig_lab1_obs, gamma=0.05)

    # Решение обученной нейросетью
    rec_nn = predict_nn(sig_lab1_obs)

    # Рассчитанные поля
    sig_calc_nn = survey.forward_solve(rec_nn)
    sig_calc_reg = survey.forward_solve(rec_class_reg)

    # Метрики
    m_nn = classical.compute_metrics(sig_lab1_obs, rec_nn, true_lab1)
    m_reg = classical.compute_metrics(sig_lab1_obs, rec_class_reg, true_lab1)

    print(
        f"  ИНС: MAE модели = {m_nn['model_mae']:.4f}, Невязка сигнала = {m_nn['misfit_norm']:.6f}, R_sq = {m_nn['R_sq']:.2f}%"
    )
    print(
        f"  Классическая gamma-рег: MAE модели = {m_reg['model_mae']:.4f}, Невязка = {m_reg['misfit_norm']:.6f}, R_sq = {m_reg['R_sq']:.2f}%"
    )

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.9])

    ax_true = fig.add_subplot(gs[0, 0])
    ax_nn = fig.add_subplot(gs[0, 1])
    ax_reg = fig.add_subplot(gs[0, 2])
    ax_sig = fig.add_subplot(gs[1, :2])
    ax_res = fig.add_subplot(gs[1, 2])

    im1 = plot_2d_model(
        ax_true, survey=survey, vector_rho=true_lab1, title="Истинная модель (Тестовая)"
    )
    plt.colorbar(im1, ax=ax_true, fraction=0.046, pad=0.04)

    im2 = plot_2d_model(
        ax_nn,
        survey=survey,
        vector_rho=rec_nn,
        title=f"Восстановлено ИНС (MAE={m_nn['model_mae']:.3f})",
    )
    plt.colorbar(im2, ax=ax_nn, fraction=0.046, pad=0.04)

    im3 = plot_2d_model(
        ax_reg,
        survey=survey,
        vector_rho=rec_class_reg,
        title=f"γ-регуляризация (MAE={m_reg['model_mae']:.3f})",
    )
    plt.colorbar(im3, ax=ax_reg, fraction=0.046, pad=0.04)

    # Сигналы
    rx = survey.receiver_x
    ax_sig.plot(
        rx, sig_lab1_obs, "k-", linewidth=2.0, label="Наблюденный сигнал (1% шум)"
    )
    ax_sig.plot(
        rx,
        sig_calc_nn,
        "r--",
        linewidth=2.0,
        label=f"Отклик модели ИНС (R_sq={m_nn['R_sq']:.1f}%)",
    )
    ax_sig.plot(
        rx,
        sig_calc_reg,
        "b:",
        linewidth=2.0,
        label=f"Отклик γ-регуляризации (R_sq={m_reg['R_sq']:.1f}%)",
    )
    ax_sig.set_title(
        "Сравнение сигналов B_x вдоль профиля", fontsize=11, fontweight="bold"
    )
    ax_sig.set_xlabel("X (м)", fontsize=9)
    ax_sig.set_ylabel("Индукция B_x", fontsize=9)
    ax_sig.grid(True)
    ax_sig.legend()

    # Невязка
    ax_res.plot(
        rx,
        sig_lab1_obs - sig_calc_nn,
        "r-",
        label=f"Невязка ИНС (||e||={m_nn['misfit_norm']:.1e})",
    )
    ax_res.plot(
        rx,
        sig_lab1_obs - sig_calc_reg,
        "b--",
        label=f"Невязка γ-рег (||e||={m_reg['misfit_norm']:.1e})",
    )
    ax_res.set_title("График невязки поля", fontsize=11, fontweight="bold")
    ax_res.set_xlabel("X (м)", fontsize=9)
    ax_res.set_ylabel("ΔB_x", fontsize=9)
    ax_res.grid(True)
    ax_res.legend()

    plt.tight_layout()
    p1 = os.path.join(results_dir, "exp1_test_model.png")
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"  Сохранен график: {p1}")

    # ЭКСПЕРИМЕНТ 2: Наклонная аномалия
    print("\n[2/5] Эксперимент 2: Наклонная аномалия (дайка/разлом)...")
    true_tilted = gen.get_benchmark_model("tilted")
    sig_tilted = survey.forward_solve(true_tilted)
    sig_tilted_obs = survey.add_noise(sig_tilted, noise_level=0.01)

    rec_tilted_nn = predict_nn(sig_tilted_obs)
    rec_tilted_reg = classical.solve(sig_tilted_obs, gamma=0.03)

    m_tilted_nn = classical.compute_metrics(
        sig_tilted_obs, rec_tilted_nn, true_tilted
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plot_2d_model(
        axes[0], survey=survey, vector_rho=true_tilted, title="Истинная наклонная аномалия"
    )
    plot_2d_model(
        axes[1],
        survey=survey,
        vector_rho=rec_tilted_nn,
        title=f"Восстановление ИНС (MAE={m_tilted_nn['model_mae']:.3f})",
    )
    plot_2d_model(
        axes[2],
        survey=survey,
        vector_rho=rec_tilted_reg,
        title="Восстановление γ-регуляризацией",
    )

    plt.tight_layout()
    p2 = os.path.join(results_dir, "exp2_tilted_anomaly.png")
    plt.savefig(p2, dpi=200)
    plt.close()
    print(f"  Сохранен график: {p2}")

    # ЭКСПЕРИМЕНТ 3: Два разделенных и перекрывающихся объекта
    print("\n[3/5] Эксперимент 3: Системы из нескольких тел...")
    true_sep = gen.get_benchmark_model("two_separated")
    sig_sep = survey.forward_solve(true_sep)
    rec_sep_nn = predict_nn(sig_sep)

    true_stk = gen.get_benchmark_model("two_stacked")
    sig_stk = survey.forward_solve(true_stk)
    rec_stk_nn = predict_nn(sig_stk)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_2d_model(
        axes[0, 0],
        survey=survey,
        vector_rho=true_sep,
        title="Истинная: 2 разделенных тела",
    )
    plot_2d_model(
        axes[0, 1],
        survey=survey,
        vector_rho=rec_sep_nn,
        title="Восстановлено ИНС (2 разделенных)",
    )
    plot_2d_model(
        axes[1, 0],
        survey=survey,
        vector_rho=true_stk,
        title="Истинная: 2 перекрывающихся тела",
    )
    plot_2d_model(
        axes[1, 1],
        survey=survey,
        vector_rho=rec_stk_nn,
        title="Восстановлено ИНС (2 перекрывающихся)",
    )

    plt.tight_layout()
    p3 = os.path.join(results_dir, "exp3_multiple_bodies.png")
    plt.savefig(p3, dpi=200)
    plt.close()
    print(f"  Сохранен график: {p3}")

    # ЭКСПЕРИМЕНТ 4: Устойчивость к шуму (1%, 5%, 10%)
    print("\n[4/5] Эксперимент 4: Исследование устойчивости к шуму...")
    noise_levels = [0.01, 0.05, 0.10]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    plot_2d_model(
        axes[0], survey=survey, vector_rho=true_lab1, title="Истинная модель"
    )

    for idx, nl in enumerate(noise_levels):
        sig_noisy = survey.add_noise(sig_lab1, noise_level=nl, seed=42)
        rec_noise = predict_nn(sig_noisy)
        m_nl = classical.compute_metrics(sig_noisy, rec_noise, true_lab1)
        plot_2d_model(
            axes[idx + 1],
            survey=survey,
            vector_rho=rec_noise,
            title=f"Шум {int(nl*100)}% (MAE={m_nl['model_mae']:.3f})",
        )

    plt.tight_layout()
    p4 = os.path.join(results_dir, "exp4_noise_robustness.png")
    plt.savefig(p4, dpi=200)
    plt.close()
    print(f"  Сохранен график: {p4}")

    # ЭКСПЕРИМЕНТ 5: Препроцессор (сплайн Эрмита) при нерегулярных приемниках
    print("\n[5/5] Эксперимент 5: Препроцессор и нерегулярная сетка приемников")
    # Создадим нерегулярный профиль (30 приемников с неравномерным шагом)
    np.random.seed(123)
    n_irreg = 30
    x_irreg = np.sort(np.random.uniform(survey.x_min, survey.x_min + survey.nx * survey.dx, n_irreg))
    # Приемники
    rec_irreg = np.column_stack(
        [
            x_irreg,
            np.full(n_irreg, survey.receiver_y),
            np.full(n_irreg, survey.receiver_z),
        ]
    )
    L_irreg = survey.build_L(rec_irreg)
    sig_irreg_clean = survey.forward_solve(true_lab1, L=L_irreg)
    # Добавим шум 10% (как на стр. 74 пособия)
    sig_irreg_noisy = survey.add_noise(sig_irreg_clean, noise_level=0.10, seed=123)

    # Применение сглаживающего кубического сплайна (препроцессор)
    preprocessor = SignalPreprocessor(
        target_x=survey.receiver_x,
        use_hermite=True,
        smoothing_factor=1e-3,
        n_elements=20,
    )
    sig_preprocessed = preprocessor.process(x_irreg, sig_irreg_noisy)

    # Подача предобработанного сигнала на вход нейросети
    rec_from_preprocessed = predict_nn(sig_preprocessed)

    fig, (ax_sig, ax_m1, ax_m2) = plt.subplots(1, 3, figsize=(16, 4.5))

    # Сравнение профилей
    ax_sig.plot(
        x_irreg,
        sig_irreg_noisy,
        "ko",
        markersize=5,
        alpha=0.6,
        label=f"Нерегулярные замеры ({n_irreg} шт, шум 10%)",
    )
    ax_sig.plot(
        survey.receiver_x,
        sig_lab1,
        "g-",
        linewidth=2.0,
        label="Истинный непрерывный сигнал",
    )
    ax_sig.plot(
        survey.receiver_x,
        sig_preprocessed,
        "r--",
        linewidth=2.2,
        label="Сплайн Эрмита (Препроцессор)",
    )
    ax_sig.set_title("Работа препроцессора (интерполяция и сглаживание)")
    ax_sig.set_xlabel("X (м)")
    ax_sig.set_ylabel("B_x")
    ax_sig.legend()
    ax_sig.grid(True)

    plot_2d_model(
        ax_m1,
        survey=survey,
        vector_rho=true_lab1,
        title="Истинная модель среды",
    )
    plot_2d_model(
        ax_m2,
        survey=survey,
        vector_rho=rec_from_preprocessed,
        title="ИНС по предобработанным данным",
    )

    plt.tight_layout()
    p5 = os.path.join(results_dir, "exp5_preprocessor_spline.png")
    plt.savefig(p5, dpi=200)
    plt.close()
    print(f"  Сохранен график: {p5}")

    print("\nВсе вычислительные эксперименты успешно завершены!")
    print(f"Результаты графиков сохранены в директории: {results_dir}")


if __name__ == "__main__":
    run_all_experiments()
