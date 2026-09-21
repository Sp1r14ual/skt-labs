"""
Графический интерфейс (GUI) для решения обратной задачи магниторазведки
с использованием искусственных нейронных сетей (ИНС) и сравнения с классической регуляризацией.
Разработан на базе Tkinter и Matplotlib.
"""

import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import torch

from forward_problem import MagneticSurvey2D
from preprocessor import SignalPreprocessor
from dataset import DatasetGenerator
from neural_net import MagneticInversionNet
from classical_inversion import ClassicalInversion
from train import ModelTrainer


class MagneticInversionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Обратная задача магниторазведки с использованием ИНС")
        self.root.geometry("1400x900")
        self.root.minsize(1100, 750)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # 1. Физическая модель и компоненты
        self.survey = MagneticSurvey2D(nx=20, nz=10, n_receivers=40, component="Bx")
        self.dataset_gen = DatasetGenerator(self.survey)
        self.classical_solver = ClassicalInversion(self.survey)

        # 2. Нейросеть
        self.device = "cpu"
        self.nn_model = MagneticInversionNet(
            n_inputs=self.survey.n_receivers,
            n_outputs=self.survey.n_cells,
            hidden_dim=400,
            num_hidden_layers=2,
        )
        self.weights_path = "best_model.pt"
        if os.path.exists(self.weights_path):
            try:
                self.nn_model.load_state_dict(
                    torch.load(self.weights_path, weights_only=True)
                )
                print("Успешно загружены веса нейросети.")
            except Exception as e:
                print(f"Ошибка загрузки весов: {e}")
        self.nn_model.eval()

        # 3. Текущее состояние данных
        self.true_model = self.dataset_gen.get_benchmark_model("base_model")
        self.current_obs_x = self.survey.receiver_x.copy()
        self.current_obs_signal = self.survey.forward_solve(self.true_model)
        self.preprocessed_signal = None
        self.nn_predicted_model = None
        self.reg_predicted_model = None

        # Создание интерфейса
        self._setup_ui()
        self.update_plots()

    def _setup_ui(self):
        # Главный контейнер
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Левая панель управления со скроллом
        left_container = ttk.Frame(main_paned, width=380)
        main_paned.add(left_container, weight=0)

        canvas_scroll = tk.Canvas(left_container, width=380, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_container, orient=tk.VERTICAL, command=canvas_scroll.yview)
        self.control_frame = ttk.Frame(canvas_scroll, padding="8 8 8 8")

        self.control_frame.bind(
            "<Configure>",
            lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all")),
        )
        canvas_scroll.create_window((0, 0), window=self.control_frame, anchor="nw")
        canvas_scroll.configure(yscrollcommand=scrollbar.set)

        canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Правая область графиков
        plot_frame = ttk.Frame(main_paned)
        main_paned.add(plot_frame, weight=1)

        self._build_controls()
        self._build_canvas(plot_frame)

    def _build_controls(self):
        pad_opts = {"padx": 5, "pady": 4}

        # --- Секция 1: Выбор модели среды ---
        grp_model = ttk.LabelFrame(self.control_frame, text="1. Геологическая модель среды")
        grp_model.pack(fill=tk.X, pady=4)

        ttk.Label(grp_model, text="Предустановленные модели:").pack(anchor=tk.W, **pad_opts)
        self.combo_model = ttk.Combobox(
            grp_model,
            values=[
                "Тестовая модель",
                "Аномалия среднего размера",
                "Наклонный пласт / разлом",
                "Два разделенных тела",
                "Два перекрывающихся тела",
                "Случайная блочная модель",
                "Очистить модель",
            ],
            state="readonly",
        )
        self.combo_model.current(0)
        self.combo_model.pack(fill=tk.X, **pad_opts)
        self.combo_model.bind("<<ComboboxSelected>>", self._on_model_selected)

        # --- Секция 2: Параметры съемки и шум ---
        grp_survey = ttk.LabelFrame(self.control_frame, text="2. Параметры измерений")
        grp_survey.pack(fill=tk.X, pady=4)

        # Уровень шума
        noise_frame = ttk.Frame(grp_survey)
        noise_frame.pack(fill=tk.X, **pad_opts)
        ttk.Label(noise_frame, text="Случайный шум:").pack(side=tk.LEFT)
        self.lbl_noise_val = ttk.Label(noise_frame, text="1%", font=("TkDefaultFont", 9, "bold"))
        self.lbl_noise_val.pack(side=tk.RIGHT)

        self.scale_noise = ttk.Scale(
            grp_survey, from_=0, to=20, value=1, orient=tk.HORIZONTAL, command=self._on_noise_change
        )
        self.scale_noise.pack(fill=tk.X, **pad_opts)

        # Сетка приемников
        self.var_grid_type = tk.StringVar(value="regular")
        rb_reg = ttk.Radiobutton(
            grp_survey,
            text="Регулярная сетка (40 приемников)",
            variable=self.var_grid_type,
            value="regular",
            command=self._on_grid_type_change,
        )
        rb_reg.pack(anchor=tk.W, padx=5, pady=2)
        rb_irreg = ttk.Radiobutton(
            grp_survey,
            text="Нерегулярный профиль (сгущения/разрядки)",
            variable=self.var_grid_type,
            value="irregular",
            command=self._on_grid_type_change,
        )
        rb_irreg.pack(anchor=tk.W, padx=5, pady=2)

        btn_recalc = ttk.Button(grp_survey, text="Сгенерировать заново поле", command=self.calculate_field)
        btn_recalc.pack(fill=tk.X, **pad_opts)

        # --- Секция 3: Препроцессор (Сплайн Эрмита) ---
        grp_prep = ttk.LabelFrame(self.control_frame, text="3. Препроцессор входных данных")
        grp_prep.pack(fill=tk.X, pady=4)

        ttk.Label(
            grp_prep,
            text="Сглаживающий кубический сплайн Эрмита\n(интерполяция и фильтрация шума)",
            justify=tk.LEFT,
            font=("TkDefaultFont", 8),
        ).pack(anchor=tk.W, **pad_opts)

        btn_prep = ttk.Button(
            grp_prep, text="Применить сглаживающий сплайн", command=self.apply_preprocessor
        )
        btn_prep.pack(fill=tk.X, **pad_opts)

        # --- Секция 4: Инверсия ---
        grp_inv = ttk.LabelFrame(self.control_frame, text="4. Решение обратной задачи")
        grp_inv.pack(fill=tk.X, pady=4)

        # Настройки параметров регуляризации для классического метода
        reg_params_frame = ttk.Frame(grp_inv)
        reg_params_frame.pack(fill=tk.X, **pad_opts)

        ttk.Label(reg_params_frame, text="γ (gamma):").grid(row=0, column=0, sticky=tk.W)
        self.ent_gamma = ttk.Entry(reg_params_frame, width=8)
        self.ent_gamma.insert(0, "0.01")
        self.ent_gamma.grid(row=0, column=1, padx=4, pady=2)

        ttk.Label(reg_params_frame, text="α (alpha):").grid(row=0, column=2, sticky=tk.W, padx=4)
        self.ent_alpha = ttk.Entry(reg_params_frame, width=8)
        self.ent_alpha.insert(0, "0.0001")
        self.ent_alpha.grid(row=0, column=3, padx=4, pady=2)

        btn_solve_nn = ttk.Button(
            grp_inv,
            text="★ Решить с помощью ИНС (Нейросеть)",
            command=self.solve_nn,
            style="Accent.TButton",
        )
        btn_solve_nn.pack(fill=tk.X, padx=5, pady=4)

        btn_solve_reg = ttk.Button(
            grp_inv,
            text="Решить γ-регуляризацией",
            command=self.solve_classical,
        )
        btn_solve_reg.pack(fill=tk.X, **pad_opts)

        btn_solve_both = ttk.Button(
            grp_inv,
            text="Сравнить оба метода",
            command=self.solve_both,
        )
        btn_solve_both.pack(fill=tk.X, **pad_opts)

        # --- Секция 5: Обучение сети ---
        grp_train = ttk.LabelFrame(self.control_frame, text="5. Обучение нейросети")
        grp_train.pack(fill=tk.X, pady=4)

        btn_train_dialog = ttk.Button(
            grp_train, text="Открыть окно обучения ИНС...", command=self.open_train_dialog
        )
        btn_train_dialog.pack(fill=tk.X, **pad_opts)

        # --- Секция 6: Метрики и экспорт ---
        grp_metrics = ttk.LabelFrame(self.control_frame, text="6. Метрики и экспорт")
        grp_metrics.pack(fill=tk.X, pady=4)

        self.txt_metrics = tk.Text(grp_metrics, height=8, width=38, font=("Consolas", 8))
        self.txt_metrics.pack(fill=tk.BOTH, padx=5, pady=4)
        self.txt_metrics.insert(tk.END, "Нажмите 'Решить' для вычисления метрик.")

        btn_export = ttk.Button(
            grp_metrics, text="Экспорт результатов в CSV...", command=self.export_csv
        )
        btn_export.pack(fill=tk.X, **pad_opts)

    def _build_canvas(self, container):
        # 4 интерактивных графика:
        # [0, 0]: Истинная модель (2D heatmap)
        # [0, 1]: Восстановленная ИНС (2D heatmap)
        # [1, 0]: Восстановленная регуляризацией (2D heatmap)
        # [1, 1]: Профиль сигналов и невязок
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 8), dpi=100)
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=container)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, container)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Клик мыши по истинной модели для интерактивного редактирования ячеек
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

    def _on_canvas_click(self, event):
        """Интерактивное переключение ячеек истинной модели мышью на верхнем левом графике."""
        if event.inaxes != self.axes[0, 0]:
            return
        if event.xdata is None or event.ydata is None:
            return

        x = event.xdata
        z = event.ydata

        ix = int((x - self.survey.x_min) / self.survey.dx)
        # z отрицательна в глубину: z_top - (iz+0.5)*dz
        iz = int((self.survey.z_top - z) / self.survey.dz)

        if 0 <= ix < self.survey.nx and 0 <= iz < self.survey.nz:
            idx = self.survey.cell_index(ix, iz)
            # Переключение: 0 -> 1 -> 0
            self.true_model[idx] = 1.0 if self.true_model[idx] < 0.5 else 0.0
            self.calculate_field()

    def _on_model_selected(self, event):
        sel = self.combo_model.get()
        if "Тестовая модель" in sel:
            self.true_model = self.dataset_gen.get_benchmark_model("test_model")
        elif "среднего" in sel:
            self.true_model = self.dataset_gen.get_benchmark_model("medium")
        elif "Наклонный" in sel:
            self.true_model = self.dataset_gen.get_benchmark_model("tilted")
        elif "разделенных" in sel:
            self.true_model = self.dataset_gen.get_benchmark_model("two_separated")
        elif "перекрывающихся" in sel:
            self.true_model = self.dataset_gen.get_benchmark_model("two_stacked")
        elif "Случайная" in sel:
            self.true_model = self.dataset_gen.generate_balanced_model()
        elif "Очистить" in sel:
            self.true_model = np.zeros(self.survey.n_cells)

        self.calculate_field()

    def _on_noise_change(self, val):
        pct = int(float(val))
        self.lbl_noise_val.config(text=f"{pct}%")
        self.calculate_field()

    def _on_grid_type_change(self):
        self.calculate_field()

    def calculate_field(self):
        noise_level = float(self.scale_noise.get()) / 100.0

        if self.var_grid_type.get() == "regular":
            self.current_obs_x = self.survey.receiver_x.copy()
            clean_sig = self.survey.forward_solve(self.true_model)
        else:
            # 30 нерегулярных приемников
            np.random.seed(123)
            n_irreg = 30
            self.current_obs_x = np.sort(
                np.random.uniform(
                    self.survey.x_min,
                    self.survey.x_min + self.survey.nx * self.survey.dx,
                    n_irreg,
                )
            )
            rec_irreg = np.column_stack(
                [
                    self.current_obs_x,
                    np.full(n_irreg, self.survey.receiver_y),
                    np.full(n_irreg, self.survey.receiver_z),
                ]
            )
            L_irreg = self.survey.build_L(rec_irreg)
            clean_sig = self.survey.forward_solve(self.true_model, L=L_irreg)

        self.current_obs_signal = self.survey.add_noise(clean_sig, noise_level=noise_level)
        self.preprocessed_signal = None
        self.nn_predicted_model = None
        self.reg_predicted_model = None
        self.update_plots()

    def apply_preprocessor(self):
        """Применение сглаживающего сплайна Эрмита."""
        prep = SignalPreprocessor(
            target_x=self.survey.receiver_x,
            use_hermite=True,
            smoothing_factor=1e-3,
            n_elements=20,
        )
        self.preprocessed_signal = prep.process(
            self.current_obs_x, self.current_obs_signal
        )
        self.update_plots()

    def solve_nn(self):
        """Решение обратной задачи с помощью обученной нейросети."""
        sig_to_use = self.current_obs_signal
        if len(sig_to_use) != self.survey.n_receivers or self.preprocessed_signal is not None:
            if self.preprocessed_signal is None:
                self.apply_preprocessor()
            sig_to_use = self.preprocessed_signal

        with torch.no_grad():
            inp = torch.from_numpy(sig_to_use).float().unsqueeze(0)
            self.nn_predicted_model = self.nn_model(inp).squeeze(0).numpy()

        self._update_metrics_display()
        self.update_plots()

    def solve_classical(self):
        """Решение классической обратной задачи с регуляризацией."""
        sig_to_use = self.current_obs_signal
        if len(sig_to_use) != self.survey.n_receivers:
            if self.preprocessed_signal is None:
                self.apply_preprocessor()
            sig_to_use = self.preprocessed_signal

        try:
            gamma = float(self.ent_gamma.get())
        except ValueError:
            gamma = 0.01

        try:
            alpha = float(self.ent_alpha.get())
        except ValueError:
            alpha = 0.0001

        self.reg_predicted_model = self.classical_solver.solve(
            sig_to_use, gamma=gamma, alpha=alpha, auto_scale=True
        )
        self._update_metrics_display()
        self.update_plots()

    def solve_both(self):
        self.solve_nn()
        self.solve_classical()

    def _update_metrics_display(self):
        self.txt_metrics.delete("1.0", tk.END)
        lines = []

        sig_eval = (
            self.preprocessed_signal
            if self.preprocessed_signal is not None
            else self.current_obs_signal
        )

        if self.nn_predicted_model is not None:
            m_nn = self.classical_solver.compute_metrics(
                sig_eval, self.nn_predicted_model, self.true_model
            )
            lines.append("=== ИНС (Нейросеть) ===")
            lines.append(f"MAE модели:     {m_nn['model_mae']:.4f}")
            lines.append(f"MSE модели:     {m_nn['model_mse']:.4f}")
            lines.append(f"R_sq поля (%):  {m_nn['R_sq']:.2f}%")
            lines.append(f"||e|| невязки:  {m_nn['misfit_norm']:.2e}\n")

        if self.reg_predicted_model is not None:
            m_reg = self.classical_solver.compute_metrics(
                sig_eval, self.reg_predicted_model, self.true_model
            )
            lines.append("=== γ-регуляризация (Лаб. 1) ===")
            lines.append(f"MAE модели:     {m_reg['model_mae']:.4f}")
            lines.append(f"MSE модели:     {m_reg['model_mse']:.4f}")
            lines.append(f"R_sq поля (%):  {m_reg['R_sq']:.2f}%")
            lines.append(f"||e|| невязки:  {m_reg['misfit_norm']:.2e}\n")

        self.txt_metrics.insert(tk.END, "\n".join(lines))

    def update_plots(self):
        for row in self.axes:
            for ax in row:
                ax.clear()

        # 1. Истинная модель (ax[0, 0])
        self._draw_grid_on_ax(
            self.axes[0, 0], self.true_model, "Истинная модель среды (кликните для ред.)"
        )

        # 2. ИНС восстановление (ax[0, 1])
        if self.nn_predicted_model is not None:
            self._draw_grid_on_ax(
                self.axes[0, 1], self.nn_predicted_model, "Восстановлено ИНС"
            )
        else:
            self.axes[0, 1].text(
                0.5,
                0.5,
                "Нажмите 'Решить с помощью ИНС'",
                ha="center",
                va="center",
                transform=self.axes[0, 1].transAxes,
            )
            self.axes[0, 1].set_title("Восстановление ИНС")

        # 3. Классическая регуляризация (ax[1, 0])
        if self.reg_predicted_model is not None:
            self._draw_grid_on_ax(
                self.axes[1, 0], self.reg_predicted_model, "Восстановлено γ-регуляризацией"
            )
        else:
            self.axes[1, 0].text(
                0.5,
                0.5,
                "Нажмите 'Решить регуляризацией'",
                ha="center",
                va="center",
                transform=self.axes[1, 0].transAxes,
            )
            self.axes[1, 0].set_title("Классическая γ-регуляризация")

        # 4. Профиль сигналов (ax[1, 1])
        ax_sig = self.axes[1, 1]
        ax_sig.plot(
            self.current_obs_x,
            self.current_obs_signal,
            "k." if self.var_grid_type.get() == "irregular" else "k-",
            label="Наблюденный B_x",
            alpha=0.7,
        )

        if self.preprocessed_signal is not None:
            ax_sig.plot(
                self.survey.receiver_x,
                self.preprocessed_signal,
                "g--",
                linewidth=1.8,
                label="Сплайн (Препроцессор)",
            )

        if self.nn_predicted_model is not None:
            sig_nn = self.survey.forward_solve(self.nn_predicted_model)
            ax_sig.plot(
                self.survey.receiver_x, sig_nn, "r-", linewidth=1.8, label="Отклик ИНС"
            )

        if self.reg_predicted_model is not None:
            sig_reg = self.survey.forward_solve(self.reg_predicted_model)
            ax_sig.plot(
                self.survey.receiver_x, sig_reg, "b:", linewidth=1.8, label="Отклик γ-рег"
            )

        ax_sig.set_title("Сигналы B_x вдоль профиля", fontsize=10, fontweight="bold")
        ax_sig.set_xlabel("X (м)", fontsize=8)
        ax_sig.set_ylabel("Индукция B_x", fontsize=8)
        ax_sig.grid(True, linestyle=":", alpha=0.6)
        ax_sig.legend(fontsize=8, loc="upper right")

        self.fig.tight_layout()
        self.canvas.draw()

    def _draw_grid_on_ax(self, ax, vector_rho, title):
        grid = np.zeros((self.survey.nz, self.survey.nx))
        for ix in range(self.survey.nx):
            for iz in range(self.survey.nz):
                grid[iz, ix] = vector_rho[self.survey.cell_index(ix, iz)]

        extent = [
            self.survey.x_min,
            self.survey.x_min + self.survey.nx * self.survey.dx,
            self.survey.z_top - self.survey.nz * self.survey.dz,
            self.survey.z_top,
        ]
        max_val = float(np.max(grid)) if len(grid) > 0 else 0.0
        vmax = max(1.0, max_val) if max_val > 0.05 else 1.0
        im = ax.imshow(
            grid,
            extent=extent,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
        )
        display_title = f"{title} (макс: {max_val:.2f})" if max_val > 0.01 else title
        ax.set_title(display_title, fontsize=10, fontweight="bold")
        ax.set_xlabel("X (м)", fontsize=8)
        ax.set_ylabel("Z (м)", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.3)

    def export_csv(self):
        """Экспорт результатов в CSV-формат (как в alt/ signals.csv, model.csv)."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV files", "*.csv")]
        )
        if not file_path:
            return

        base, _ = os.path.splitext(file_path)
        # 1. Сигналы
        sig_file = f"{base}_signals.csv"
        sig_data = []
        sig_nn = (
            self.survey.forward_solve(self.nn_predicted_model)
            if self.nn_predicted_model is not None
            else np.zeros_like(self.survey.receiver_x)
        )
        sig_reg = (
            self.survey.forward_solve(self.reg_predicted_model)
            if self.reg_predicted_model is not None
            else np.zeros_like(self.survey.receiver_x)
        )

        with open(sig_file, "w", encoding="utf-8") as f:
            f.write("x,obs,preprocessed,calc_nn,calc_reg\n")
            for i in range(len(self.survey.receiver_x)):
                x = self.survey.receiver_x[i]
                prep = (
                    self.preprocessed_signal[i]
                    if self.preprocessed_signal is not None
                    else ""
                )
                f.write(f"{x},{self.current_obs_signal[i] if i < len(self.current_obs_signal) else ''},{prep},{sig_nn[i]},{sig_reg[i]}\n")

        # 2. Модель
        model_file = f"{base}_model.csv"
        with open(model_file, "w", encoding="utf-8") as f:
            f.write("ix,iz,x,z,lambda_true,lambda_nn,lambda_reg\n")
            for ix in range(self.survey.nx):
                for iz in range(self.survey.nz):
                    idx = self.survey.cell_index(ix, iz)
                    xc, zc = self.survey.cells[idx]
                    l_true = self.true_model[idx]
                    l_nn = (
                        self.nn_predicted_model[idx]
                        if self.nn_predicted_model is not None
                        else 0.0
                    )
                    l_reg = (
                        self.reg_predicted_model[idx]
                        if self.reg_predicted_model is not None
                        else 0.0
                    )
                    f.write(f"{ix},{iz},{xc},{zc},{l_true},{l_nn},{l_reg}\n")

        messagebox.showinfo(
            "Экспорт завершен",
            f"Файлы успешно сохранены:\n{sig_file}\n{model_file}",
        )

    def open_train_dialog(self):
        """Окно адаптивного обучения нейросети с графиком кривых потерь в реальном времени."""
        win = tk.Toplevel(self.root)
        win.title("Обучение искусственной нейронной сети")
        win.geometry("750x550")

        frame_top = ttk.Frame(win, padding="10")
        frame_top.pack(fill=tk.X)

        ttk.Label(frame_top, text="Размер датасета:").grid(row=0, column=0, sticky=tk.W)
        ent_samples = ttk.Entry(frame_top, width=8)
        ent_samples.insert(0, "2500")
        ent_samples.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(frame_top, text="Эпох:").grid(row=0, column=2, sticky=tk.W, padx=10)
        ent_epochs = ttk.Entry(frame_top, width=8)
        ent_epochs.insert(0, "60")
        ent_epochs.grid(row=0, column=3, padx=5, pady=2)

        ttk.Label(frame_top, text="Learning Rate:").grid(row=0, column=4, sticky=tk.W, padx=10)
        ent_lr = ttk.Entry(frame_top, width=8)
        ent_lr.insert(0, "0.002")
        ent_lr.grid(row=0, column=5, padx=5, pady=2)

        lbl_status = ttk.Label(frame_top, text="Статус: готов к обучению", font=("TkDefaultFont", 9, "bold"))
        lbl_status.grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=8)

        # Matplotlib canvas для живых кривых обучения
        fig_loss, ax_loss = plt.subplots(figsize=(7, 4), dpi=90)
        ax_loss.set_title("Кривые обучения (MSE Loss)")
        ax_loss.set_xlabel("Эпоха")
        ax_loss.set_ylabel("MSE (log scale)")
        ax_loss.grid(True, linestyle=":")
        canvas_loss = FigureCanvasTkAgg(fig_loss, master=win)
        canvas_loss.draw()
        canvas_loss.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        def on_train_close():
            try:
                plt.close(fig_loss)
            except Exception:
                pass
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_train_close)

        def start_train():
            try:
                n_samples = int(ent_samples.get())
                epochs = int(ent_epochs.get())
                lr = float(ent_lr.get())
            except ValueError:
                messagebox.showerror("Ошибка", "Проверьте правильность введенных чисел.")
                return

            btn_start.config(state=tk.DISABLED)
            lbl_status.config(text="Генерация датасета...")

            def worker():
                gen = DatasetGenerator(self.survey)
                # Генерация датасета
                X_tr, y_tr, X_val, y_val = gen.generate_dataset(
                    n_samples=n_samples, noise_level=0.01
                )

                lbl_status.config(text="Идет обучение нейросети...")
                trainer = ModelTrainer(self.nn_model, learning_rate=lr)

                ep_list, tr_list, val_list = [], [], []

                def on_epoch(ep, tot, tr_loss, val_loss, tr_mae, val_mae):
                    ep_list.append(ep)
                    tr_list.append(tr_loss)
                    val_list.append(val_loss)

                    if ep % 2 == 0 or ep == tot:
                        ax_loss.clear()
                        ax_loss.plot(ep_list, tr_list, "b-", label="Train Loss")
                        ax_loss.plot(ep_list, val_list, "r--", label="Val Loss")
                        ax_loss.set_yscale("log")
                        ax_loss.set_title(
                            f"Эпоха {ep}/{tot} | Val MSE: {val_loss:.5f} | Val MAE: {val_mae:.4f}"
                        )
                        ax_loss.set_xlabel("Эпоха")
                        ax_loss.set_ylabel("MSE (log)")
                        ax_loss.grid(True, linestyle=":")
                        ax_loss.legend()
                        canvas_loss.draw()

                trainer.train(
                    X_tr,
                    y_tr,
                    X_val,
                    y_val,
                    epochs=epochs,
                    save_path=self.weights_path,
                    callback=on_epoch,
                )

                lbl_status.config(text="Обучение завершено! Веса сохранены.")
                btn_start.config(state=tk.NORMAL)
                self.solve_nn()

            threading.Thread(target=worker, daemon=True).start()

        btn_start = ttk.Button(
            frame_top, text="Запустить обучение", command=start_train
        )
        btn_start.grid(row=1, column=4, columnspan=2, sticky=tk.E, pady=8)

    def on_close(self):
        """Полное и корректное завершение программы при закрытии окна."""
        try:
            plt.close("all")
        except Exception:
            pass
        try:
            self.root.quit()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass
        os._exit(0)


def main():
    root = tk.Tk()

    # Настройка стиля ttk
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    app = MagneticInversionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
