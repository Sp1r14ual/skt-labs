import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
import torch

from gravity_core import GravitySolver
from model import GravityNN

# Настройка внешнего вида CustomTkinter
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class GravityInversionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Гравиразведка")
        self.geometry("1400x880")
        self.minsize(1100, 750)

        # Ядро расчетов
        self.solver = GravitySolver()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.load_neural_network()

        # Данные текущей сессии
        self.true_rho = self.solver.get_standard_test_model()
        self.g_obs = None
        self.g_true = None
        self.classic_rho = None
        self.nn_rho = None

        self._build_ui()
        self.on_compute_forward()
        self.on_run_both()

        # Полное завершение программы при закрытии окна по крестику
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        try:
            plt.close("all")
            self.quit()
            self.destroy()
        except Exception:
            pass
        finally:
            os._exit(0)

    def load_neural_network(self):
        weights_path = os.path.join(os.path.dirname(__file__), "gravity_nn.pth")
        if os.path.exists(weights_path):
            try:
                self.model = GravityNN(in_features=self.solver.n_receivers, out_features=self.solver.n_cells).to(self.device)
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
                self.model.eval()
            except Exception as e:
                print(f"Ошибка загрузки весов: {e}")
                self.model = None

    def _build_ui(self):
        # Главная сетка: левая колонка (настройки), правая (графики)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ----------------- ЛЕВАЯ ПАНЕЛЬ (Управление) -----------------
        self.sidebar = ctk.CTkScrollableFrame(self, width=340, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Заголовок
        lbl_title = ctk.CTkLabel(self.sidebar, text="ГРАВИРАЗВЕДКА", font=ctk.CTkFont(size=20, weight="bold"))
        lbl_title.pack(padx=10, pady=(10, 5))
        lbl_sub = ctk.CTkLabel(self.sidebar, text="Решение обратной задачи гравиразведки", font=ctk.CTkFont(size=12), text_color="gray")
        lbl_sub.pack(padx=10, pady=(0, 15))

        # Блок выбора модели
        sec_model = ctk.CTkFrame(self.sidebar)
        sec_model.pack(fill="x", padx=5, pady=8)
        ctk.CTkLabel(sec_model, text="1. Выбор геометрии среды", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=10, pady=5)

        self.model_preset_var = ctk.StringVar(value="Стандартная (ЛР №1)")
        preset_menu = ctk.CTkOptionMenu(
            sec_model,
            variable=self.model_preset_var,
            values=["Стандартная (ЛР №1)", "Случайная (1 объект)", "Случайная (2-3 объекта)", "Наклонный пласт"],
            command=self.on_preset_change
        )
        preset_menu.pack(fill="x", padx=10, pady=(0, 8))

        btn_regen = ctk.CTkButton(sec_model, text="Сгенерировать заново", command=self.on_regen_model)
        btn_regen.pack(fill="x", padx=10, pady=(0, 10))

        # Блок параметров наблюдения (шум)
        sec_obs = ctk.CTkFrame(self.sidebar)
        sec_obs.pack(fill="x", padx=5, pady=8)
        ctk.CTkLabel(sec_obs, text="2. Система измерений и шум", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=10, pady=5)

        self.lbl_noise = ctk.CTkLabel(sec_obs, text="Уровень шума: 1.0%")
        self.lbl_noise.pack(anchor="w", padx=10, pady=(0, 2))
        self.slider_noise = ctk.CTkSlider(sec_obs, from_=0.0, to=5.0, number_of_steps=50, command=self._on_noise_slider)
        self.slider_noise.set(1.0)
        self.slider_noise.pack(fill="x", padx=10, pady=(0, 10))

        # Блок параметров классической инверсии
        sec_inv = ctk.CTkFrame(self.sidebar)
        sec_inv.pack(fill="x", padx=5, pady=8)
        ctk.CTkLabel(sec_inv, text="3. Классическая регуляризация", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=10, pady=5)

        self.lbl_gamma = ctk.CTkLabel(sec_inv, text="Параметр γ")
        self.lbl_gamma.pack(anchor="w", padx=10, pady=(0, 2))
        self.gamma_var = ctk.StringVar(value="0.1")
        gamma_entry = ctk.CTkEntry(sec_inv, textvariable=self.gamma_var)
        gamma_entry.pack(fill="x", padx=10, pady=(0, 10))

        # Кнопки действий
        sec_actions = ctk.CTkFrame(self.sidebar)
        sec_actions.pack(fill="x", padx=5, pady=8)
        ctk.CTkLabel(sec_actions, text="4. Выполнение инверсии", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=10, pady=5)

        btn_run_both = ctk.CTkButton(
            sec_actions,
            text="▶ Запустить сравнение методов",
            fg_color="#1f538d",
            hover_color="#14375e",
            height=38,
            font=ctk.CTkFont(weight="bold"),
            command=self.on_run_both
        )
        btn_run_both.pack(fill="x", padx=10, pady=(8, 6))

        btn_run_classic = ctk.CTkButton(sec_actions, text="Решить: Классический метод (γ-регуляризация)", command=self.on_run_classic)
        btn_run_classic.pack(fill="x", padx=10, pady=4)

        btn_run_nn = ctk.CTkButton(sec_actions, text="Решить: Нейронная сеть", fg_color="#2b7a4b", hover_color="#1d5533", command=self.on_run_nn)
        btn_run_nn.pack(fill="x", padx=10, pady=(4, 10))

        # Блок метрик и статистики
        sec_metrics = ctk.CTkFrame(self.sidebar)
        sec_metrics.pack(fill="x", padx=5, pady=8)
        ctk.CTkLabel(sec_metrics, text="Сводные метрики", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=10, pady=5)

        self.txt_metrics = ctk.CTkTextbox(sec_metrics, height=130, font=ctk.CTkFont(family="Consolas", size=11))
        self.txt_metrics.pack(fill="x", padx=10, pady=(0, 10))

        # ----------------- ПРАВАЯ ПАНЕЛЬ (Графики) -----------------
        self.plots_frame = ctk.CTkFrame(self)
        self.plots_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.plots_frame.grid_columnconfigure(0, weight=1)
        self.plots_frame.grid_rowconfigure(0, weight=1)

        # Matplotlib Figure
        plt.style.use("dark_background")
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 7.5), facecolor="#242424")
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plots_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = ctk.CTkFrame(self.plots_frame, height=35)
        toolbar_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

    def _on_noise_slider(self, val):
        self.lbl_noise.configure(text=f"Уровень шума: {val:.1f}%")

    def on_preset_change(self, choice):
        if choice == "Стандартная (ЛР №1)":
            self.true_rho = self.solver.get_standard_test_model()
        elif choice == "Наклонный пласт":
            rho = np.zeros((self.solver.nx, self.solver.nz))
            for i in range(8):
                rho[12 + i, 4 + i // 2 : 7 + i // 2] = 1.0
            self.true_rho = rho.flatten()
        elif choice == "Случайная (1 объект)":
            self.true_rho = self.solver.generate_random_model(max_anomalies=1)
        else:
            self.true_rho = self.solver.generate_random_model(max_anomalies=3)

        self.on_compute_forward()
        self.on_run_both()

    def on_regen_model(self):
        self.on_preset_change(self.model_preset_var.get())

    def on_compute_forward(self):
        self.g_true = self.solver.forward(self.true_rho)
        noise_level = self.slider_noise.get() / 100.0
        scale = np.max(np.abs(self.g_true)) if np.max(np.abs(self.g_true)) > 0 else 1.0
        noise = noise_level * scale * np.random.randn(len(self.g_true))
        self.g_obs = self.g_true + noise

    def on_run_classic(self):
        if self.g_obs is None:
            self.on_compute_forward()

        try:
            gamma = float(self.gamma_var.get())
        except ValueError:
            gamma = 0.1

        t0 = time.time()
        self.classic_rho = self.solver.solve_classical_inverse(self.g_obs, gamma=gamma)
        t_classic = (time.time() - t0) * 1000

        self.update_plots()
        self._update_metrics(time_classic=t_classic)

    def on_run_nn(self):
        if self.model is None:
            messagebox.showwarning("Внимание", "Модель нейросети не загружена (файл gravity_nn.pth не найден).")
            return
        if self.g_obs is None:
            self.on_compute_forward()

        t0 = time.time()
        with torch.no_grad():
            x_t = torch.from_numpy(self.g_obs.astype(np.float32)).unsqueeze(0).to(self.device)
            pred = self.model(x_t)
            self.nn_rho = pred.squeeze(0).cpu().numpy()
        t_nn = (time.time() - t0) * 1000

        self.update_plots()
        self._update_metrics(time_nn=t_nn)

    def on_run_both(self):
        if self.g_obs is None:
            self.on_compute_forward()

        # Классика
        try:
            gamma = float(self.gamma_var.get())
        except ValueError:
            gamma = 0.1

        t0 = time.time()
        self.classic_rho = self.solver.solve_classical_inverse(self.g_obs, gamma=gamma)
        t_classic = (time.time() - t0) * 1000

        # Нейросеть
        t_nn = 0.0
        if self.model is not None:
            t0 = time.time()
            with torch.no_grad():
                x_t = torch.from_numpy(self.g_obs.astype(np.float32)).unsqueeze(0).to(self.device)
                pred = self.model(x_t)
                self.nn_rho = pred.squeeze(0).cpu().numpy()
            t_nn = (time.time() - t0) * 1000

        self.update_plots()
        self._update_metrics(time_classic=t_classic, time_nn=t_nn)

    def _update_metrics(self, time_classic=None, time_nn=None):
        self.txt_metrics.delete("1.0", tk.END)
        lines = []

        if self.classic_rho is not None and time_classic is not None:
            g_c = self.solver.forward(self.classic_rho)
            misfit_c = np.linalg.norm(self.g_obs - g_c)
            mse_c = np.mean((self.true_rho - self.classic_rho)**2)
            lines.append("--- КЛАССИКА (γ-регуляриз.) ---")
            lines.append(f" Время:    {time_classic:.2f} мс")
            lines.append(f" Невязка:  {misfit_c:.4f}")
            lines.append(f" MSE ρ:    {mse_c:.5f}\n")

        if self.nn_rho is not None and time_nn is not None:
            g_n = self.solver.forward(self.nn_rho)
            misfit_n = np.linalg.norm(self.g_obs - g_n)
            mse_n = np.mean((self.true_rho - self.nn_rho)**2)
            lines.append("--- НЕЙРОСЕТЬ (PyTorch) ---")
            lines.append(f" Время:    {time_nn:.2f} мс")
            lines.append(f" Невязка:  {misfit_n:.4f}")
            lines.append(f" MSE ρ:    {mse_n:.5f}")
            if time_classic and time_nn > 0:
                speedup = time_classic / time_nn
                lines.append(f" Ускорение:{speedup:.1f}x быстрее!")

        self.txt_metrics.insert(tk.END, "\n".join(lines))

    def update_plots(self):
        for ax in self.axes.flatten():
            ax.clear()

        rec_x = self.solver.receivers[:, 0]

        # 1. Поле на профиле
        ax0 = self.axes[0, 0]
        if self.g_obs is not None:
            ax0.plot(rec_x, self.g_obs, color="#00ffcc", lw=1.5, label="Наблюденные (g_obs)")
        if self.classic_rho is not None:
            g_c = self.solver.forward(self.classic_rho)
            ax0.plot(rec_x, g_c, color="#ffaa00", linestyle="--", lw=1.5, label="Классика")
        if self.nn_rho is not None:
            g_n = self.solver.forward(self.nn_rho)
            ax0.plot(rec_x, g_n, color="#ff3366", linestyle=":", lw=2.0, label="Нейросеть")
        ax0.set_title("Поле Δg вдоль профиля", fontsize=11, fontweight="bold")
        ax0.set_xlabel("X (м)")
        ax0.set_ylabel("Δg")
        ax0.grid(True, linestyle=":", alpha=0.5)
        ax0.legend(loc="upper right", fontsize=8)

        # 2. Истинная модель
        self._plot_density(self.axes[0, 1], self.true_rho, "Истинная модель среды (True ρ)")

        # 3. Классическая модель
        if self.classic_rho is not None:
            self._plot_density(self.axes[1, 0], self.classic_rho, "Классическая инверсия (γ)")
        else:
            self.axes[1, 0].set_title("Классическая инверсия (не рассчитана)")

        # 4. Нейросетевая модель
        if self.nn_rho is not None:
            self._plot_density(self.axes[1, 1], self.nn_rho, "Восстановление нейросетью")
        else:
            self.axes[1, 1].set_title("Нейросеть (не рассчитана)")

        self.fig.tight_layout()
        self.canvas.draw()

    def _plot_density(self, ax, rho_vec, title):
        mat = rho_vec.reshape((self.solver.nx, self.solver.nz)).T
        im = ax.imshow(
            mat,
            origin='lower',
            aspect='auto',
            cmap='viridis',
            extent=[0, self.solver.nx * self.solver.dx, -self.solver.nz * self.solver.dz, 0]
        )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("X")
        ax.set_ylabel("Z (глубина)")


if __name__ == "__main__":
    app = GravityInversionApp()
    app.mainloop()
