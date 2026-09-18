"""
Обратная задача магниторазведки с использованием аппарата искусственных нейронных сетей

Использование:
  python main.py             - Запуск графического интерфейса (GUI)
  python main.py --gui       - Запуск графического интерфейса (GUI)
  python main.py --exp       - Автоматический запуск экспериментов и сохранение графиков
  python main.py --train     - Обучение нейронной сети и сохранение весов
  python main.py --test      - Запуск проверки модулей и тестов
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Решение обратной задачи магниторазведки с использованием ИНС"
    )
    parser.add_argument(
        "--gui", action="store_true", help="Запустить графический интерфейс (по умолчанию)"
    )
    parser.add_argument(
        "--exp", action="store_true", help="Запустить эксперименты и сохранить графики"
    )
    parser.add_argument(
        "--train", action="store_true", help="Запустить генерацию датасета и обучение ИНС"
    )
    parser.add_argument(
        "--test", action="store_true", help="Запустить самотестирование всех модулей"
    )

    args = parser.parse_args()

    if args.exp:
        from run_experiments import run_all_experiments
        run_all_experiments()
    elif args.train:
        from train import run_full_training
        run_full_training()
    elif args.test:
        print("=== ТЕСТИРОВАНИЕ МОДУЛЕЙ ===")
        import forward_problem
        import preprocessor
        import dataset
        import neural_net
        import classical_inversion
        print(" Все модули успешно импортированы и готовы к работе.")
    else:
        # По умолчанию запускаем GUI
        import gui
        gui.main()


if __name__ == "__main__":
    main()
