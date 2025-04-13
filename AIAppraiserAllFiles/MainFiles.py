from AiDirectory.NextLvlMain import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    """
    Главная функция программы с поддержкой нескольких языков.
    """
    # Инициализация модели
    config_path = r'C:\Users\user\PycharmProjects\SchoolProject\AIAppraiserAllFiles\AiDirectory\neat_config.txt'
    model_path = r'C:\Users\user\PycharmProjects\SchoolProject\AIAppraiserAllFiles\AiDirectory\AIAppraiser.neat'
    training_path = r'C:\Users\user\PycharmProjects\SchoolProject\AIAppraiserAllFiles\ExecutableDirectory\first_DFFP.csv'
    test_path = r'C:\Users\user\PycharmProjects\SchoolProject\AIAppraiserAllFiles\ExecutableDirectory\last_DFFP.csv'

    # Initialize model
    model = AI_Creait(
        config_file=config_path,
        model_path=model_path,
        training_path=training_path,
        test_path=test_path
    )

    # Train model if needed
    model.train_model_with_neat()

    # Настройка языков

    languages = ['English', 'Russian']
    language_index = 0  # По умолчанию английский (индекс 0)

    while True:
        # Отображение меню в зависимости от выбранного языка
        if language_index == 0:  # English
            print("\n=== AI Price Predictor ===")
            print("1. Manual Input Mode")
            print("2. File Analysis Mode")
            print("3. Evaluate Model Performance")
            print("4. Language")
            print("5. Quit")
            choice = input("Enter your choice (1-5): ").strip()

            if choice == '1':
                manual_input(model)
            elif choice == '2':
                file_path = input("Enter path to CSV file: ").strip()
                if not file_path:
                    file_path = r'C:\Users\user\PycharmProjects\SchoolProject\AIAppraiserAllFiles\ExecutableDirectory\DataParsFunPay.csv'
                create_graph(model, file_path)
            elif choice == '3':
                model.evaluate_model()
            elif choice == '4':
                print("\n=== Select a language ===")
                for i, lang in enumerate(languages):
                    print(f"{i + 1}. {lang}")

                try:
                    selection = int(input(f"Enter your choice (1-{len(languages)}): ").strip())
                    if 1 <= selection <= len(languages):
                        language_index = selection - 1
                    else:
                        print("Invalid selection. Please try again.")
                except ValueError:
                    print("Please enter a number.")
                continue
            elif choice == '5':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")

        elif language_index == 1:  # Russian
            print("\n=== AI Предсказатель Цен ===")
            print("1. Режим ручного ввода")
            print("2. Режим анализа файлов")
            print("3. Оценка производительности модели")
            print("4. Язык")
            print("5. Выйти")
            choice = input("Введите свой выбор (1-5): ").strip()

            if choice == '1':
                manual_input(model)
            elif choice == '2':
                file_path = input("Введите путь к CSV-файлу: ").strip()
                if not file_path:
                    file_path = r'C:\Users\user\PycharmProjects\SchoolProject\AIAppraiserAllFiles\ExecutableDirectory\DataParsFunPay.csv'
                create_graph(model, file_path)
            elif choice == '3':
                model.evaluate_model()
            elif choice == '4':
                print("\n=== Выберите язык ===")
                for i, lang in enumerate(languages):
                    print(f"{i + 1}. {lang}")

                try:
                    selection = int(input(f"Введите свой выбор (1-{len(languages)}): ").strip())
                    if 1 <= selection <= len(languages):
                        language_index = selection - 1
                    else:
                        print("Неверный выбор. Пожалуйста, попробуйте снова.")
                except ValueError:
                    print("Пожалуйста, введите число.")
                continue
            elif choice == '5':
                print("До свидания!")
                break
            else:
                print("Неверный выбор. Пожалуйста, попробуйте снова.")

if __name__ == "__main__":
    main()
