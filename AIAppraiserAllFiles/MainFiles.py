from AiDirectory.NextLvlMain import *


def main():
    """
    Главная функция программы с поддержкой нескольких языков.
    """
    # Получаем абсолютный путь к текущей директории скрипта
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Формируем абсолютные пути к файлам конфигурации и данных, используя os.path.join
    config_path = os.path.join(script_dir,  "AiDirectory", 'neat_config.txt')
    model_path = os.path.join(script_dir,  "AiDirectory",  'AIAppraiser.neat')
    training_path = os.path.join(script_dir,  'ExecutableDirectory', 'first_DFFP.csv')
    test_path = os.path.join(script_dir,   "ExecutableDirectory", 'last_DFFP.csv')

    # Initialize model

    model = AI_Creait(
        config_file=config_path,
        model_path=model_path,
        training_path=training_path,
        test_path=test_path
    )
    languages = ['en', 'ru']
    language_index = 0
    # Train model if needed
    model.train_model_with_neat()

    # Настройка языков
      # По умолчанию английский (индекс 0)

    while True:
        # Отображение меню в зависимости от выбранного языка
        if language_index == 0:  # English
            print("\n=== AI Price Predictor ===")
            print("1. Manual Input Mode")
            print("2. File Analysis Mode")
            print("3. Evaluate Model Performance")
            print("4. Manual analytical chart")
            print("5. Language")
            print("6. Quit")
            choice = input("Enter your choice (1-6): ").strip()

            if choice == '1':
                manual_input(model)
            elif choice == '2':
                file_path = input("Enter path to CSV file: ").strip()
                if not file_path:
                    file_path = os.path.join(script_dir,  "ExecutableDirectory", 'DataParsFunPay.csv')
                create_graph(model, file_path)
            elif choice == '3':
                model.evaluate_model()
            elif choice == '4':
                while True:
                    print("\n=== Price Prediction Console ===")
                    print('Enter Cup and Hero, Price values separated by comma (e.g., 1000, 25, 400)')
                    print('Enter "q" to quit')
                    user_input = input('Input: ').strip()
                    if user_input.lower() == 'q':
                        break
                    try:
                        values = [int(x.strip()) for x in user_input.split(',')]
                        if len(values) != 3:
                            raise ValueError("Need exactly three values")

                        cup, hero, price = values
                        prediction = model.predict(cup, hero)
                        create_graph_local(price,prediction)
                    except ValueError as e:
                        print(f"Invalid input: {e}. Please try again.")
                    except Exception as e:
                        print(f"Error: {e}")

            elif choice == '5':
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
            elif choice == '6':
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
                    file_path = os.path.join(script_dir, "ExecutableDirectory", 'DataParsFunPay.csv')
                create_graph(model, file_path)
            elif choice == '3':
                model.evaluate_model()
            elif choice == '4':
                while True:
                    print("\n=== Консоль прогнозирования цен ===")
                    print('Введите значения кубков и количесто героев и вашу цену разделяя их запятой (e.g., 1000, 25, 400)')
                    print('Введите "q", чтобы выйти')
                    user_input = input('Input: ').strip()
                    if user_input.lower() == 'q':
                        break
                    try:
                        values = [int(x.strip()) for x in user_input.split(',')]
                        if len(values) != 3:
                            raise ValueError("Нужны ровно три значения")

                        cup, hero, price = values
                        prediction = model.predict(cup, hero)
                        create_graph_local(price, prediction)
                    except ValueError as e:
                        print(f"Ошибка ввода: {e}. Пожалуйста, повторите.")
                    except Exception as e:
                        print(f"Ошибка: {e}")
                continue
            elif choice == '5':
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
            elif choice == '6':
                print("До свидания!")
                break
            else:
                print("Неверный выбор. Пожалуйста, попробуйте снова.")

if __name__ == "__main__":
    main()
