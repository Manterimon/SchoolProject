from AiDirectory import AiRefactor
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # Загрузка модели
    model = AiRefactor.AI_Creait('/home/I/SchoolProject/AIAppraiserAllFiles/AiDirectory/AIAppraiser.keras')
    model.train_model_with_kfold()

    # Загрузка данных из CSV
    full_df_check = pd.read_csv('/home/I/SchoolProject/AIAppraiserAllFiles/ExecutableDirectory/DataParsFunPay.csv')

    # Создаем пустой список для хранения предсказанных цен
    predicted_prices = []

    # Итерируемся по DataFrame и делаем предсказания
    for index, row in full_df_check.iterrows():
        # Получаем предсказание цены для текущей строки
        predicted_price = model.predict(row['Cup'], row['Hero'])
        # Добавляем предсказанную цену в список
        predicted_prices.append(predicted_price)

    # Добавляем предсказанные значения в DataFrame
    full_df_check['PredictedPrice'] = predicted_prices

    # Визуализация
    y_true = full_df_check['Price']
    y_pred = full_df_check['PredictedPrice']
    x = np.arange(len(full_df_check))  # Создаем массив индексов для оси X

    # Создаем график
    plt.figure(figsize=(12, 6))

    # Рисуем точки для фактических цен
    plt.scatter(x, y_true, label='Actual Price', color='blue', s=5)  # s - размер точек

    # Рисуем точки для предсказанных цен
    plt.scatter(x, y_pred, label='Predicted Price', color='red', s=5, marker='x')  # s - размер точек, marker - форма маркера

    # Настраиваем график
    plt.xlabel("Account Index")
    plt.ylabel("Price")
    plt.title("Actual vs Predicted Prices")
    plt.xlim(0, len(full_df_check))  # Устанавливаем границы оси X
    plt.ylim(
        1,
              30000)
    print(
        f'Min, Max: {min(full_df_check["Price"].min(axis=0), full_df_check["PredictedPrice"].min(axis=0))} {max(full_df_check["Price"].max(axis=0), full_df_check["PredictedPrice"].max(axis=0))}')# Устанавливаем границы оси Y
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # while True:
    #     print('Для того что бы остановить программу введите два раза 0')
    #     try:
    #         cup, hero = map(int, input('Введите сначала количество кубков аккаунта, а затем количество персонажей.\n').split(', '))
    #     except Exception:
    #         print('Что-то пошло не так! Попробуйте еще раз!')
    #         continue
    #     prediction = Ai.predict(cup, hero)
    #     print(f'Предсказание для Cup: {cup}, Hero: {hero} - {prediction}')
    #     if cup == 0:
    #         break