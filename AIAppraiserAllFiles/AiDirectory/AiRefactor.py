# pandas используется для работы с данными в формате таблиц (DataFrame).
# Dense и Input из Keras используются для создания слоев нейронной сети.
# Sequential из Keras используется для создания последовательной модели нейронной сети.
# KFold из sklearn.model_selection используется для кросс-валидации.
# StandardScaler из sklearn.preprocessing используется для нормализации данных.
# matplotlib.pyplot используется для построения графиков.
# scipy.stats используется для статистического анализа данных.
# L2 из Keras добавляет L2 регуляризацию к слоям модели.
# EarlyStopping из Keras используется для остановки обучения, если модель перестает улучшаться.
import pandas as pd
from keras.src.layers import Dense, Input
from keras.src.models import Sequential
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import scipy.stats as stats
from keras.src.regularizers import L2
from keras.src.callbacks import EarlyStopping
from keras import ops
import keras
import os
import joblib
#Определяет класс AI_Creait, который будет содержать методы для работы с данными и моделью.
class AI_Creait():

    # Конструктор загружает два CSV файла в DataFrame: df_training для обучения модели и df_check для проверки.
    def __init__(self, model_path=''):
        self.df_training = pd.read_csv('/home/I/SchoolProject/AIAppraiserAllFiles/ExecutableDirectory/first_DFFP.csv')
        self.df_check = pd.read_csv('/home/I/SchoolProject/AIAppraiserAllFiles/ExecutableDirectory/last_DFFP.csv')
        self.model = None
        self.scaler = None
        self.scaler_path = model_path.replace('.keras', '_scaler.pkl')

        if os.path.exists(model_path) and os.path.exists(model_path.replace('.keras', '_scaler.pkl')):
            print("Loading existing model and scaler...")
            self.model = keras.models.load_model(model_path)
            self.scaler = joblib.load(self.scaler_path)
        else:
            print("Model or scaler not found. Training new model...")

    # Разделяет данные на признаки X (столбцы Cup и Hero) и целевую переменную y (столбец Price).
    def df_division(self, df):
        X = df[['Cup', 'Hero']]
        y = df['Price']
        return X, y

    # n_splits=16: Указывает количество фолдов для кросс-валидации.
    def train_model_with_kfold(self, n_splits=30):
        if self.model is not None and self.scaler is not None:
            print("Model and scaler already loaded. Skipping training.")
            return

        print("Starting K-Fold training...")
        X, y = self.df_division(self.df_training)
        kf = (KFold                                 # KFold делит данные на n_splits частей для кросс-валидации.
              (n_splits=n_splits, shuffle=True,     # shuffle=True: Перемешивает данные перед разделением.
               random_state=42))                    # random_state=42: Фиксирует случайное состояние для воспроизводимости.
        fold_losses = []

        # Перебирает каждый фолд, разделяя данные на обучающую и тестовую выборки.
        for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
            print(f"Training fold {fold}...")
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # StandardScaler нормализует данные, чтобы они имели среднее значение 0 и стандартное отклонение 1.
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Создает последовательную модель с несколькими полносвязными слоями.
            # Input(shape=(X_train_scaled.shape[1],)): Определяет входной слой с количеством признаков, равным количеству столбцов в X_train_scaled.
            # Dense: Полносвязный слой с указанным количеством нейронов и функцией активации relu.
            # kernel_regularizer=L2(0.01): Добавляет L2 регуляризацию с коэффициентом 0.01 для предотвращения переобучения.
            self.model = Sequential([
                Input(shape=(X_train_scaled.shape[1],)),
                Dense(64, activation='relu', kernel_regularizer=L2(0.01)),
                Dense(32, activation='relu', kernel_regularizer=L2(0.01)),
                Dense(16, activation='relu', kernel_regularizer=L2(0.01)),
                Dense(8, activation='relu', kernel_regularizer=L2(0.01)),
                Dense(4, activation='relu', kernel_regularizer=L2(0.01)),
                Dense(1)
            ])

            # optimizer='adam': Использует оптимизатор Adam для обучения.
            # loss='mean_squared_error': Использует среднеквадратичную ошибку в качестве функции потерь.
            # EarlyStopping: Останавливает обучение, если функция потерь не улучшается в течение 10 эпох.
            # epochs=2550: Количество эпох обучения.
            # batch_size=32: Размер батча для обучения.
            # verbose=1: Выводит прогресс обучения в консоль.
            self.model.compile(optimizer='adam', loss='mean_squared_error')
            early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
            self.history = self.model.fit(X_train_scaled, y_train, epochs=2550, batch_size=32, verbose=1, callbacks=[early_stopping])

            # Оценивает модель на тестовых данных и сохраняет потери для каждого фолда.
            loss = self.model.evaluate(X_test_scaled, y_test, verbose=0)
            print(f"Fold {fold} loss: {loss}")
            fold_losses.append(loss)

        # Вычисляет среднее значение потерь по всем фолдам.
        # Сохраняет обученную модель в файл AIAppraiser.keras.
        average_loss = sum(fold_losses) / len(fold_losses)
        print(f'Average Loss across {n_splits} folds: {average_loss}')
        self.model.save('AIAppraiser.keras')
        joblib.dump(self.scaler, 'AIAppraiser_scaler.pkl')

        # Строит и сохраняет график потерь модели по эпохам.
        plt.plot(self.history.history['loss'])
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(f'{n_splits}-model_loss.png')
        plt.close()


    # Принимает значения cup и hero, нормализует их и возвращает предсказанное значение.
    def predict(self, cup, hero):
        if self.scaler is None or self.model is None:
            raise Exception("Model and scaler must be loaded or trained before prediction.")
        # Пример предсказания
        example_data = pd.DataFrame([[cup, hero]], columns=['Cup', 'Hero'])
        example_data_scaled = self.scaler.transform(example_data)
        predicted_value = self.model.predict(example_data_scaled)
        return predicted_value[0][0]


    # Строит и сохраняет гистограмму и Q-Q Plot для проверки нормальности распределения данных.
    def Q_Q_plot(self, data):
        # Гистограмма
        plt.hist(data, bins=10, alpha=0.6, color='g')
        plt.title(f'{data.name}-Histogram')
        plt.savefig(f'{data.name}-Histogram.png')
        plt.show()
        plt.close()

        # Q-Q Plot
        stats.probplot(data, dist="norm", plot=plt)
        plt.title(f'{data.name}-Q-Q Plot')
        plt.savefig(f'{data.name}-Q_Q_pot.png')
        plt.show()
        plt.close()

# Создает экземпляр класса AI_Creait и обучает модель.
if __name__ == '__main__':
    model = AI_Creait('AIAppraiser.keras')
    model.train_model_with_kfold()

    # Итерируется по каждой строке в df_check, делает предсказание и выводит его.
    for index, row in model.df_check.iterrows():
        cup = row['Cup']
        hero = row['Hero']
        prediction = model.predict(cup, hero)
        print(f'Предсказание для Cup: {cup}, Hero: {hero} - {prediction}')








