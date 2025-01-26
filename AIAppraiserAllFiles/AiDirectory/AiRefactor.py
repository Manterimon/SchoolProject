import pandas as pd
from keras.src.layers import Dense
from keras.src.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import scipy.stats as stats
from keras.src.regularizers import L2
from tensorflow.python.framework.test_ops import kernel_label_required


class AI_Creait():
    def __init__(self):
        self.df_training = pd.read_csv('/AIAppraiserAllFiles/ExecutableDirectory/first_DFFP.csv')
        self.df_check = pd.read_csv('/AIAppraiserAllFiles/ExecutableDirectory/last_DFFP.csv')

    def df_division(self, df, test_size=0.2, random_state=42):
        X = df[['Cup', 'Hero']]
        y = df['Price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

    def train_model(self):
        # Разделение данных
        X_train, X_test, y_train, y_test = self.df_division(self.df_training)

        # Нормализация данных
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Создание модели
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],), kernel_regularizer=L2(0.01)),
            Dense(32, activation='relu', kernel_regularizer=L2(0.01)),
            Dense(1)
        ])

        # Компиляция модели
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Обучение модели
        model.fit(X_train_scaled, y_train, epochs=100, batch_size=10, verbose=1)

        # Оценка модели
        loss = model.evaluate(X_test_scaled, y_test, verbose=0)
        print(f'Loss: {loss}')

        # Сохранение модели и скейлера для дальнейшего использования
        self.model = model
        self.scaler = scaler

    def predict(self, cup, hero):
        # Пример предсказания
        example_data = pd.DataFrame([[cup, hero]], columns=['Cup', 'Hero'])
        example_data_scaled = self.scaler.transform(example_data)
        predicted_value = self.model.predict(example_data_scaled)
        return predicted_value[0][0]

    def Q_Q_plot(self, data):#провека нормального расспределения данных
        # Гистограмма
        plt.hist(data, bins=10, alpha=0.6, color='g')
        plt.title('Histogram')
        plt.show()

        # Q-Q Plot
        stats.probplot(data, dist="norm", plot=plt)
        plt.title('Q-Q Plot')
        plt.show()

if __name__ == '__main__':
    model = AI_Creait()
    model.Q_Q_plot(model.df_training['Hero'])
    model.train_model()

    # Итерируемся по каждой строке в DataFrame df_check
    for index, row in model.df_check.iterrows():
        cup = row['Cup']
        hero = row['Hero']
        prediction = model.predict(cup, hero)
        print(f'Предсказание для Cup: {cup}, Hero: {hero} - {prediction}')









