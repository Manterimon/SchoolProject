import pandas as pd
from keras.src.layers import Dense, Input
from keras.src.models import Sequential
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import scipy.stats as stats
from keras.src.regularizers import L2
from keras.src.callbacks import EarlyStopping
from tensorflow.python.framework.test_ops import kernel_label_required


class AI_Creait():
    def __init__(self):
        self.df_training = pd.read_csv('/home/I/SchoolProject/AIAppraiserAllFiles/ExecutableDirectory/first_DFFP.csv')
        self.df_check = pd.read_csv('/home/I/SchoolProject/AIAppraiserAllFiles/ExecutableDirectory/last_DFFP.csv')

    def df_division(self, df):
        X = df[['Cup', 'Hero']]
        y = df['Price']
        return X, y

    def train_model_with_kfold(self, n_splits=16):
        print("Starting K-Fold training...")
        X, y = self.df_division(self.df_training)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_losses = []

        for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
            print(f"Training fold {fold}...")
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Нормализация данных
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Создание модели
            self.model = Sequential([
                Input(shape=(X_train_scaled.shape[1],)),
                Dense(64, activation='relu', kernel_regularizer=L2(0.01)),
                Dense(32, activation='relu', kernel_regularizer=L2(0.01)),
                Dense(16, activation='relu', kernel_regularizer=L2(0.01)),
                Dense(8, activation='relu', kernel_regularizer=L2(0.01)),
                Dense(4, activation='relu', kernel_regularizer=L2(0.01)),
                Dense(1)
            ])

            # Компиляция модели
            self.model.compile(optimizer='adam', loss='mean_squared_error')
            early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
            # Обучение модели
            self.history = self.model.fit(X_train_scaled, y_train, epochs=2550, batch_size=32, verbose=1, callbacks=[early_stopping])

            # Оценка модели
            loss = self.model.evaluate(X_test_scaled, y_test, verbose=0)
            print(f"Fold {fold} loss: {loss}")
            fold_losses.append(loss)


        average_loss = sum(fold_losses) / len(fold_losses)
        print(f'Average Loss across {n_splits} folds: {average_loss}')

        self.model.save('AIAppraiser.keras')

        plt.plot(self.history.history['loss'])
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(f'{n_splits}-model_loss.png')
        plt.close()

    def predict(self, cup, hero):
        # Пример предсказания
        example_data = pd.DataFrame([[cup, hero]], columns=['Cup', 'Hero'])
        example_data_scaled = self.scaler.transform(example_data)
        predicted_value = self.model.predict(example_data_scaled)
        return predicted_value[0][0]

    def Q_Q_plot(self, data):#провека нормального расспределения данных
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

if __name__ == '__main__':
    model = AI_Creait()
    model.train_model_with_kfold()

    # Итерируемся по каждой строке в DataFrame df_check
    for index, row in model.df_check.iterrows():
        cup = row['Cup']
        hero = row['Hero']
        prediction = model.predict(cup, hero)
        print(f'Предсказание для Cup: {cup}, Hero: {hero} - {prediction}')








