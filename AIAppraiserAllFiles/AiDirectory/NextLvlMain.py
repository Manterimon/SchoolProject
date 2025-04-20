import pandas as pd
import neat
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import joblib
import math
from typing import Tuple, Union, List, Optional, Dict, Any
import goslate

# Activation and aggregation functions
def my_sinc_function(x):
    """Custom sinc activation function."""
    if x == 0:
        return 1.0
    else:
        return math.sin(x) / x


def my_l2norm_function(x):
    """Custom L2 norm aggregation function."""
    return np.sqrt(np.sum(np.square(x)))


class AI_Creait:
    """
    Neural network model using NEAT for predicting prices based on Cup and Hero values.
    """

    def __init__(self, config_file: str, model_path: str = '',
                 training_path: str = None, test_path: str = None):
        """
        Initialize the AI model.

        Args:
            config_file: Path to NEAT configuration file
            model_path: Path to saved model file (optional)
            training_path: Path to training CSV file (optional)
            test_path: Path to test CSV file (optional)
        """
        # Load data if paths provided
        self.df_training = None
        self.df_check = None
        if training_path:
            self.df_training = pd.read_csv(training_path)
        if test_path:
            self.df_check = pd.read_csv(test_path)

        # Load NEAT configuration
        self.config_file = config_file
        self.config = self._load_config(config_file)

        # Initialize model and scalers
        self.model, self.scaler, self.price_scaler = self._load_or_init_model(model_path)

    def _load_config(self, config_file: str) -> neat.Config:
        """Load NEAT configuration with custom functions."""
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_file
        )

        # Register custom functions
        config.genome_config.activation_defs.add("my_sinc_function", my_sinc_function)
        config.genome_config.aggregation_function_defs.add("my_l2norm_function", my_l2norm_function)

        return config

    def _load_or_init_model(self, model_path: str) -> Tuple[Any, Any, Any]:
        """Load existing model and scalers or initialize new ones."""
        if model_path and os.path.exists(model_path):
            scaler_path = model_path.replace('.neat', '_scaler.pkl')
            price_scaler_path = model_path.replace('.neat', '_price_scaler.pkl')

            if all(os.path.exists(p) for p in [model_path, scaler_path, price_scaler_path]):
                print("Loading existing model and scalers...")
                return (
                    joblib.load(model_path),
                    joblib.load(scaler_path),
                    joblib.load(price_scaler_path)
                )

        print("Model or scalers not found. Will train new model...")
        return None, None, None

    def df_division(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract features and target from DataFrame with validation."""
        if not all(col in df.columns for col in ['Cup', 'Hero', 'Price']):
            raise ValueError("DataFrame must contain 'Cup', 'Hero', and 'Price' columns")

        X = df[['Cup', 'Hero']]
        y = df['Price']
        return X, y

    def eval_genome(self, genome, config, X_scaled, y_scaled) -> float:
        """Evaluate a single genome on the provided data."""
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        # Make predictions
        predictions = []
        for xi in X_scaled:
            output = net.activate(xi)
            predictions.append(output[0])

        predictions = np.array(predictions)
        fitness = np.mean((predictions - y_scaled) ** 2)  # MSE as fitness
        return -fitness  # NEAT maximizes fitness, so minimize MSE

    def train_model_with_neat(self, training_data=None, n_splits=10, generations=300,
                              save_path=None) -> float:
        """
        Train model with NEAT using k-fold cross-validation.

        Args:
            training_data: DataFrame to use for training (uses self.df_training if None)
            n_splits: Number of folds for cross-validation
            generations: Number of generations for each NEAT run
            save_path: Path to save the trained model

        Returns:
            average_fitness: The average fitness across all folds
        """
        if self.model is not None and self.scaler is not None and self.price_scaler is not None:
            print("Model and scalers already loaded. Skipping training.")
            return 0.0

        df = training_data if training_data is not None else self.df_training
        if df is None:
            raise ValueError("No training data provided. Either pass training_data or initialize with training_path")

        print("Starting NEAT training...")
        X, y = self.df_division(df)

        # Initialize scalers
        self.scaler = StandardScaler()
        self.price_scaler = MinMaxScaler()

        # Scale features and target
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.price_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

        # K-fold cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_fitnesses = []
        best_genome = None
        best_fitness = float('-inf')

        for fold, (train_index, test_index) in enumerate(kf.split(X_scaled), 1):
            print(f"Training fold {fold}/{n_splits}...")
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y_scaled[train_index], y_scaled[test_index]

            # Define evaluation function for this fold
            def eval_genomes(genomes, config):
                for genome_id, genome in genomes:
                    genome.fitness = self.eval_genome(genome, config, X_test, y_test)

            # Create population and add reporters
            p = neat.Population(self.config)
            p.add_reporter(neat.StdOutReporter(True))
            stats_reporter = neat.StatisticsReporter()
            p.add_reporter(stats_reporter)
            p.add_reporter(neat.Checkpointer(generation_interval=100, filename_prefix="neat-checkpoint-"))

            # Run NEAT
            winner = p.run(eval_genomes, n=generations)

            # Evaluate the best genome
            fitness = -self.eval_genome(winner, self.config, X_test, y_test)
            fold_fitnesses.append(fitness)
            print(f"Fold {fold} fitness: {fitness}")

            # Track the best genome across all folds
            if fitness > best_fitness:
                best_fitness = fitness
                best_genome = winner

        # Store the best genome as the model
        self.model = best_genome

        # Calculate and report average fitness
        average_fitness = sum(fold_fitnesses) / len(fold_fitnesses)
        print(f'Average Fitness across {n_splits} folds: {average_fitness}')

        # Save the model and scalers
        save_path = save_path or 'AIAppraiser.neat'
        joblib.dump(self.model, save_path)
        joblib.dump(self.scaler, save_path.replace('.neat', '_scaler.pkl'))
        joblib.dump(self.price_scaler, save_path.replace('.neat', '_price_scaler.pkl'))
        print(f"Model and scalers saved to {save_path}")

        return average_fitness

    def predict(self, cup, hero) -> float:
        """
        Make prediction for given Cup and Hero values.

        Args:
            cup: Cup value
            hero: Hero value

        Returns:
            predicted_price: The predicted price
        """
        if self.scaler is None or self.price_scaler is None or self.model is None:
            raise RuntimeError("Model and scalers must be loaded or trained before prediction.")

        # Validate inputs
        if not isinstance(cup, (int, float)) or not isinstance(hero, (int, float)):
            raise TypeError("Cup and Hero must be numeric values")

        # Make prediction
        example_data = pd.DataFrame([[cup, hero]], columns=['Cup', 'Hero'])
        example_data_scaled = self.scaler.transform(example_data)
        net = neat.nn.FeedForwardNetwork.create(self.model, self.config)
        predicted_value_scaled = net.activate(example_data_scaled[0])[0]

        # Inverse transform the prediction
        predicted_value = self.price_scaler.inverse_transform([[predicted_value_scaled]])[0][0]
        return predicted_value

    def evaluate_model(self, test_data=None) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.

        Args:
            test_data: DataFrame to use for evaluation (uses self.df_check if None)

        Returns:
            metrics: Dictionary containing evaluation metrics and predictions
        """
        df = test_data if test_data is not None else self.df_check
        if df is None:
            raise ValueError("No test data provided. Either pass test_data or initialize with test_path")

        X, y_true = self.df_division(df)

        # Generate predictions
        y_pred = []
        for _, row in df.iterrows():
            pred = self.predict(row['Cup'], row['Hero'])
            y_pred.append(pred)

        y_pred = np.array(y_pred)

        # Calculate metrics
        mse = np.mean((y_pred - y_true) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_pred - y_true))

        print(f"Model Evaluation Metrics:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'predictions': y_pred
        }

    def Q_Q_plot(self, data, save_prefix=''):
        """Generate histogram and Q-Q plot for data."""
        if not isinstance(data, pd.Series):
            if isinstance(data, np.ndarray):
                data = pd.Series(data)
            else:
                raise TypeError("Data must be a pandas Series or numpy array")

        name = data.name if data.name else 'data'

        # Create histogram
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=20, alpha=0.6, color='g', density=True)
        plt.title(f'{name} Histogram')

        # Add normal distribution curve
        x = np.linspace(data.min(), data.max(), 100)
        mean, std = data.mean(), data.std()
        plt.plot(x, stats.norm.pdf(x, mean, std), 'r-', linewidth=2)

        plt.grid(True, alpha=0.3)
        filename = f"{save_prefix}{name}-Histogram.png"
        plt.savefig(filename)
        print(f"Saved histogram to {filename}")
        plt.close()

        # Create Q-Q plot
        plt.figure(figsize=(10, 6))
        stats.probplot(data, dist="norm", plot=plt)
        plt.title(f'{name} Q-Q Plot')
        plt.grid(True, alpha=0.3)
        filename = f"{save_prefix}{name}-Q_Q_plot.png"
        plt.savefig(filename)
        print(f"Saved Q-Q plot to {filename}")
        plt.close()


def manual_input(model):
    """Interactive console for making predictions."""
    while True:
        print("\n=== Price Prediction Console ===")
        print('Enter Cup and Hero values separated by comma (e.g., 1000, 25)')
        print('Enter "q" to quit')

        user_input = input('Input: ').strip()
        if user_input.lower() == 'q':
            break

        try:
            values = [int(x.strip()) for x in user_input.split(',')]
            if len(values) != 2:
                raise ValueError("Need exactly two values")

            cup, hero = values
            prediction = model.predict(cup, hero)
            print(f"Predicted price: {prediction:.2f}")
        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")
        except Exception as e:
            print(f"Error: {e}")


def create_graph(model, file_path):
    """Create and display comparison graph of actual vs predicted prices."""
    try:
        # Load data
        df = pd.read_csv(file_path)

        # Generate predictions
        predictions = []
        for _, row in df.iterrows():
            pred = model.predict(row['Cup'], row['Hero'])
            predictions.append(pred)

        # Add predictions to DataFrame
        df['PredictedPrice'] = predictions

        # Create plot
        plt.figure(figsize=(12, 8))

        # Plot actual and predicted values
        x = np.arange(len(df))
        plt.scatter(x, df['Price'], label='Actual Price', color='blue', alpha=0.7, s=20)
        plt.scatter(x, df['PredictedPrice'], label='Predicted Price', color='red', alpha=0.7, s=20, marker='x')

        # Add regression line
        z = np.polyfit(df['Price'], df['PredictedPrice'], 1)
        p = np.poly1d(z)
        plt.plot(x, p(df['Price']), "g--", linewidth=2)

        # Calculate metrics
        mse = np.mean((df['Price'] - df['PredictedPrice']) ** 2)
        try:
            r2 = np.corrcoef(df['Price'], df['PredictedPrice'])[0, 1] ** 2
        except:
            r2 = 0

            # Add title and labels
        plt.title(f"Actual vs Predicted Prices (MSE: {mse:.2f}, R²: {r2:.2f})")
        plt.xlabel("Account Index")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Set y-axis limits with some padding
        y_min = min(df['Price'].min(), df['PredictedPrice'].min()) * 0.9
        y_max = max(df['Price'].max(), df['PredictedPrice'].max()) * 1.1
        plt.ylim(y_min, y_max)

        # Show plot
        plt.tight_layout()
        plt.show()

        return df

    except Exception as e:
        print(f"Error creating graph: {e}")
        return None


def create_graph_local(user_price, predicted_price):
    """
        Создает и отображает график сравнения предсказанной и введенной пользователем цены.

        Args:
            predicted_price (float): Предсказанная цена.
            user_price (float): Цена, введенная пользователем.
        """
    try:
        # Определяем максимальное значение для оси Y
        max_y = max(predicted_price, user_price, 2500)

        # Создаем график
        plt.figure(figsize=(6, 6))  # Adjust size as needed

        # Координаты X для точек
        x_values = [1, 2]  # Две точки

        # Координаты Y для точек
        y_values = [predicted_price, user_price]

        # Создаем график рассеяния
        plt.scatter(x_values, y_values, color=['red', 'blue'], s=100)  # s - размер точек

        # Добавляем подписи к точкам
        plt.text(x_values[0], y_values[0], f'Предсказанная: {predicted_price:.2f}', ha='center', va='bottom')
        plt.text(x_values[1], y_values[1], f'Введена пользователем: {user_price:.2f}', ha='center', va='bottom')

        # Настраиваем оси
        plt.xlim(0, 3)  # Чтобы точки не были по краям
        plt.ylim(0, max_y * 1.1)  # Y от 0 до max_y + немного места сверху

        # Убираем отметки на осях x
        plt.xticks([])

        # Добавляем заголовок и подписи
        plt.title('Сравнение цен')
        plt.ylabel('Цена')

        # Отображаем сетку для удобства
        plt.grid(True)

        # Отображаем график
        plt.show()

    except Exception as e:
        print(f'Error creating graph: {e}')

