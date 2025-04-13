import pandas as pd
import neat
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler# Import MinMaxScaler
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import joblib
import math

# 1. Define your custom activation function
def my_sinc_function(x):
    if x == 0:
        return 1.0
    else:
        return math.sin(x) / x

# 2. Define your custom aggregation function
def my_l2norm_function(x):
    return np.sqrt(np.sum(np.square(x)))

class AI_Creait():
    def __init__(self, config_file, model_path=''):
        self.df_training = pd.read_csv(r'C:\Users\user\PycharmProjects\SchoolProject\AIAppraiserAllFiles\ExecutableDirectory\first_DFFP.csv')
        self.df_check = pd.read_csv(r'C:\Users\user\PycharmProjects\SchoolProject\AIAppraiserAllFiles\ExecutableDirectory\last_DFFP.csv')
        self.config_file = config_file
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 config_file)

        # 3. Register the custom activation function in config
        self.config.genome_config.activation_defs.add("my_sinc_function", my_sinc_function)

        # 4. Register the custom aggregation function in config
        self.config.genome_config.aggregation_function_defs.add("my_l2norm_function", my_l2norm_function)

        self.model = None
        self.scaler = None
        self.price_scaler = None  # Scaler for the 'Price' column
        self.scaler_path = model_path.replace('.neat', '_scaler.pkl')
        self.price_scaler_path = model_path.replace('.neat', '_price_scaler.pkl')

        if os.path.exists(model_path) and os.path.exists(self.scaler_path) and os.path.exists(self.price_scaler_path):
            print("Loading existing model and scalers...")
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.price_scaler = joblib.load(self.price_scaler_path)
        else:
            print("Model or scalers not found. Training new model...")

    def df_division(self, df):
        X = df[['Cup', 'Hero']]
        y = df['Price']
        return X, y

    def eval_genome(self, genome, config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        X, y = self.df_division(self.df_training)

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Scale target variable 'Price'
        y_scaled = self.price_scaler.transform(y.values.reshape(-1, 1)).flatten()

        predictions = []
        for xi in X_scaled:
            output = net.activate(xi)
            predictions.append(output[0])

        predictions = np.array(predictions)
        fitness = np.mean((predictions - y_scaled) ** 2)  # MSE as fitness
        return -fitness  # NEAT maximizes fitness, so minimize MSE


    def train_model_with_neat(self, n_splits=30):
        if self.model is not None and self.scaler is not None and self.price_scaler is not None:
            print("Model and scalers already loaded. Skipping training.")
            return

        print("Starting NEAT training...")
        X, y = self.df_division(self.df_training)

        # Initialize scalers
        self.scaler = StandardScaler()
        self.price_scaler = MinMaxScaler()  # Use MinMaxScaler for 'Price'

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Scale target variable 'Price'
        y_scaled = self.price_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

        # K-Fold cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_fitnesses = []

        for fold, (train_index, test_index) in enumerate(kf.split(X_scaled, y_scaled), 1):
            print(f"Training fold {fold}...")
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y_scaled[train_index], y_scaled[test_index]

            # Prepare the fitness function with the current fold data
            def eval_genomes(genomes, config):
                for genome_id, genome in genomes:
                    net = neat.nn.FeedForwardNetwork.create(genome, config)

                    # Make predictions on the test set for this fold
                    predictions = []
                    for xi in X_test:
                        output = net.activate(xi)
                        predictions.append(output[0])  # Assuming a single output neuron
                    predictions = np.array(predictions)

                    # Calculate the fitness (mean squared error)
                    fitness = np.mean((predictions - y_test) ** 2)
                    genome.fitness = -fitness  # NEAT tries to maximize fitness

            # Create the population, which is the top-level object for a NEAT run.
            p = neat.Population(self.config)

            # Add a stdout reporter to show progress in the terminal.
            p.add_reporter(neat.StdOutReporter(True))
            stats = neat.StatisticsReporter()
            p.add_reporter(stats)
            p.add_reporter(neat.Checkpointer(generation_interval=100, filename_prefix="neat-checkpoint-"))

            # Run NEAT
            winner = p.run(eval_genomes, n=300)

            # Store the best genome
            self.model = winner

            # Evaluate the best genome on the test set
            net = neat.nn.FeedForwardNetwork.create(winner, self.config)
            predictions = []
            for xi in X_test:
                output = net.activate(xi)
                predictions.append(output[0])  # Assuming a single output neuron
            predictions = np.array(predictions)
            fitness = np.mean((predictions - y_test) ** 2)
            fold_fitnesses.append(-fitness)
            print(f"Fold {fold} fitness: {-fitness}")

        average_fitness = sum(fold_fitnesses) / len(fold_fitnesses)
        print(f'Average Fitness across {n_splits} folds: {average_fitness}')

        joblib.dump(self.model, 'AIAppraiser.neat')
        joblib.dump(self.scaler, 'AIAppraiser_scaler.pkl')
        joblib.dump(self.price_scaler, 'AIAppraiser_price_scaler.pkl')

    def predict(self, cup, hero):
        if self.scaler is None or self.price_scaler is None or self.model is None:
            raise Exception("Model and scalers must be loaded or trained before prediction.")

        example_data = pd.DataFrame([[cup, hero]], columns=['Cup', 'Hero'])
        example_data_scaled = self.scaler.transform(example_data)
        net = neat.nn.FeedForwardNetwork.create(self.model, self.config)
        predicted_value_scaled = net.activate(example_data_scaled[0])[0]

        # Inverse transform the prediction
        predicted_value = self.price_scaler.inverse_transform([[predicted_value_scaled]])[0][0]
        return predicted_value

    def Q_Q_plot(self, data):
        plt.hist(data, bins=10, alpha=0.6, color='g')
        plt.title(f'{data.name}-Histogram')
        plt.savefig(f'{data.name}-Histogram.png')
        plt.close()

        stats.probplot(data, dist="norm", plot=plt)
        plt.title(f'{data.name}-Q-Q Plot')
        plt.savefig(f'{data.name}-Q_Q_pot.png')
        plt.close()

if __name__ == '__main__':
    # Create a configuration file for NEAT
    config_file = "neat_config.txt"  # You'll need to create this file

    # Initialize and train the model with NEAT
    model = AI_Creait(config_file, 'AIAppraiser.neat')
    model.train_model_with_neat()

    # Make predictions using the trained model
    for index, row in model.df_check.iterrows():
        cup = row['Cup']
        hero = row['Hero']
        prediction = model.predict(cup, hero)
        print(f'Предсказание для Cup: {cup}, Hero: {hero} - {prediction}')
