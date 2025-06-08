import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

class BayesianClassifier:
    def __init__(self):
        self.model = GaussianNB()
        self.X_train = None  # Assign appropriate training data
        self.y_train = None  # Assign appropriate target labels


    def train_and_evaluate(self, use_synthetic=True):
        real_data = pd.read_csv('data/simulated_rssi_data.csv')
        synthetic_data = pd.read_csv('data/synthetic_rssi_data.csv')

        real_data['source'] = 'real'
        synthetic_data['source'] = 'synthetic'

        if use_synthetic:
            # Combine real and synthetic data
            combined_data = pd.concat([real_data, synthetic_data], ignore_index=True)
        else:
            # Use only real data for baseline comparison
            combined_data = real_data

        X = combined_data[['RSSI']]
        y = combined_data[['X', 'Y']].astype(str).agg('_'.join, axis=1)  # Create 'X_Y' strings

        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        # Split the dataset into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")

        # Train the model
        self.model.fit(X_train, y_train)

        # Predict and calculate accuracy
        preds = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, preds) * 100
        return accuracy

    def plot_accuracy(self):
        # Get accuracy before and after using GAN
        accuracy_before_gan = self.train_and_evaluate(use_synthetic=False)  # Training without synthetic data (baseline)
        accuracy_after_gan = self.train_and_evaluate(use_synthetic=True)    # Training with synthetic data (after GAN)

        # Plot the results
        labels = ['Before GAN', 'After GAN']
        values = [accuracy_before_gan, accuracy_after_gan]

        plt.bar(labels, values, color=['red', 'green'])
        plt.ylabel('Accuracy (%)')
        plt.title('Localization Accuracy Improvement')
        plt.show()

# Instantiate and run the model
classifier = BayesianClassifier()
classifier.plot_accuracy()


