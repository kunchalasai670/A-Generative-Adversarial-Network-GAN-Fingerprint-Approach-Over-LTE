from utils.data_simulator import DataSimulator
from utils.gan_trainer import GANTrainer
from models.bayesian_classifier import BayesianClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def main():
    # Step 1: Simulate RSSI data
    simulator = DataSimulator()
    simulator.generate_data()

    # Step 2: Train GAN and augment dataset
    gan_trainer = GANTrainer()
    gan_trainer.train_gan()

    # Step 3: Train Bayesian classifier
    classifier = BayesianClassifier()
    classifier.train()

    # Step 4: Test and visualize results
    classifier.evaluate()

class WirelessSignalVisualizer:
    def __init__(self):
        self.rssi_data = None
        self.rsrp_data = None
        self.sample_indices = None

    def load_sample_data(self, num_samples=60):
        """
        Generate sample RSSI and RSRP data or load from file
        """
        # Random data generation for demonstration
        self.sample_indices = np.arange(num_samples)

        # Generate RSSI values from -85 to -55 dBm with some variations
        rssi_base = np.linspace(-85, -55, num_samples)
        rssi_noise = np.random.normal(0, 3, num_samples)
        self.rssi_data = rssi_base + rssi_noise

        # RSRP is typically lower than RSSI
        rsrp_offset = np.random.normal(0, 2, num_samples)
        self.rsrp_data = self.rssi_data + rsrp_offset - 2

        # Add some realistic variations to make the data more interesting
        # First samples have lower signal strength
        self.rssi_data[:15] = np.random.uniform(-85, -78, 15)
        self.rsrp_data[:15] = np.random.uniform(-89, -83, 15)

        # Middle samples have stronger signal
        self.rssi_data[25:40] = np.random.uniform(-60, -55, 15)
        self.rsrp_data[30:40] = np.random.uniform(-68, -62, 10)

        # Some fluctuations at the end
        self.rssi_data[45:55] = np.random.uniform(-73, -60, 10)
        self.rsrp_data[45:55] = np.random.uniform(-71, -56, 10)

        return self.sample_indices, self.rssi_data, self.rsrp_data

    def load_data_from_file(self, filepath):
        """
        Load RSSI and RSRP data from a CSV file
        """
        df = pd.read_csv(filepath)
        self.sample_indices = df['sample_index'].values
        self.rssi_data = df['rssi'].values
        self.rsrp_data = df['rsrp'].values
        return self.sample_indices, self.rssi_data, self.rsrp_data

    def plot_line_graph(self, save_path=None):
        """
        Create a line graph similar to the one shown in the image (a)
        """
        plt.figure(figsize=(10, 8))
        plt.plot(self.sample_indices, self.rssi_data, 'o-', color='orangered', label='RSSI', linewidth=2, markersize=8)
        plt.plot(self.sample_indices, self.rsrp_data, 'o-', color='steelblue', label='RSRP', linewidth=2, markersize=8)

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('dB', fontsize=12)
        plt.ylim([-90, -50])
        plt.xlim([0, len(self.sample_indices)])
        plt.legend(fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def create_heatmap(self, num_rows=20, save_path=None):
        """
        Create a heatmap visualization similar to the one shown in the image (b)
        """
        # Create data matrices for the heatmap
        # We'll create rows of data points within signal strength ranges
        rssi_min, rssi_max = np.min(self.rssi_data), np.max(self.rssi_data)
        rsrp_min, rsrp_max = np.min(self.rsrp_data), np.max(self.rsrp_data)

        # Create matrices for the heatmap
        rssi_matrix = np.zeros((num_rows, 1))
        rsrp_matrix = np.zeros((num_rows, 1))

        # Fill with values distributed across the range
        for i in range(num_rows):
            # Use a combination of the original data distribution with some randomization
            if i < num_rows / 4:
                rssi_matrix[i, 0] = np.random.uniform(rssi_min, rssi_min + 10)
                rsrp_matrix[i, 0] = np.random.uniform(rsrp_min, rsrp_min + 5)
            elif i < num_rows / 2:
                rssi_matrix[i, 0] = np.random.uniform(rssi_min + 10, rssi_min + 20)
                rsrp_matrix[i, 0] = np.random.uniform(rsrp_min + 5, rsrp_min + 15)
            elif i < 3 * num_rows / 4:
                rssi_matrix[i, 0] = np.random.uniform(rssi_max - 20, rssi_max - 10)
                rsrp_matrix[i, 0] = np.random.uniform(rsrp_max - 15, rsrp_max - 5)
            else:
                rssi_matrix[i, 0] = np.random.uniform(rssi_max - 10, rssi_max)
                rsrp_matrix[i, 0] = np.random.uniform(rsrp_max - 5, rsrp_max)

        # Combine matrices side by side
        combined_matrix = np.hstack((rssi_matrix, rsrp_matrix))

        # Create a custom colormap similar to the one in the image
        colors = ['purple', 'blue', 'cyan', 'green', 'yellow', 'red']
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

        plt.figure(figsize=(8, 10))
        ax = sns.heatmap(combined_matrix, cmap=cmap, vmin=-90, vmax=-50,
                         cbar_kws={'label': 'dBm', 'ticks': [-90, -80, -70, -60, -50]})

        # Set x-axis labels
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(['RSSI', 'RSRP'], fontsize=12)

        # Remove y-axis labels
        ax.set_yticks([])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def visualize_all(self, save_line_graph=None, save_heatmap=None):
        """
        Generate both visualizations in separate figures
        """
        self.plot_line_graph(save_path=save_line_graph)
        self.create_heatmap(save_path=save_heatmap)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os


def generate_signal_confusion_matrix():
    """
    Generate a confusion matrix visualization based on the wireless signal data.
    This shows the relationship between RSSI and RSRP signal strength categories.
    """
    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Define exact RSSI and RSRP data from the original image
    rssi_data = np.array([
        -85, -87, -86, -85, -82, -78, -82, -80, -83, -77, -79, -81, -83, -81, -78,
        -77, -76, -78, -79, -77, -78, -82, -79, -77, -74, -70, -68, -67, -65, -62,
        -54, -55, -55, -57, -54, -56, -59, -55, -57, -60, -62, -65, -60, -61, -69,
        -70, -68, -67, -65, -69, -69, -73, -56, -61, -60, -61, -65, -66, -60
    ])

    rsrp_data = np.array([
        -89, -88, -84, -86, -82, -82, -84, -81, -78, -83, -84, -76, -83, -79, -77,
        -79, -81, -79, -77, -77, -76, -74, -71, -69, -67, -65, -65, -62, -67, -65,
        -62, -63, -65, -65, -69, -68, -68, -67, -65, -63, -67, -68, -69, -70, -65,
        -66, -64, -57, -59, -62, -65, -67, -70, -65, -63, -62, -60, -71, -70
    ])

    # Define signal strength categories
    def categorize_signal(signal_strength):
        if signal_strength >= -60:
            return "Excellent (-60 to -50 dBm)"
        elif signal_strength >= -70:
            return "Good (-70 to -60 dBm)"
        elif signal_strength >= -80:
            return "Fair (-80 to -70 dBm)"
        else:
            return "Poor (< -80 dBm)"

    # Categorize the signal data
    rssi_categories = [categorize_signal(val) for val in rssi_data]
    rsrp_categories = [categorize_signal(val) for val in rsrp_data]

    # Define the categories in order
    categories = ["Excellent (-60 to -50 dBm)", "Good (-70 to -60 dBm)",
                  "Fair (-80 to -70 dBm)", "Poor (< -80 dBm)"]

    # Create confusion matrix
    cm = confusion_matrix(rssi_categories, rsrp_categories, labels=categories)

    # Create normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot the confusion matrix
    plt.figure(figsize=(12, 10))

    # Plot raw counts
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=categories, yticklabels=categories)
    plt.xlabel('RSRP Classification', fontsize=12)
    plt.ylabel('RSSI Classification', fontsize=12)
    plt.title('Confusion Matrix: RSSI vs RSRP', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Plot normalized percentages
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", cbar=True,
                xticklabels=categories, yticklabels=categories)
    plt.xlabel('RSRP Classification', fontsize=12)
    plt.ylabel('RSSI Classification', fontsize=12)
    plt.title('Normalized Confusion Matrix (%)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    # Save the confusion matrix visualization
    confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')

    print(f"Confusion matrix saved to: {confusion_matrix_path}")
    plt.show()

    # Create a more detailed comparison plot
    plt.figure(figsize=(14, 8))

    # Scatter plot comparing RSSI and RSRP
    plt.subplot(1, 2, 1)
    plt.scatter(rssi_data, rsrp_data, c=np.abs(rssi_data - rsrp_data),
                cmap="viridis", alpha=0.8, s=80, edgecolors='k')
    plt.colorbar(label="Absolute Difference (dB)")
    plt.xlabel("RSSI (dBm)", fontsize=12)
    plt.ylabel("RSRP (dBm)", fontsize=12)
    plt.title("RSSI vs RSRP Scatter Plot", fontsize=14)

    # Add 45-degree line (perfect correlation)
    min_val = min(np.min(rssi_data), np.min(rsrp_data))
    max_val = max(np.max(rssi_data), np.max(rsrp_data))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)

    plt.grid(True, alpha=0.3)

    # Add regions
    plt.axvline(-80, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(-70, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(-60, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(-80, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(-70, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(-60, color='gray', linestyle='--', alpha=0.5)

    # Add category labels
    plt.text(-85, -50, "Poor RSSI", fontsize=10, ha='center', va='bottom', rotation=90)
    plt.text(-75, -50, "Fair RSSI", fontsize=10, ha='center', va='bottom', rotation=90)
    plt.text(-65, -50, "Good RSSI", fontsize=10, ha='center', va='bottom', rotation=90)
    plt.text(-55, -50, "Excellent RSSI", fontsize=10, ha='center', va='bottom', rotation=90)

    plt.text(-50, -85, "Poor RSRP", fontsize=10, ha='left', va='center')
    plt.text(-50, -75, "Fair RSRP", fontsize=10, ha='left', va='center')
    plt.text(-50, -65, "Good RSRP", fontsize=10, ha='left', va='center')
    plt.text(-50, -55, "Excellent RSRP", fontsize=10, ha='left', va='center')

    # Add divergence histogram
    plt.subplot(1, 2, 2)
    difference = rssi_data - rsrp_data
    plt.hist(difference, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(difference), color='red', linestyle='--',
                label=f'Mean Difference: {np.mean(difference):.2f} dB')
    plt.axvline(0, color='green', linestyle='-',
                label='No Difference')
    plt.xlabel("RSSI - RSRP Difference (dB)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Distribution of RSSI-RSRP Differences", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    # Save the comparison plot
    comparison_path = os.path.join(output_dir, "signal_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')

    print(f"Signal comparison plot saved to: {comparison_path}")
    plt.show()

    return confusion_matrix_path, comparison_path


if __name__ == "__main__":
    generate_signal_confusion_matrix()



def main():
    # Create the visualizer
    visualizer = WirelessSignalVisualizer()

    # Option 1: Generate sample data
    visualizer.load_sample_data(60)

    # Option 2: Load from CSV file (uncomment if you have a file)
    # visualizer.load_data_from_file('wireless_signal_data.csv')

    # Generate both visualizations
    visualizer.visualize_all(save_line_graph='line_graph.png', save_heatmap='heatmap.png')

    # Or generate them separately
    # visualizer.plot_line_graph(save_path='line_graph.png')
    # visualizer.create_heatmap(save_path='heatmap.png')


if __name__ == "__main__":
    main()
