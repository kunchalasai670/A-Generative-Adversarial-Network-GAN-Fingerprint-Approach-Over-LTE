import numpy as np
import pandas as pd
import random

class DataSimulator:
    def __init__(self, output_file='data/simulated_rssi_data.csv'):
        self.output_file = output_file
        self.grid_size = 4  # 4x4 grid

    def generate_data(self):
        locations = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        data = []
        for loc in locations:
            rssi_samples = np.random.normal(loc=-80, scale=5, size=30)  # around -80dBm
            for rssi in rssi_samples:
                imsi = random.randint(100000000000000, 999999999999999)
                data.append({'X': loc[0], 'Y': loc[1], 'RSSI': rssi, 'IMSI': imsi})

        df = pd.DataFrame(data)
        df.to_csv(self.output_file, index=False)
        print(f"[DataSimulator] Data generated and saved to {self.output_file}")
