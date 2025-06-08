import numpy as np
import pandas as pd
import random
import os

# Create 'data' folder if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# Set parameters
grid_size = 4  # 4x4 grid (16 cells)
samples_per_cell = 30  # number of samples at each cell

# List to hold all data
data = []

# Create data
for i in range(grid_size):
    for j in range(grid_size):
        # Simulate RSSI samples for each location
        rssi_samples = np.random.normal(loc=-80, scale=5, size=samples_per_cell)  # Mean -80dBm, SD 5dB
        for rssi in rssi_samples:
            imsi = random.randint(100000000000000, 999999999999999)  # random IMSI
            data.append({
                'X': i,         # X coordinate (row)
                'Y': j,         # Y coordinate (col)
                'RSSI': rssi,   # Signal strength
                'IMSI': imsi    # Random mobile ID
            })

# Convert to pandas DataFrame
df = pd.DataFrame(data)

# Save to CSV
csv_path = 'data/simulated_rssi_data.csv'
df.to_csv(csv_path, index=False)

print(f"[Data Generation] Simulated RSSI data saved to {csv_path}")
