This project implements a localization method using **Generative Adversarial Networks (GANs)** over LTE wireless signals (RSSI and RSRP). It uses simulated wireless signal data, augments the dataset using GAN, and applies a **Bayesian classifier** for position classification. The results are visualized through line graphs, heatmaps, confusion matrices, and comparative scatter plots.

How It Works:-

1. **Data Simulation**:
   - Generates synthetic RSSI and RSRP signals over 60 sample points.

2. **GAN Augmentation**:
   - Trains a GAN to augment wireless signal fingerprints.

3. **Bayesian Classification**:
   - Classifies locations based on learned signal patterns.

4. **Visualization**:
   - Includes:
     - Line graph of RSSI/RSRP signals
     - Heatmap visualization
     - Confusion matrix (raw and normalized)
     - Signal comparison scatter plot
     - Localization Accuracy improvement
