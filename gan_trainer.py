from models.discriminator import Discriminator

class GANTrainer:
    def __init__(self, input_file='data/simulated_rssi_data.csv'):
        self.input_file = input_file
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.noise_dim = 10

    def train_gan(self, epochs=2000, batch_size=64):
        data = pd.read_csv(self.input_file)
        rssi_values = torch.tensor(data['RSSI'].values, dtype=torch.float32).unsqueeze(1).to(self.device)

        criterion = nn.BCELoss()
        optimizer_g = optim.Adam(self.generator.parameters(), lr=0.001)
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=0.001)

        for epoch in range(epochs):
            idx = torch.randint(0, rssi_values.size(0), (batch_size,))
            real_samples = rssi_values[idx]

            # Train Discriminator
            noise = torch.randn(batch_size, self.noise_dim).to(self.device)
            fake_samples = self.generator(noise)

            real_labels = torch.ones(batch_size, 1).to(self.device)
            fake_labels = torch.zeros(batch_size, 1).to(self.device)

            outputs_real = self.discriminator(real_samples)
            outputs_fake = self.discriminator(fake_samples.detach())

            d_loss_real = criterion(outputs_real, real_labels)
            d_loss_fake = criterion(outputs_fake, fake_labels)
            d_loss = d_loss_real + d_loss_fake

            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            outputs_fake = self.discriminator(fake_samples)
            g_loss = criterion(outputs_fake, real_labels)

            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            if (epoch + 1) % 500 == 0:
                print(f"[GANTrainer] Epoch [{epoch+1}/{epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

        # Generate synthetic samples
        noise = torch.randn(1000, self.noise_dim).to(self.device)
        synthetic_samples = self.generator(noise).detach().cpu().numpy()

        # Save synthetic samples
        synthetic_df = pd.DataFrame({'RSSI': synthetic_samples.flatten()})
        synthetic_df.to_csv('data/synthetic_rssi_data.csv', index=False)
        print(f"[GANTrainer] Synthetic RSSI data saved to data/synthetic_rssi_data.csv")
