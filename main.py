import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_sequences(data, window_size):
    sequences = []
    for i in range(len(data) - window_size):
        sequences.append(data[i:i+window_size])
    return np.array(sequences)


class Autoencoder(nn.Module):
    def __init__(self, window_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(window_size, 5),
            nn.ReLU(),
            nn.Linear(5, 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 5),
            nn.ReLU(),
            nn.Linear(5, window_size),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AD:

    def __init__(self):
        self.data_length = 300
        self.window_size = 10
        self.linspace = np.linspace(0, 20, self.data_length)
        self.normal_data = np.sin(self.linspace) + np.random.normal(scale=0.5, size=self.data_length)
        self.normal_data= self.normal_data

        self.anormal_data = self.normal_data.copy()
        self.anormal_data[50] += 6  # Anomaly 1
        self.anormal_data[150] += 7  # Anomaly 2
        self.anormal_data[250] += 8  # Anomaly 3

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info('Device: %s', self.device)

        self.model = Autoencoder(10).to(self.device)
        self.criterion = nn.MSELoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        self.normal_sequences = create_sequences(self.normal_data, self.window_size)
        self.normal_sequences = torch.tensor(self.normal_sequences, dtype=torch.float32).to(self.device)

        self.anormal_sequences = create_sequences(self.anormal_data, self.window_size)
        self.anormal_sequences = torch.tensor(self.anormal_sequences, dtype=torch.float32).to(self.device)




    def plot_data(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.plot(self.linspace, self.normal_data)
        ax2.plot(self.linspace, self.anormal_data)
        plt.show()

    def train(self):
        self.model.train()
        num_epochs = 100
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            output = self.model(self.normal_sequences)
            loss = self.criterion(output, self.normal_sequences)
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    def detect(self):
        # Anomaly detection
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(self.anormal_sequences)
            losses = torch.mean((predictions - self.anormal_sequences) ** 2, dim=1)
            if self.device == "cpu":
                plt.hist(losses.numpy(), bins=50)
                plt.xlabel("Loss")
                plt.ylabel("Frequency")
                plt.show()

        # Threshold for defining an anomaly

        threshold = losses.mean() + 2 * losses.std()
        print(f"Anomaly threshold: {threshold.item()}")

        # Detecting anomalies
        anomalies = losses > threshold
        print(f"Anomalies found at positions: {np.where(anomalies.cpu().numpy())[0]}")


if __name__ == '__main__':
    ad = AD()
    ad.train()
    ad.detect()






