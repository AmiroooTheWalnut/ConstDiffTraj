import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
from tqdm.auto import trange

np.random.seed(42)

N = 1000
x1 = np.random.uniform(-10, 10, N)
x2 = np.random.uniform(-10, 10, N)
noise = np.random.normal(0, 1, N)

# True function: y = 3*x1 - 2*x2 + 5 + noise
y = 3 * x1 - 2 * x2 + 5 + noise

# ========== Dataset ========== #
class RegressionDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.X = torch.tensor(df[['x1', 'x2']].values, dtype=torch.float32)
        self.y = torch.tensor(df['y'].values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

df = pd.DataFrame({
    'x1': x1,
    'x2': x2,
    'y': y
})

df.to_csv('simple_regression.csv', index=False)

# ========== Load Data ========== #
dataset = RegressionDataset('simple_regression.csv')
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# ========== Model ========== #
model = nn.Linear(2, 1)

# ========== Optimizer ========== #
# Choose either SGD or Adam
optimizer = optim.Adam(model.parameters(), lr=0.00036)  # <-- Try changing this!
# optimizer = optim.Adam(model.parameters(), lr=1e-10)  # Uncomment to try Adam

# ========== Loss Function ========== #
criterion = nn.MSELoss()

# ========== Training Loop ========== #
num_epochs = 10000
scheduler1 = ExponentialLR(optimizer, gamma=0.999)

bar = trange(num_epochs)

for i in bar:
    epoch_loss = 0.0

    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()



    #epoch_loss /= len(dataloader.dataset)
    scheduler1.step()

    bar.set_description(f'loss: {epoch_loss:.5f}, lr: {scheduler1.get_last_lr()[0]:.5f}')

    # # Print epoch info and current LR
    # current_lr = scheduler1.get_last_lr()[0]
    # print(f"Epoch {i + 1:02d} | Loss: {epoch_loss:.6f} | LR: {current_lr:.10f}")