import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import trange
from torch.optim.lr_scheduler import ExponentialLR
from pathlib import Path
from LossVisualizer import LossVisualizer

# Minimal model
class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        x = nn.functional.softmax(x, dim=-1)
        return x

continueTrain=True

# Create model
input_dim = 20
output_dim = 3
model = SimpleModel(input_dim, output_dim)

# Dummy data
x = torch.randn(2, input_dim)
target = torch.tensor([0, 2])

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adagrad(model.parameters(), lr=0.0001)

scheduler1 = ExponentialLR(optimizer, gamma=0.995)

my_file = Path('model_test.pytorch')
if my_file.is_file() and continueTrain==True:
    model.load_state_dict(torch.load('model_test.pytorch'))
    optimizer.load_state_dict(torch.load('optim_model_test.pytorch'))

# Forward + backward pass
bar = trange(100)
lv=LossVisualizer(100)
for i in bar:
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    lv.values[i] = loss
    bar.set_description(f'loss: {loss:.5f}, lr: {scheduler1.get_last_lr()[0]:.5f}')

torch.save(model.state_dict(), 'model_test.pytorch')
torch.save(optimizer.state_dict(), 'optim_model_test.pytorch')

lv.visualize(saveFileName='Loss_pytorch_model_test.png',
                     titleText="Pytorch loss value over iterations. Model model_test.")