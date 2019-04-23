import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):

    def __init__(self, input_size):
        super(Net, self).__init__()

        self.linear1 = nn.Linear(input_size, 30)
        self.linear2 = nn.Linear(30, 30)
        self.linear3 = nn.Linear(30, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


input = torch.randn(20)

model = Net(20)
criterion = nn.MSELoss()
model.zero_grad()
y = model.forward(input)
print(y)