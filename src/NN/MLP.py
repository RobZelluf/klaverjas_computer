import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split

class Net(nn.Module):

    def __init__(self, input_size):
        super(Net, self).__init__()

        self.linear1 = nn.Linear(input_size, 20)
        self.linear2 = nn.Linear(20, 10)
        self.linear3 = nn.Linear(10, 5)
        self.linear4 = nn.Linear(5, 1)

        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x


data = pd.read_csv("winequality-white.csv", delimiter=";")
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
input_size = len(X_train.columns)

X_train = torch.tensor(X_train.values)
X_test = torch.tensor(X_test.values)
y_train = torch.tensor(y_train.values)
y_test = torch.tensor(y_test.values)

model = Net(input_size)
model.zero_grad()

criterion = nn.MSELoss()

epochs = 100

for i in range(epochs):
    output = model.forward(X_train.float())
    loss = criterion(output, y_train.float())
    print('epoch: ', i, ' loss: ', loss.item())
    model.optimizer.zero_grad()
    loss.backward()
    model.optimizer.step()

correct = 0
error = 0

for i in range(len(X_test)):
    output = model.forward(X_test[i].float())
    print(output.item(), y_test[i].item())
    if round(output.item()) == y_test[i].item():
        correct += 1
    else:
        error += 1

print("Accuracy:", round(correct / (correct + error), 2))