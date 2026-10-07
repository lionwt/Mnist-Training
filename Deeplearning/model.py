import torch.nn as nn

class NeuralNetwork(nn.Module):
  def __init__(self):
    super(NeuralNetwork, self).__init__()

    self.flatten = nn.Flatten()

    self.fc1 = nn.Linear(28 * 28, 256)
    self.relu1 = nn.ReLU()
    self.fc2 = nn.Linear(256, 128)
    self.relu2 = nn.ReLU()
    self.fc3 = nn.Linear(128, 10)

    nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
    nn.init.zeros_(self.fc1.bias)

    nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
    nn.init.zeros_(self.fc2.bias)

    nn.init.xavier_normal_(self.fc3.weight)
    nn.init.zeros_(self.fc3.bias)

  def forward(self, x):
    x = self.flatten(x)
    x = self.fc1(x)
    x = self.relu1(x)
    x = self.fc2(x)
    x = self.relu2(x)
    x = self.fc3(x)
    return x
