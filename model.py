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

class ConvolutionalNeuralNetwork(nn.Module):
  def __init__(self):
    super(ConvolutionalNeuralNetwork, self).__init__()

    self.conv = nn.Sequential(
      nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )

    self.fc = nn.Sequential(
      nn.Flatten(),
      nn.Linear(64 * 7 * 7, 128),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(128, 10),
    )

  def forward(self, x):
    x = self.conv(x)
    x = self.fc(x)
    return x