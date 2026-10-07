import torch

from util.device import get_device
from data.prepair import test_data
from torch.utils.data import DataLoader
from eval import evaluate
from model import ConvolutionalNeuralNetwork

test_loader = DataLoader(
  test_data,
  batch_size=64,
  shuffle=False
)

device = get_device()

model = ConvolutionalNeuralNetwork()

model.load_state_dict(torch.load("./model.pth", map_location="cpu", weights_only=False))

model.to(device)

acc, losses = evaluate(model, test_loader, device, 5)

print(f"Accuracy: {acc:.4f}")
for i in range(len(losses)):
  print(f"Test Loss {i+1}: {losses[i]:.4f}")