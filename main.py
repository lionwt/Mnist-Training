import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.prepair import train_data, test_data
from model import NeuralNetwork, ConvolutionalNeuralNetwork
from eval import evaluate
from util.device import get_device

train_loader = DataLoader(
  train_data,
  batch_size=64,
  shuffle=True
)

test_loader = DataLoader(
  test_data,
  batch_size=64,
  shuffle=False
)

device = get_device()
# model = NeuralNetwork().to(device)

# model.load_state_dict(
#   torch.load("./model.pth", map_location="cpu", weights_only=False)
# )
# model.to(device)

model = ConvolutionalNeuralNetwork().to(device)

# model.load_state_dict(
#   torch.load("./model.pth", map_location="cpu", weights_only=False)
# )
# model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 5

for epoch in range(epochs):
  total_loss = 0
  start_time = time.time()

  for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    predictions = model(images)
    loss = loss_fn(predictions, labels)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()

  average_loss = total_loss / len(train_loader)

  print(
    f"Epoch {epoch+1}/{epochs}, Loss: {average_loss}, Time: {time.time() - start_time:.2f}s"
  )

torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, "model.pth")

acc, losses = evaluate(model, test_loader, device)

print(f"Accuracy: {acc:.4f}")
for i in range(len(losses)):
  print(f"Test Loss {i+1}: {losses[i]:.4f}")