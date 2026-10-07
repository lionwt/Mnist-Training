import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch

def evaluate(model, test_loader, device):
  model.eval()

  correct = 0
  total = 0

  with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        predictions = model(images)
        predicted_labels = predictions.argmax(dim=1)
        total += labels.size(0)
        correct += (
            predicted_labels == labels
        ).sum().item()

  accuracy = correct / total

  return accuracy

