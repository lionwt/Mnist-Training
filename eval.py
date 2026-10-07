import torch
import torch.nn as nn

def evaluate(model, test_loader, device, epoches = 5):
  model.eval()

  correct = 0
  total = 0

  loss_fn = nn.CrossEntropyLoss()
  losses = []
  count = len(test_loader) / epoches
  print(len(test_loader),count)

  with torch.no_grad():
    for images, labels in test_loader:
      images, labels = images.to(device), labels.to(device)
      predictions = model(images)
      loss = loss_fn(predictions, labels)
      losses.append(loss.item())
      predicted_labels = predictions.argmax(dim=1)
      total += labels.size(0)
      correct += (predicted_labels == labels).sum().item()

  accuracy = correct / total

  return accuracy, losses