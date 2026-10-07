import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Download MNIST
train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# Get one image and its label
images = [train_data[i][0] for i in range(25)]

grid = torch.cat([
  torch.cat(images[i * 5:(i + 1) * 5], dim=2) for i in range(5)
], dim=1)

plt.imshow(grid.squeeze(), cmap="gray")
plt.axis("off")
plt.show()
