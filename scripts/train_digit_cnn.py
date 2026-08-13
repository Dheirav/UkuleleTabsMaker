import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.vision.cnn_classifier import SmallDigitCNN


def train(output_path: str, epochs: int = 3) -> None:
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(train_data, batch_size=128, shuffle=True)

    model = SmallDigitCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        running = 0.0
        for images, labels in loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f"epoch {epoch + 1}/{epochs} loss {running / len(loader):.4f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"saved {output_path}")


if __name__ == "__main__":
    train("./models/digit_cnn.pth")
