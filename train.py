import torch
from torch import nn
from torch import optim
from tqdm import tqdm
import os
from torchvision import transforms
import numpy as np

from model import CNN
from data import FruitData

def init_weights(layer):
    if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

def main():
    train_dir = './data/train'
    test_dir = './data/test'

    training_transforms = transforms.Compose([
        transforms.Resize((250, 250)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    testing_transforms = transforms.Compose([
        transforms.Resize((250, 250)),
        transforms.ToTensor()
    ])

    training_set = FruitData(train_dir, transform=training_transforms)
    testing_set = FruitData(test_dir, transform=testing_transforms)

    train_loader = torch.utils.data.DataLoader(training_set, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testing_set, batch_size=64, shuffle=True)

    model = CNN()
    model.apply(init_weights)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.90)
    loss_func = nn.CrossEntropyLoss()

    num_epochs = 15

    print('Training...')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, total=len(train_loader)):
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f'Epoch {epoch}, Total Training Loss: {train_loss:.4f}')

        model.eval()
        correct = 0
        for images, labels in test_loader:
            with torch.inference_mode():
                predictions = model(images)
                predicted_classes = torch.argmax(predictions, dim=1)

            correct += np.array([predicted_classes[i]==labels[i] for i in range(len(labels))]).sum().item()  # Convert to Python scalar

        accuracy = correct / 1025
        print(f'Epoch {epoch}, Test Accuracy: {accuracy*100:.2f}%')

    if not os.path.exists('weights'):
        os.makedirs('weights')
    torch.save(model.state_dict(), './weights/model_weight.pth')

if __name__ == '__main__':
    main()
