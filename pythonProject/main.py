import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt # For data viz
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm

from datasetDirectories import DatasetDirectories


class SimpleSpeciesClassifier(nn.Module):
    def __init__(self, num_classes = 525):
        super(SimpleSpeciesClassifier, self).__init__()
        self.base_model = timm.create_model('efficientnet_b3', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        enet_out_size = 1536
        self.classifier = nn.Linear(enet_out_size, num_classes)

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output


class BirdSpeciesDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @property
    def classes(self):
        return self.data.classes


paths = DatasetDirectories()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomChoice(transforms=[transforms.RandomPerspective(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.RandomAdjustSharpness(sharpness_factor=0.5, p=0.8),
                                        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.8),
                                        transforms.RandomAutocontrast(),
                                        transforms.RandomRotation(degrees=(-10, 10)),]),
    transforms.ToTensor(),
])
train_dataset = BirdSpeciesDataset(data_dir=paths.trainPath, transform=transform)
val_dataset = BirdSpeciesDataset(data_dir=paths.valPath, transform=transform)
test_dataset = BirdSpeciesDataset(data_dir=paths.testPath, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

CLASSES = list(train_dataset.data.class_to_idx.keys())

model = SimpleSpeciesClassifier()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00004)

num_epochs = 8
train_losses, val_losses, train_accuracy, val_accuracy = [], [], [], []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

def calc_accuracy(true,pred):
    pred = F.softmax(pred, dim = 1)
    true = torch.zeros(pred.shape[0], pred.shape[1]).scatter_(1, true.unsqueeze(1), 1.)
    acc = (true.argmax(-1) == pred.argmax(-1)).float().detach().numpy()
    acc = float((100 * acc.sum()) / len(acc))
    return round(acc, 4)


def train_model(model, criterion, optimizer, num_epochs, train_losses, val_losses, train_accuracy, val_accuracy):
    for epoch in range(num_epochs):
        train_epoch_accuracy = []
        val_epoch_accuracy = []
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc="Training:"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            acc = calc_accuracy(labels.cpu(), outputs.cpu())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            train_epoch_accuracy.append(acc)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        train_epoch_accuracy = np.mean(train_epoch_accuracy)
        train_accuracy.append(train_epoch_accuracy)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation:"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                acc = calc_accuracy(labels.cpu(), outputs.cpu())
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * labels.size(0)
                val_epoch_accuracy.append(acc)
        val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_epoch_accuracy = np.mean(val_epoch_accuracy)
        val_accuracy.append(val_epoch_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss},"
              f" Train accuracy: {train_epoch_accuracy}, Validation accuracy: {val_epoch_accuracy}")

        plt.plot(train_losses, label='Training loss')
        plt.plot(val_losses, label='Validation loss')
        plt.yticks(np.arange(0, max(train_losses), 0.2))
        plt.legend()
        plt.title("Loss over epochs")
        plt.show()
    return model, train_losses, val_losses

def imshowaxis(ax, img, orig, pred):
    img = img / 2 + 0.5
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    if orig != pred:
        ax.set_title(orig + "\n" + pred, color="red")
    else:
        ax.set_title(orig + "\n" + pred)
    ax.axis("off")


def vis_model(model, num_images=25):
    was_training = model.training
    model.eval()
    images_so_far = 0
    figure, ax = plt.subplots(5, 5, figsize=(20, 20))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for i in range(5):
                for j in range(5):
                    if images_so_far < num_images:
                        imshowaxis(ax[i][j], inputs.cpu().data[images_so_far], CLASSES[labels[images_so_far]],
                                   CLASSES[preds[images_so_far]])
                    else:
                        model.train(mode=was_training)
                        return
                    images_so_far += 1
        model.train(mode=was_training)


model, train_losses, val_losses = train_model(model, criterion, optimizer, num_epochs, train_losses, val_losses, train_accuracy, val_accuracy)
vis_model(model)
plt.show()