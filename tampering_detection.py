import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import warnings

warnings.filterwarnings("ignore")
plt.ion()

path = 'C:/dataset/MICC-F220/MICC-F220/'

class TamperingDataset(Dataset):
    """Tampering dataset."""

    def __init__(self, root_dir, transform=None):
        self.root = root_dir
        self.imageNames = os.listdir(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.imageNames)

    def __getitem__(self, idx):
        img_name = self.imageNames[idx]
        img_path = os.path.join(self.root, img_name)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = 1 if 'tamp' in img_name else 0
        return (img, label)

transformation = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

dataset = TamperingDataset(path, transform=transformation)
print('Dataset read successfully')
print(len(dataset))

# Split the dataset into training and testing sets
trainset_size = int(0.9 * len(dataset))
testset_size = len(dataset) - trainset_size
trainset, testset = torch.utils.data.random_split(dataset, [trainset_size, testset_size])

print('Number of training images:', len(trainset))
print('Number of test images:', len(testset))

# Initialize DataLoaders with num_workers=0 for debugging
trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
testloader = DataLoader(testset, batch_size=4, shuffle=True, num_workers=0)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:
            # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')
PATH = './tampering_net.pth'
torch.save(net.state_dict(), PATH)

# Test the network on the test data
dataiter = iter(testloader)
images, labels = next(dataiter)

# Print images and ground truth labels
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

imshow(torchvision.utils.make_grid(images))
print('GroundTruth:', ' '.join(f'{labels[j]}' for j in range(4)))

# Load the trained model for inference
net = Net()
net.load_state_dict(torch.load(PATH))

# Perform prediction
outputs = net(images)
_, predicted = torch.max(outputs, 1)

print('Predicted:', ' '.join(f'{predicted[j]}' for j in range(4)))

# Calculate accuracy on the test set
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct // total} %')
