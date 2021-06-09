# -*- coding: utf-8 -*-

#  DATA IMPORT
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from PIL import Image

# import matplotlib.pyplot as plt

data_path = 'D:/HSE/КУРСАЧ/DATA/'
file_name = 'my_fds_e20_b160.ptch'
print_out = "    Epoch: {:03d}, Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}"

# IMG TRANSFORMING
img_size = 128


class MyResize(object):  # resize function
    def __call__(self, img):
        # img_dtype = img.dtype
        w, h = img.size
        if w > img_size or h > img_size:
            img = transforms.Resize(size=(img_size, img_size))(img)
        # return img.astype(img_dtype)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


image_transforms = {  # transformation of input images
    'train': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],  # посчитать точнее
                             [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])
}

# DATA LOAD
train_directory = data_path + 'train'  # setting directory paths
val_directory = data_path + 'val'

batch_s = 160  # batch size
epochs_num = 30  # amount of epochs

data = {  # loading data from folders
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'val': datasets.ImageFolder(root=val_directory, transform=image_transforms['val'])
}

train_data_size = len(data['train'])  # size of data
val_data_size = len(data['val'])

train_data = DataLoader(data['train'], batch_size=batch_s, shuffle=True)  # iterators creation
val_data = DataLoader(data['val'], batch_size=batch_s, shuffle=True)

print(train_data_size, val_data_size)  # print the data sets sizes
print(data['train'].class_to_idx)


# INITIALIZATION OF NET
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 18, 7, padding=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(18),
            nn.MaxPool2d(2, 2))
        self.layer2 = torch.nn.Sequential(
            nn.Conv2d(18, 36, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(36),
            nn.MaxPool2d(2, 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(36, 55, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(55),
            nn.MaxPool2d(2, 2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(55, 72, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(72),
            nn.MaxPool2d(2, 2))
        self.fc = nn.Sequential(
            nn.Linear(72 * 8 * 8, 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, 8))

    def forward(self, x):
        # print(f"Shape of tensor: {x.shape}")
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc(x)
        # x = F.softmax(x, dim = 1)
        # print(f"Shape of tensor: {x.shape}")
        return x


net = Net()

criterion = nn.CrossEntropyLoss()  # loss function
optimizer = optim.AdamW(net.parameters(), lr=0.001)  # optimizer function


# VALIDATION
def validation(model, criterion):
    val_loss = 0.0
    val_acc = 0.0
    val_loss_arr = []
    val_acc_arr = []
    k = 0
    with torch.no_grad():
        model.eval()
        for j, (val_features, val_labels) in enumerate(val_data):
            outputs = model(val_features)
            # Compute loss
            loss = criterion(outputs, val_labels)
            # Compute the total loss for the batch and add it to valid_loss
            val_loss += loss.item() * val_features.size(0)
            # Calculate validation accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(val_labels.data.view_as(predictions))
            # Convert correct_counts to float and then compute the mean
            acc = torch.sum(correct_counts.type(torch.FloatTensor))
            # Compute total accuracy in the whole batch and add to valid_acc
            val_acc += acc.item()
            # val_loss_arr.append(loss.item() * val_features.size(0))
            # val_acc_arr.append(torch.sum(correct_counts).item())
        print("Epoch: {:03d}, Val Loss: {:.4f}, Val Accuracy: {:.4f}".format(epoch, val_loss / val_data_size,
                                                                             val_acc / val_data_size))
        model.train()
    print('Finished Validation')


# NET TRAINING
for epoch in range(epochs_num):
    train_loss = 0.0
    train_acc = 0.0
    train_loss_arr = []
    train_acc_arr = []
    train_iter_size = []

    for i, (train_features, train_labels) in enumerate(train_data):
        optimizer.zero_grad()

        outputs = net(train_features)

        # print(outputs)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()

        ret, predictions = torch.max(outputs.data, 1)
        correct_counts = predictions.eq(train_labels.data.view_as(predictions))

        acc = torch.mean(correct_counts.type(torch.FloatTensor))

        train_acc += acc.item()  # total accuracy

        train_loss_arr.append(loss.item() * train_features.size(0))
        train_iter_size.append(train_features.size(0))
        train_acc_arr.append(torch.sum(correct_counts).item())

        """
        if (i + 1) % 25 == 0 and (i > 100 or epoch > 0):
            print(print_out.format(epoch, i,
                                   sum(train_loss_arr[-50:]) / sum(train_iter_size[-50:]),
                                   sum(train_acc_arr[-50:]) / sum(train_iter_size[-50:])))
        """

    validation(net, criterion)  # vadidation
    torch.save(net.state_dict(), data_path + file_name + '_' + str(epoch))
print('Finished Training')

torch.save(net.state_dict(), data_path + file_name)  # saving the model weights
