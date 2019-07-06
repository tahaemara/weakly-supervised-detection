"""
Created on Fri Jul 3 15:12:30 2019

@author: Taha Emara  @email: taha@emaraic.com

"""
import torch
import torch.nn as nn
import torch.optim  as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import numpy as np
import os
import time
import argparse
import PIL
from datetime import datetime

# To make reproducible results
torch.manual_seed(125)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(125)

ap = argparse.ArgumentParser()
ap.add_argument('--dataset_train_path', required=False,
                help='path to train dataset store', default='./dataset/train')
ap.add_argument('--classes', required=False,
                help='number of classes', default=3)

args = ap.parse_args()
dataset_train_path = args.dataset_train_path
classes = args.classes

epochs = 150
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
# make a folder -with name of current time- for every experiment
experiment_id = datetime.now().strftime("%Y-%m-%d_%H_%M")
save_path = os.path.join(save_dir_root, 'experiments', 'experiment_' + str(experiment_id))
os.mkdir(save_path)

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, classes)

model.to(device)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

full_dst = datasets.ImageFolder(root=dataset_train_path, transform=data_transforms['train'])
train_size = int(0.8 * len(full_dst))
test_size = len(full_dst) - train_size
train_dst, val_dst = torch.utils.data.random_split(full_dst, [train_size, test_size])

trainloader = DataLoader(train_dst, batch_size=32, shuffle=True)
valloader = DataLoader(val_dst, batch_size=1)

criterion = nn.CrossEntropyLoss()
optim = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optim, step_size=6, gamma=0.1)

best_acc = 0.0
since = time.time()
for epoch in range(0, epochs):
    epoch_since = time.time()
    hs = open(os.path.join(save_path, "output.txt"), "a")

    running_loss_tr = 0.0
    running_corrects = 0

    exp_lr_scheduler.step()
    print("Learning Rate: ", str(exp_lr_scheduler.get_lr()))
    model.train()

    for ii, sample_batched in enumerate(trainloader):

        inputs, labels = sample_batched[0], sample_batched[1]
        inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
        inputs, labels = inputs.to(device), labels.to(device)

        optim.zero_grad()
        outputs = model.forward(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optim.step()
        ls = loss.item()
        if ii % 10 == 0:#print loss every 10 iterations
            print(str(ls))
            hs.write('Loss: {:.6f} \n'.format(ls))
        running_loss_tr += ls
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss_tr / len(train_dst)
    epoch_acc = running_corrects.double() / len(train_dst)
    time_elapsed = time.time() - epoch_since

    print('Epoch takes {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    print('Epoch:{} Train Loss: {:.6f} Acc: {:.4f}'.format(
        epoch, epoch_loss, epoch_acc))
    hs.write('Epoch takes {:.0f}m {:.0f}s \n'.format(
        time_elapsed // 60, time_elapsed % 60))
    hs.write('Epoch:{} Train Loss: {:.6f} Acc: {:.4f} \n'.format(
        epoch, epoch_loss, epoch_acc))

    if epoch % 4 == 0:
        model.eval()
        running_loss_val = 0.0
        running_corrects = 0
        for ii, sample_batched in enumerate(valloader):
            inputs, labels = sample_batched[0], sample_batched[1]
            inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model.forward(inputs)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)
            ls = loss.item()
            running_loss_tr += ls
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss_val / len(val_dst)
        epoch_acc_val = running_corrects.double() / len(val_dst)
        print('Val Loss: {:.4f} Acc: {:.4f}'.format(
            epoch_loss, epoch_acc_val))
        hs.write('Epoch:{}  Val Loss: {:.4f} Acc: {:.4f} \n'.format(
            epoch, epoch_loss, epoch_acc_val))
        if epoch_acc_val > best_acc:
            best_acc = epoch_acc_val
            torch.save(model.state_dict(), os.path.join(save_path, "epoch " + str(epoch) + ".pth"))
    hs.close()
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))

