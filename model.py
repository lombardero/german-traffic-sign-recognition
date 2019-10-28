import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB as 43 classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 148, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(148)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(148, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(4096, 256)
        self.fc2 = nn.Linear(256, nclasses)

    def forward(self, x):
        x = self.bn1(F.leaky_relu(self.conv1(x)))
        x = self.bn2(F.leaky_relu(self.conv2_drop(self.conv2(x))))
        x = self.bn3(F.leaky_relu(self.conv3_drop(self.conv3(x))))
        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
