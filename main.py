from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# import matplotlib.pyplot as plt

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=250, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

torch.manual_seed(args.seed)

workers = 1
# Check if CUDA is available
if torch.cuda.is_available():
    using_gpu = True
    print("GPU enabled")
    workers = 4
else:
	using_gpu = False
	print("GPU not enabled")

### Data Initialization and Loading
from data import initialize_data, data_transforms, data_crop, data_rot, data_rotshear,\
                data_transl, data_cropshear, data_jitter1, data_grey, data_jitter3, data_jitter4 # data.py in the same folder
initialize_data(args.data) # extracts the zip files, makes a validation set

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.ConcatDataset([
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_crop),
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_rot),
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_jitter1),
#    datasets.ImageFolder(args.data + '/train_images',
#                         transform=data_flip),
#    datasets.ImageFolder(args.data + '/train_images',
#                         transform=data_grey),
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_rotshear),
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transl),
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_jitter3),
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_cropshear),
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_jitter4),
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transforms)]),
    batch_size=args.batch_size, shuffle=True, num_workers=4)
    
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=4)

### Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
from model import Net
model = Net()

# Update model if using GPU
if using_gpu:
    model.cuda()

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        # GPU modif
        if using_gpu:
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        # GPU modif
        if using_gpu:
            data = data.cuda()
            target = target.cuda()
        output = model(data)
        validation_loss += F.nll_loss(output, target, size_average=False).data.item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


for epoch in range(1, args.epochs + 1):

    # addded code:
    # train_loss = []
    # validation_err = []
    
    # changed code:
    # train_loss.extend(train(epoch))
    # validation_err.append(validation())
    train(epoch)
    validation()

    model_file = 'model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
    print('\nSaved model to ' + model_file + '. You can run `python evaluate.py --model' + model_file + '` to generate the Kaggle formatted csv file')

# Plotting loss fn & test error (validation)
# plt.plot(range(len(train_loss)), train_loss, 'b')
# plt.xlabel('iteration')
# plt.ylabel('train_loss')
# plt.show()

# plt.plot(range(len(validation_err)), validation_err, 'r')
# plt.xlabel('epoch')
# plt.ylabel('validation_err')
# plt.show()
