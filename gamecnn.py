"""
Author: Da Chen
CNN model for real-time hand gesture recognition.
"""
import sys
import cv2
import torch
import random
import pyautogui
import numpy as np
import pandas as pd
import torch.nn as nn
from scipy import stats
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from skimage import io, transform
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


"""
1. HyperParameters and auxiliary functions.
"""
# Show the whole nparray.
np.set_printoptions(threshold=np.inf)
# Number of epochs to run.
num_epochs = 6
# Number of samples per batch to load.
batch_size = 32
# Number of subprocesses to use for data loading.
num_workers = 4
# Times of validation splitting.
times_val_split = 10
# Use gpu or not.
use_gpu = torch.cuda.is_available()
# Set FAILSAFE = False.
pyautogui.FAILSAFE= False

# Read command line arguments.
mode = None
weights_file_name = None
if len(sys.argv) >= 3:
    mode = str(sys.argv[1])
    weights_file_name = str(sys.argv[2])

# Create binary image.
def binaryMask(frame):
    roi = frame

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return res

# Resize the input image.
def reSize(img, output_size):
    if len(img.shape) < 3:
        img = img.reshape(img.shape[0], img.shape[1], 1)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img.reshape(img.shape[0], img.shape[1], 1)
    img = transform.resize(img, (output_size, output_size), mode='constant')
    return img

# Normalize the image.
def normalize(img, mean, std):
    img = (img - mean) / std
    return img

# Transfer to tensor.
def toTensor(img):
    img = img.transpose((2, 0, 1))
    return torch.from_numpy(img).type(torch.DoubleTensor)


"""
2. (a)Define customized dataset. (b)Define transforms. (c)Load data and pre-processing.
"""
# (a)First, let's define a customized dataset.
class HandGestureDataset(Dataset):
    """
    Args:
        text_file (string): Path to the dataset file.
        trans (callable, optional): Optional transform to be applied
            on a sample.
    """

    def __init__(self, text_file, trans=None):
        super(HandGestureDataset, self).__init__()
        self.data = pd.read_csv(text_file, header=None)
        self.trans = trans

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[0][idx].split(' ')
        img = io.imread(img_path)
        label = int(label)
        if len(img.shape) < 3:
            img = img.reshape(img.shape[0], img.shape[1], 1)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = img.reshape(img.shape[0], img.shape[1], 1)
        sample = {'img': img, 'label': label}

        if self.trans:
            sample = self.trans(sample)

        return sample

# (b)Secondly, define transforms.
class Resize(object):
    """
    Resize the image in a sample to a given size.
    Args:
        output_size (int): Return image size as output_size * output_size.
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        img = sample['img']
        img = transform.resize(img, (self.output_size, self.output_size), mode='constant')

        return {'img': img, 'label': sample['label']}

class Normalize(object):
    """
    Transform image to normalized range [-1, 1].
    Args:
        mean (float): same mean for every channel.
        std (float): same std for every channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['img']
        img = (img - self.mean) / self.std

        return {'img': img, 'label': sample['label']}

class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """

    def __call__(self, sample):
        img = sample['img']
        img = img.transpose((2, 0, 1))
        return {'img': torch.from_numpy(img).type(torch.DoubleTensor),
                'label': sample['label']}

trans = transforms.Compose(
    [Resize(200),
     ToTensor(),
     Normalize(0.5, 0.5)])

# (c)Finally, create dataset.
trainset = HandGestureDataset('./gittrain.txt', trans=trans)


"""
3. Define network.
"""
class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout2d()
        )

        self.fc = nn.Sequential(
            nn.Linear(80000, 128),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Dropout2d(),
            nn.Linear(128, 5)
        )

    def forward(self, img):
        # print img.size(), img.data[0][0][20:27, 20:27]
        out = self.conv(img)
        # print out.size(), out.data[0][0][:7, :7]
        out = out.view(-1, 80000)
        # print out.size(), out.data[0][20:26]
        out = self.fc(out)
        # print out.size(), out.data[0]

        out = F.softmax(out)
        # print 'Output of net:'
        # print str(out.data.numpy())

        return out


"""
4. Define loss function and optimizer.
"""
criterion = nn.CrossEntropyLoss()


"""
5. Train network and apply cross-validation on training data.
"""
if mode == '--save':
    np.random.seed(3)
    best_record = 0
    best_acc = []
    best_model_loss = []
    best_val_acc = []
    best_val_model_loss = []
    # Calculate the amount of data needed for validation.
    num_train = len(trainset)
    indices = list(range(len(trainset)))
    split = int(np.floor(0.1 * num_train))
    for val_split in range(times_val_split):
        # Create cross-validation split.
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # Load train data and validation data.
        trainloader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler,
                                 num_workers=num_workers, pin_memory=use_gpu)
        validloader = DataLoader(trainset, batch_size=batch_size, sampler=valid_sampler,
                                 num_workers=num_workers, pin_memory=use_gpu)

        # Generate a new CNN.
        cnn = Cnn()
        cnn.double()
        if use_gpu:
            cnn.cuda()

        # Create optimizer.
        optimizer = optim.Adam(cnn.parameters())

        # Record accuracy and loss on training set and validation set for the current network.
        acc = []
        model_loss = []
        val_acc = []
        val_model_loss = []
        for epoch in range(num_epochs):
            # Training.
            running_acc = 0
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                print 'Training Epoch:', epoch, 'Batch:', i
                # Get the inputs.
                imgs = data['img']
                labels = data['label']

                # Wrap them in Variable.
                if use_gpu:
                    imgs, labels = Variable(imgs.cuda()), Variable(labels.type(torch.LongTensor).cuda())
                else:
                    imgs, labels = Variable(imgs), Variable(labels.type(torch.LongTensor))

                # Zero the parameter gradients.
                optimizer.zero_grad()

                # Forward, backward and optimize.
                outputs = cnn(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Record loss.
                running_loss += loss.data[0]

                # Calculate accuracy.
                outputs = outputs.data.cpu().numpy()
                outputs = outputs.argmax(1)
                labels = labels.data.cpu().numpy()
                running_acc += np.sum(outputs == labels)

            # Print loss and accuracy for this epoch.
            model_loss.append(running_loss / (num_train - np.floor(0.1 * num_train)))
            acc.append(running_acc * 100.0 / (num_train - np.floor(0.1 * num_train)))
            print ('Val %d, Epoch %d' % (val_split, epoch))
            print ('loss: %.4f' % (running_loss / (num_train - np.floor(0.1 * num_train))))
            print 'acc:', running_acc
            print 'Total', num_train - np.floor(0.1 * num_train)
            print running_acc * 100.0 / (num_train - np.floor(0.1 * num_train))
            print '*************************************'
            running_loss = 0.0
            running_acc = 0

            # Validation.
            val_running_acc = 0
            val_running_loss = 0.0
            for i, data in enumerate(validloader, 0):
                print 'Validation Epoch:', epoch, 'Batch', i
                # Get the inputs.
                imgs = data['img']
                labels = data['label']

                # Wrap them in Variable.
                if use_gpu:
                    imgs, labels = Variable(imgs.cuda()), Variable(labels.type(torch.LongTensor).cuda())
                else:
                    imgs, labels = Variable(imgs), Variable(labels.type(torch.LongTensor))

                # Forward, backward and optimize.
                outputs = cnn(imgs)
                loss = criterion(outputs, labels)

                # Record loss.
                val_running_loss += loss.data[0]

                # Calculate accuracy.
                outputs = outputs.data.cpu().numpy()
                outputs = outputs.argmax(1)
                labels = labels.data.cpu().numpy()
                val_running_acc += np.sum(outputs == labels)

            # Print loss and accuracy for this epoch.
            val_model_loss.append(val_running_loss / np.floor(0.1 * num_train))
            val_acc.append(val_running_acc * 100.0 / np.floor(0.1 * num_train))
            print ('Val %d, Epoch %d' % (val_split, epoch))
            print ('loss: %.4f' % (val_running_loss / np.floor(0.1 * num_train)))
            print 'acc:', val_running_acc
            print 'Total', np.floor(0.1 * num_train)
            print val_running_acc * 100.0 / np.floor(0.1 * num_train)
            print '*************************************'
            val_running_loss = 0.0
            val_running_acc = 0

        # Record this model if it outperforms its predecessors.
        if val_acc[-1] > best_record:
            best_record = val_acc[-1]
            best_acc = acc
            best_model_loss = model_loss
            best_val_acc = val_acc
            best_val_model_loss = val_model_loss
            print 'Current model is better!'
            print 'Saving loss.'
            thefile = open('loss.txt', 'w')
            thefile.write(str(model_loss))
            thefile = open('val_loss.txt', 'w')
            thefile.write(str(val_model_loss))
            print 'Saving model.'
            torch.save(cnn.state_dict(), './' + weights_file_name + '.pt')
            print '*************************************'


"""
6. Hand gesture recognition.
"""
meaning = ['OK', 'Stop', 'Punch', 'Peace', 'Nothing']
if mode == '--load':
    # Generate a new CNN.
    cnn = Cnn()
    cnn.double()
    if use_gpu:
        cnn.cuda()
    cnn.load_state_dict(torch.load('./' + weights_file_name + '.pt', map_location=lambda storage, loc: storage))
    cnn.eval()

    ###############################################
    # Capture video image
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.namedWindow('Bigger', cv2.WINDOW_NORMAL)
    # set rt size as 1080x540
    ret = cap.set(3, 1080)
    ret = cap.set(4, 540)
    col = 77
    row = 233
    width = 400
    height = 400

    count = 0
    images = torch.DoubleTensor(1, 1, 200, 200)
    text = "Nothing"
    while (True):
        count += 1
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        roi = frame[row:row+height, col:col+width]
        roi = binaryMask(roi)

        image = reSize(roi, 200)
        image = toTensor(image)
        image = normalize(image, 0.5, 0.5)
        image = image.resize_(1, 1, 200, 200)
        images[count - 1] = image

        # if count % 3 == 0:
        images = Variable(images)

        out_img = cnn(images)
        # print out_img.data.numpy()
        out_img = out_img.data.numpy()
        out_img = out_img.argmax(1)
        text = meaning[stats.mode(out_img, axis=None)[0][0]]
        count = 0
        images = torch.DoubleTensor(1, 1, 200, 200)

        if text == "Peace":
           pyautogui.press('space')
        cv2.putText(frame, text, (77, 200), font, 1.0, (0, 0, 255), 2, 1)
        cv2.rectangle(frame, (col, row), (col + width, row + height), (0, 255, 0), 1)
        cv2.imshow('frame', frame)
        cv2.imshow('roi', roi)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    ###############################################

    # testset = HandGestureDataset('./new_test.txt', trans=trans)
    # testloader = DataLoader(testset, batch_size=1, num_workers=num_workers, pin_memory=use_gpu)
    #
    # # Validation.
    # val_running_acc = 0
    # val_running_loss = 0.0
    # for i, data in enumerate(testloader, 0):
    #     print 'Batch', i
    #     # Get the inputs.
    #     imgs = data['img']
    #     labels = data['label']
    #
    #     # Wrap them in Variable.
    #     if use_gpu:
    #         imgs, labels = Variable(imgs.cuda()), Variable(labels.type(torch.LongTensor).cuda())
    #     else:
    #         imgs, labels = Variable(imgs), Variable(labels.type(torch.LongTensor))
    #
    #     # Forward, backward and optimize.
    #     outputs = cnn(imgs)
    #     loss = criterion(outputs, labels)
    #
    #     # Record loss.
    #     val_running_loss += loss.data[0]
    #
    #     # Calculate accuracy.
    #     outputs = outputs.data.cpu().numpy()
    #     outputs = outputs.argmax(1)
    #     labels = labels.data.cpu().numpy()
    #     print 'outputs', outputs
    #     print 'labels', labels
    #     print '*************************************'
    #     val_running_acc += np.sum(outputs == labels)
    #
    # print ('loss: %.4f' % (val_running_loss / len(testset)))
    # print 'acc:', val_running_acc
    # print 'Total', len(testset)
    # print val_running_acc * 100.0 / len(testset)
