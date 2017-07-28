import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms

import numpy as np
from tqdm import tqdm

#----------------------- Data -----------------------#
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),  # (0, 1) => (-0.5, 0.5) => (-1, 1)

])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),  # (0, 1) => (-0.5, 0.5) => (-1, 1)
])

train_dataset = datasets.MNIST(root='/Users/jmin/workspace/ml/datasets',
                               train=True, transform=train_transform)
test_dataset = datasets.MNIST(root='/Users/jmin/workspace/ml/datasets',
                              train=False, transform=test_transform)

train_dataloader = data.DataLoader(
    dataset=train_dataset, batch_size=100, shuffle=True)
test_dataloader = data.DataLoader(
    dataset=test_dataset, batch_size=100, shuffle=False)
#----------------------- Model -----------------------#


class NNClassifier(nn.Module):
    def __init__(self):
        super(NNClassifier, self).__init__()
        self.layer_1 = nn.Linear(28 * 28, 200)
        self.layer_2 = nn.Linear(200, 50)
        self.layer_3 = nn.Linear(50, 10)

        self.lrelu = nn.LeakyReLU()

        self.softmax = nn.Softmax()

        self.net = nn.Sequential(
            self.layer_1,  # 784 => 200
            self.lrelu,
            self.layer_2,  # 200 => 50
            self.lrelu,
            self.layer_3,  # 50 => 10
            self.softmax,
        )

    def forward(self, x):
        # [batch_size, 784] => [batch_size, 1]
        return self.net(x)


#----------------------- Build -----------------------#
classifier = NNClassifier()

loss_fn = nn.CrossEntropyLoss()
# Args:
#     Input: (batch_size, number of classes)
#     Target: (batch_size)

optimizer = optim.SGD(params=classifier.parameters(), lr=1e-3)


#----------------------- Train -----------------------#
print('Start training!\n')
for epoch in tqdm(range(20)):
    # epoch_loss = average of batch losses
    loss_history = []
    for images, true_labels in train_dataloader:
        # images: [batch_size, 1, 28, 28]
        # true_labels: [batch_size]

        # Tensor -> Variable
        images = Variable(images)
        true_labels = Variable(true_labels)

        # Resize (for loss function)
        images = images.view(-1, 28 * 28)  # [batch_size, 1, 28, 28] => [batch_size, 28x28]
        true_labels = true_labels.view(-1)  # [batch_size, 1] => [batch_size]

        # [batch_size, 28x28] => [batch_size, 10]
        predicted_labels = classifier(images)

        # Calculate loss
        average_batch_loss = loss_fn(predicted_labels, true_labels)  # [1]
        loss_history.append(average_batch_loss.data[0])  # Variable -> Tensor

        # Flush out remaining gradient
        optimizer.zero_grad()

        # Backpropagation
        average_batch_loss.backward()

        # Gradient descent
        optimizer.step()

    if (epoch + 1) % 2 == 0:
        epoch_loss = np.mean(loss_history)
        log_str = 'Epoch {} | loss: {:.2f}\n'.format(epoch + 1, epoch_loss)
        print(log_str)

#----------------------- Evaluation -----------------------#
print('Start Evaluation!\n')
test_loss_history = []
for images, true_labels in tqdm(test_dataloader):
    # images: [batch_size, 1, 28, 28]
    # true_labels: [batch_size]

    # Tensor -> Variable
    images = Variable(images)
    true_labels = Variable(true_labels)

    # Resize (for loss function)
    images = images.view(-1, 28 * 28)  # [batch_size, 1, 28, 28] => [batch_size, 28x28]
    true_labels = true_labels.view(-1)  # [batch_size, 1] => [batch_size]

    # [batch_size, 28x28] => [batch_size, 10]
    predicted_labels = classifier(images)

    # Calculate loss
    average_batch_loss = loss_fn(predicted_labels, true_labels)  # [1]
    test_loss_history.append(average_batch_loss.data[0])  # Variable -> Tensor

test_loss = np.mean(test_loss_history)
log_str = 'Test loss: {:.2f}\n'.format(test_loss)
print(log_str)
