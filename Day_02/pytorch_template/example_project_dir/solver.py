import torch
from torch.autograd import Variable
from models import NNClassifier

from tqdm import tqdm
import os
import numpy as np


class Solver(object):
    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader

    def build(self, is_train):
        self.model = NNClassifier(self.config)
        self.loss_fn = self.config.loss_fn()

        if is_train:
            self.optimizer = self.config.optimizer(self.model.parameters(), lr=self.config.lr)

    def train(self):
        for epoch in tqdm(range(self.config.epochs)):
            loss_history = []
            for images, true_labels in self.data_loader:
                # images: [batch_size, 1, 28, 28]
                # true_labels: [batch_size]

                # Tensor -> Variable
                images = Variable(images)
                true_labels = Variable(true_labels)

                # Resize (for loss function)
                images = images.view(-1, 28 * 28)  # [batch_size, 1, 28, 28] => [batch_size, 28x28]
                true_labels = true_labels.view(-1)  # [batch_size, 1] => [batch_size]

                # [batch_size, 28x28] => [batch_size, 10]
                predicted_labels = self.model(images)

                # Calculate loss
                average_batch_loss = self.loss_fn(predicted_labels, true_labels)  # [1]
                loss_history.append(average_batch_loss.data[0])  # Variable -> Tensor

                # Flush out remaining gradient
                self.optimizer.zero_grad()

                # Backpropagation
                average_batch_loss.backward()

                # Gradient descent
                self.optimizer.step()

            # Log intermediate loss
            if (epoch + 1) % self.config.log_every_epoch == 0:
                epoch_loss = np.mean(loss_history)
                log_str = f'Epoch {epoch + 1} | loss: {epoch_loss:.2f}\n'
                print(log_str)

            # Save model parameters
            if (epoch + 1) % self.config.save_every_epoch == 0:
                ckpt_path = os.path.join(self.config.save_dir, f'epoch-{epoch+1}.pkl')
                print('Save parameters at ', ckpt_path)
                torch.save(self.model.state_dict(), ckpt_path)

    def eval(self, epoch=None):

        # Load model parameters
        if not isinstance(epoch, int):
            epoch = self.config.epochs
        ckpt_path = os.path.join(self.config.save_dir, f'epoch-{epoch}.pkl')
        print('Load parameters from ', ckpt_path)
        self.model.load_state_dict(torch.load(ckpt_path))

        loss_history = []
        for images, true_labels in self.data_loader:
            # images: [batch_size, 1, 28, 28]
            # true_labels: [batch_size]

            # Tensor -> Variable
            images = Variable(images)
            true_labels = Variable(true_labels)

            # Resize (for loss function)
            images = images.view(-1, 28 * 28)  # [batch_size, 1, 28, 28] => [batch_size, 28x28]
            true_labels = true_labels.view(-1)  # [batch_size, 1] => [batch_size]

            # [batch_size, 28x28] => [batch_size, 10]
            predicted_labels = self.model(images)

            # Calculate loss
            average_batch_loss = self.loss_fn(predicted_labels, true_labels)  # [1]
            loss_history.append(average_batch_loss.data[0])  # Variable -> Tensor

        epoch_loss = np.mean(loss_history)

        print('Loss: {epoch_loss:.2f}')
