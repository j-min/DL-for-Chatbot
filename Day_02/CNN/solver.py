import torch
from torch.autograd import Variable
from models import CNN

from tqdm import tqdm
import os
import numpy as np


class Solver(object):
    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader

    def build(self, is_train):
        self.model = CNN(self.config)
        self.loss_fn = self.config.loss_fn()

        if is_train:
            self.model.train()
            self.optimizer = self.config.optimizer(self.model.parameters(), lr=self.config.lr)
        else:
            self.model.eval()

    def save(self, ckpt_path):
        """Save model parameters"""
        print('Save parameters at ', ckpt_path)
        torch.save(self.model.state_dict(), ckpt_path)

    def load(self, ckpt_path=None, epoch=None):
        """Load model parameters"""
        if not (ckpt_path or epoch):
            epoch = self.config.epochs
        if epoch:
            ckpt_path = os.path.join(self.config.save_dir, f'epoch-{epoch}.pkl')
        print('Load parameters from ', ckpt_path)
        self.model.load_state_dict(torch.load(ckpt_path))

    def train(self):
        """Train model with training data"""
        for epoch in tqdm(range(self.config.epochs)):
            loss_history = []

            for batch_i, batch in enumerate(tqdm(self.data_loader)):
                # text: [max_seq_len, batch_size]
                # label: [batch_size]
                text, label = batch.text, batch.label

                # [batch_size, max_seq_len]
                text.data.t_()

                # [batch_size, 2]
                logit = self.model(text)

                # Calculate loss
                average_batch_loss = self.loss_fn(logit, label)  # [1]
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
                self.save(ckpt_path)

    def eval(self):
        """Evaluate model from text data"""

        n_total_data = 0
        n_correct = 0
        loss_history = []
        import ipdb
        ipdb.set_trace()
        for _, batch in enumerate(tqdm(self.data_loader)):
            # text: [max_seq_len, batch_size]
            # label: [batch_size]
            text, label = batch.text, batch.label

            # [batch_size, max_seq_len]
            text.data.t_()

            # [batch_size, 2]
            logit = self.model(text)

            # Calculate loss
            average_batch_loss = self.loss_fn(logit, label)  # [1]
            loss_history.append(average_batch_loss.data[0])  # Variable -> Tensor

            # Calculate accuracy
            n_total_data += len(label)

            # [batch_size]
            _, prediction = logit.max(1)

            n_correct += (prediction == label).sum().data

        epoch_loss = np.mean(loss_history)

        accuracy = n_correct / n_total_data

        print(f'Loss: {epoch_loss:.2f}')

        print(f'Accuracy: {accuracy}')

    def inference(self, text):

        text = Variable(torch.LongTensor([text]))

        # [batch_size, 2]
        logit = self.model(text)

        _, prediction = torch.max(logit)

        return prediction
