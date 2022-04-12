from collections import defaultdict
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch


class TorchModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = None
        self.train_loader = None
        self.valid_loader = None
        self.scheduler = None
        self.metrics = defaultdict(list)

    def fit(self, train_dataset, batch_size, epochs, device, optimizer, metrics):
        """
        Function to fit the model over the given dataset

        Args:
            train_dataset (torch.utils.data.Dataset|pandatoch.data.DataFrame): Training data as a PyTorch Dataset or PandaTorch DataFrame
            batch_size (int): Batch size for training
            epochs (int): Number of epochs to be trained
            device (string): Device for training. {'cuda:0','cpu'}
            optimizer : Any PyTorch Optimizer from the torch.nn subpackage
            metrics (dict): A dictionary of the format {'metric_name':metric_callable}. Currently uses scikit-learn metrics

        Returns:
            training_losses (list): A list of averaged epoch losses
        """
        if self.optimizer is None:
            self.optimizer = optimizer
        if self.train_loader is None:
            self.train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
        if next(self.parameters()).device != device:
            self.to(device)
        training_losses = []
        for e in range(epochs):
            epoch_loss = self.train_one_epoch(self.train_loader, device, metrics)
            training_losses.append(epoch_loss.item())
            print(f"Epoch {e+1} of {epochs}\n Loss:{epoch_loss}")
        return training_losses

    def train_one_epoch(self, data_loader, device, metrics):
        """
        Trains one epoch given a DataLoader and device

        Args:
            data_loader (torch,utils.data.DataLoader): A PyTorch DataLoader for the training dataset
            device (string): Device for training. {'cuda:0','cpu'}
            metrics (dict): A dictionary of the format {'metric_name':metric_callable}. Currently uses scikit-learn metrics

        Returns:
            epoch_loss(float): Batch loss averaged over an epoch
        """
        self.train()
        epoch_loss, count = 0, 0
        for data in data_loader:
            count += 1
            loss = self.train_one_step(data, device, metrics)
            epoch_loss += loss
        return epoch_loss / count

    def train_one_step(self, data, device, metrics):
        """
        Trains one batch, backpropagates the loss and optimizes the network

        Args:
            data (list): A list of the form [features, labels]
            device (string): Device for training. {'cuda:0','cpu'}
            metrics (dict): A dictionary of the format {'metric_name':metric_callable}. Currently uses scikit-learn metrics

        Returns:
            batch_loss (float): Batch loss for given batch
        """
        self.optimizer.zero_grad()
        features, targets = data
        features, targets = features.to(device), targets.to(device)
        outputs, batch_loss = self(features, targets)
        _, predictions = torch.max(outputs, 1)
        batch_loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.evaluate("batch_", predictions, targets, metrics)
        return batch_loss

    def convert_tensor_to_numpy(self, tensor):
        clone = tensor.detach().clone()
        clone = clone.cpu().numpy()
        return clone

    def evaluate(self, prefix, outputs, targets, metrics):
        if metrics is None:
            return
        predictions = self.convert_tensor_to_numpy(outputs)
        y_true = self.convert_tensor_to_numpy(targets)

        for metric, callable in metrics.items():
            self.metrics[prefix + metric].append(callable(y_true, predictions))

    def forward(self, *args, **kwargs):
        super().forward(*args, **kwargs)
