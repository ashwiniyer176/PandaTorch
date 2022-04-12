import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim


class TorchModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = None
        self.train_loader = None
        self.valid_loader = None
        self.scheduler = None

    def fit(self, train_dataset, batch_size, epochs, device, optimizer):
        if self.optimizer is None:
            self.optimizer = optimizer
        if self.train_loader is None:
            self.train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
        if next(self.parameters()).device != device:
            self.to(device)
        losses = []
        for e in range(epochs):
            epoch_loss = self.train_one_epoch(self.train_loader, device)
            losses.append(epoch_loss.item())
            print(f"Epoch {e+1} of {epochs}\n Loss:{epoch_loss}")
        return losses

    def train_one_epoch(self, data_loader, device):
        self.train()
        epoch_loss, count = 0, 0
        for data in data_loader:
            count += 1
            loss = self.train_one_step(data, device)
            epoch_loss += loss
        return epoch_loss / count

    def train_one_step(self, data, device):
        self.optimizer.zero_grad()
        features, targets = data
        features, targets = features.to(device), targets.to(device)
        _, loss = self(features, targets)
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return loss

    def forward(self, *args, **kwargs):
        super().forward(*args, **kwargs)
