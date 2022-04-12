from collections import defaultdict
from pandatorch.model import TorchModel
from pandatorch import data
import pandas as pd
from torch import nn
from sklearn import metrics
import torch

device = "cpu"
df = pd.read_csv("tests/IRIS.csv")

torch_df = data.DataFrame(df.drop(columns=["petal_length", "species"]), df["species"])


class SampleModel(TorchModel):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(
            in_features=torch_df.get_number_of_columns(torch_df.features),
            out_features=5,
        )
        self.output = nn.Linear(5, 3)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def loss(self, outputs, targets):
        if targets is None:
            return None
        criterion = nn.CrossEntropyLoss().to(device)
        return criterion(outputs, targets)

    def forward(self, x, targets=None):
        x = self.relu(self.l1(x))
        outputs = self.softmax(self.output(x))
        loss = self.loss(outputs, targets)
        return outputs, loss


model = SampleModel()
optimizer = torch.optim.Adam(model.parameters())


def test_fit_returns_list():
    losses = model.fit(
        torch_df,
        batch_size=1,
        epochs=1,
        device="cuda:0",
        optimizer=optimizer,
        metrics=None,
    )
    assert type(losses) == type([])


def test_losses_returned_are_float():
    losses = model.fit(
        torch_df,
        batch_size=1,
        epochs=1,
        device="cuda:0",
        optimizer=optimizer,
        metrics=None,
    )
    assert type(losses[0]) == float


def test_evaluate_function():
    model.evaluate(
        "batch_",
        torch.tensor([1, 2, 3, 4, 5]),
        torch.tensor([0, 0, 0, 0, 0]),
        {"R2": metrics.r2_score},
    )
    for k, v in model.metrics.items():
        assert type(k) == str
        assert type(v) == list
