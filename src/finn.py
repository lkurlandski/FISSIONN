"""
FINN: Fingerprinting Network Flows with Neural Networks.

TODO:
- Maybe the encoder would work better if it also had the IPD to operate on?
"""

from dataclasses import dataclass
from pprint import pprint
import random
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
from torch import nn, Tensor
from torch.distributions import Laplace
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def random_one_hot_vector(length: int) -> Tensor:
    index = torch.randint(0, length, (1,)).item()
    vector = torch.zeros(length)
    vector[index] = 1
    return vector


class FINNDataset(Dataset):

    def __init__(self, fingerprint_length: int, flow_length: int) -> None:
        self.fingerprint_length = fingerprint_length
        self.flow_length = flow_length
        self.distribution = Laplace(0, 0.75)

    def __len__(self) -> int:
        return 100000

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
          idx: Index of the sample to retrieve.

        Return:
          fingerprint: The fingerprint of the flow.
          ipd: The inter-packet delay of the flow.
          fingerprint_delay: The fingerprint delay of the flow.
          noise: The noise to add to the flow.
        """
        fingerprint = random_one_hot_vector(self.fingerprint_length)
        ipd = torch.rand(self.flow_length)
        fingerprint_delay = torch.rand(self.flow_length)
        noise = self.distribution.sample((self.flow_length,))
        return fingerprint, ipd, fingerprint_delay, noise


class FINNCollateFn:

    def __call__(self, *args, **kwds) -> Any:
        return args, kwds


class FINNLoss(nn.Module):

    def __init__(
        self,
        encoder_weight: float,
        decoder_weight: float,
    ) -> None:
        super().__init__()
        self.encoder_weight = encoder_weight
        self.decoder_weight = decoder_weight

    def forward(
        self,
        encoder_logits: Tensor,
        encoder_labels: Tensor,
        decoder_logits: Tensor,
        decoder_labels: Tensor,
    ) -> Tensor:
        encoder_loss = F.l1_loss(encoder_logits, encoder_labels)
        decoder_loss = F.cross_entropy(decoder_logits, decoder_labels)
        return self.encoder_weight * encoder_loss + self.decoder_weight * decoder_loss


@dataclass
class FINNModelOutput:
    delay: Tensor
    fingerprint_prediction: Tensor


class FINNEncoder(nn.Module):

    def __init__(self, fingerprint_length: int, flow_length: int) -> None:
        super().__init__()
        self.layer_1 = nn.Linear(fingerprint_length, 1000)
        self.layer_2 = nn.Linear(1000, 2000)
        self.layer_3 = nn.Linear(2000, 2000)
        self.layer_4 = nn.Linear(2000, 500)
        self.layer_5 = nn.Linear(500, flow_length)

    def forward(self, fingerprint: Tensor) -> Tensor:
        x = self.layer_1(fingerprint)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)
        x = F.relu(x)
        x = self.layer_4(x)
        x = F.relu(x)
        x = self.layer_5(x)
        return x


class FINNDecoder(nn.Module):

    def __init__(self, fingerprint_length: int, flow_length: int) -> None:
        super().__init__()
        self.conv_1 = nn.Conv1d(1, 50, 10, stride=1)
        self.conv_2 = nn.Conv1d(50, 10, 10, stride=1)
        self.mlp_1 = nn.Linear(820, 256)
        self.mlp_2 = nn.Linear(256, 256)

    def forward(self, noisy_marked_ipd: Tensor) -> Tensor:
        x = noisy_marked_ipd.unsqueeze(1)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = x.flatten(start_dim=1)
        x = self.mlp_1(x)
        x = F.relu(x)
        x = self.mlp_2(x)
        return x


class FINNModel(nn.Module):

    def __init__(self, fingerprint_length: int, flow_length: int) -> None:
        super().__init__()
        self.encoder = FINNEncoder(fingerprint_length, flow_length)
        self.decoder = FINNDecoder(fingerprint_length, flow_length)

    def forward(self, fingerprint: Tensor, ipd: Tensor, noise: Tensor) -> FINNModelOutput:
        if any(x.dim() != 2 for x in (fingerprint, ipd, noise)):
            raise ValueError("Expected 2D tensors.")

        delay: Tensor = self.encoder(fingerprint)
        noisy_marked_ipd = delay + ipd + noise
        fingerprint_prediction = self.decoder(noisy_marked_ipd)

        return FINNModelOutput(delay, fingerprint_prediction)


@dataclass
class FINNTrainerArgs:
    num_train_epochs: int = 1
    batch_size: int = 1
    dataloader_num_workers: int = 0
    device: torch.cuda.device = torch.device("cpu")


class FINNTrainer:

    def __init__(
        self,
        args: FINNTrainerArgs,
        model: FINNModel,
        tr_dataset: FINNDataset,
        vl_dataset: FINNDataset,
    ) -> None:
        self.args = args
        self.model = model
        self.model.to(self.args.device)
        self.tr_dataset = tr_dataset
        self.vl_dataset = vl_dataset

        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = FINNLoss(encoder_weight=0.5, decoder_weight=0.5)
        self.collate_fn = None

    def __call__(self) -> None:
        for epoch in tqdm(range(1, self.args.num_train_epochs + 1), desc="Epochs"):
            tr_loss = self.train()
            vl_loss, metrics = self.evaluate()
            d = {"epoch": epoch, "tr_loss": tr_loss, "vl_loss": vl_loss} | metrics
            d = {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in d.items()}
            pprint(d)

    def train(self) -> float:
        self.model.train()
        cum_loss, n_samples = 0, 0
        dataloader = self.get_dataloader(self.tr_dataset, shuffle=True)
        pbar = tqdm(dataloader, total=len(self.tr_dataset) // self.args.batch_size)
        for batch in pbar:

            fingerprint: Tensor = batch[0].to(self.args.device)
            ipd: Tensor = batch[1].to(self.args.device)
            fingerprint_delay: Tensor = batch[2].to(self.args.device)
            noise: Tensor = batch[3].to(self.args.device)

            self.model.zero_grad()
            output = self.model.forward(fingerprint, ipd, noise)
            loss: Tensor = self.loss_fn.forward(output.delay, fingerprint_delay, output.fingerprint_prediction, fingerprint)
            loss.backward()
            self.optimizer.step()

            cum_loss += loss.item()
            n_samples += fingerprint.size(0)

        return cum_loss / n_samples

    def evaluate(self) -> tuple[float, dict[str, float]]:
        self.model.eval()
        cum_loss, y_true, y_pred = 0, [], []
        dataloader = self.get_dataloader(self.vl_dataset, shuffle=False)
        pbar = tqdm(dataloader, total=len(self.vl_dataset) // self.args.batch_size)
        for batch in pbar:

            fingerprint: Tensor = batch[0].to(self.args.device)
            ipd: Tensor = batch[1].to(self.args.device)
            fingerprint_delay: Tensor = batch[2].to(self.args.device)
            noise: Tensor = batch[3].to(self.args.device)
            
            output = self.model.forward(fingerprint, ipd, noise)
            loss: Tensor = self.loss_fn.forward(output.delay, fingerprint_delay, output.fingerprint_prediction, fingerprint)
            predictions = F.softmax(output.fingerprint_prediction, dim=-1)
            cum_loss += loss.item()
            y_true.extend(fingerprint.tolist())
            y_pred.extend(predictions.tolist())

        # metrics = {
        #     "accuracy": accuracy_score(y_true, y_pred),
        #     "f1": f1_score(y_true, y_pred),
        #     "roc_auc": roc_auc_score(y_true, y_pred),
        # }
        metrics = {}

        return cum_loss / len(y_pred), metrics

    def predict(self) -> None:
        ...

    def save(self) -> None:
        ...

    def get_dataloader(self, dataset: FINNDataset, shuffle: bool = False) -> None:
        return DataLoader(
            dataset,
            self.args.batch_size,
            shuffle,
            num_workers=self.args.dataloader_num_workers,
            collate_fn=self.collate_fn,
        )


def main():
    seed_everything(0)

    fingerprint_length = 256
    flow_length = 100

    dataset = FINNDataset(fingerprint_length=fingerprint_length, flow_length=flow_length)
    tr_dataset, ts_dataset = random_split(dataset, [0.80, 0.20])
    model = FINNModel(fingerprint_length, flow_length)
    args = FINNTrainerArgs(num_train_epochs=64, dataloader_num_workers=4, batch_size=256, device=torch.device("cuda:0"))
    trainer = FINNTrainer(args, model, tr_dataset, ts_dataset)
    trainer()


if __name__ == "__main__":
    main()
