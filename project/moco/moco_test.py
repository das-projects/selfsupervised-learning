from argparse import ArgumentParser
from typing import Any, Dict, List, Tuple, Type
import os
from copy import deepcopy

import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
import pytorch_lightning as pl
import pl_bolts
from torch import Tensor, nn
from torch.nn import functional as F
import torch.utils.data as data
from torch.nn.functional import softmax
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torchmetrics.functional import accuracy
from torchvision.datasets import STL10
from torchvision import transforms
from tqdm.notebook import tqdm
from moco2_module import Moco_v2

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# In this notebook, we use data loaders with heavier computational processing. It is recommended to use as many
# workers as possible in a data loader, which corresponds to the number of CPU cores
NUM_WORKERS = os.cpu_count()


class LogisticRegression(LightningModule):
    """Logistic regression model."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        bias: bool = True,
        learning_rate: float = 1e-4,
        optimizer: Type[Optimizer] = Adam,
        l1_strength: float = 0.0,
        l2_strength: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            input_dim: number of dimensions of the input (at least 1)
            num_classes: number of class labels (binary: 2, multi-class: >2)
            bias: specifies if a constant or intercept should be fitted (equivalent to fit_intercept in sklearn)
            learning_rate: learning_rate for the optimizer
            optimizer: the optimizer to use (default: ``Adam``)
            l1_strength: L1 regularization strength (default: ``0.0``)
            l2_strength: L2 regularization strength (default: ``0.0``)
        """
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer

        self.linear = nn.Linear(in_features=self.hparams.input_dim, out_features=self.hparams.num_classes, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        y_hat = softmax(x)
        return y_hat

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        x, y = batch

        # flatten any input
        x = x.view(x.size(0), -1)

        y_hat = self.linear(x)

        # PyTorch cross_entropy function combines log_softmax and nll_loss in single function
        loss = F.cross_entropy(y_hat, y, reduction="sum")

        # L1 regularizer
        if self.hparams.l1_strength > 0:
            l1_reg = self.linear.weight.abs().sum()
            loss += self.hparams.l1_strength * l1_reg

        # L2 regularizer
        if self.hparams.l2_strength > 0:
            l2_reg = self.linear.weight.pow(2).sum()
            loss += self.hparams.l2_strength * l2_reg

        loss /= x.size(0)

        tensorboard_logs = {"train_ce_loss": loss}
        progress_bar_metrics = tensorboard_logs
        return {"loss": loss, "log": tensorboard_logs, "progress_bar": progress_bar_metrics}

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.linear(x)
        acc = accuracy(F.softmax(y_hat, -1), y)
        return {"val_loss": F.cross_entropy(y_hat, y), "acc": acc}

    def validation_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        acc = torch.stack([x["acc"] for x in outputs]).mean()
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_ce_loss": val_loss, "val_acc": acc}
        progress_bar_metrics = tensorboard_logs
        return {"val_loss": val_loss, "log": tensorboard_logs, "progress_bar": progress_bar_metrics}

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.linear(x)
        acc = accuracy(F.softmax(y_hat, -1), y)
        return {"test_loss": F.cross_entropy(y_hat, y), "acc": acc}

    def test_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        acc = torch.stack([x["acc"] for x in outputs]).mean()
        test_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        tensorboard_logs = {"test_ce_loss": test_loss, "test_acc": acc}
        progress_bar_metrics = tensorboard_logs
        return {"test_loss": test_loss, "log": tensorboard_logs, "progress_bar": progress_bar_metrics}

    def configure_optimizers(self) -> Optimizer:
        return self.optimizer(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=0.0001)
        parser.add_argument("--input_dim", type=int, default=None)
        parser.add_argument("--num_classes", type=int, default=None)
        parser.add_argument("--bias", default="store_true")
        parser.add_argument("--batch_size", type=int, default=16)
        return parser


@torch.no_grad()
def prepare_data_features(model, dataset):
    # Prepare model
    network = deepcopy(model.encoder_q)
    # network.fc = nn.Identity()  # Removing projection head g(.)
    network.eval()
    network.to(device)

    # Encode all images
    data_loader = data.DataLoader(dataset, batch_size=64, num_workers=NUM_WORKERS, shuffle=False, drop_last=False)
    feats, labels = [], []
    for batch_imgs, batch_labels in tqdm(data_loader):
        batch_imgs = batch_imgs.to(device)
        batch_feats = network(batch_imgs)
        feats.append(batch_feats.detach().cpu())
        labels.append(batch_labels)

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)

    # Sort images by labels
    labels, idxs = labels.sort()
    feats = feats[idxs]

    return data.TensorDataset(feats, labels)


def cli_main() -> None:
    from pl_bolts.datamodules.sklearn_datamodule import SklearnDataModule
    from pl_bolts.utils import _SKLEARN_AVAILABLE
    # Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)

    DATASET_PATH = "../data"
    # Path to the folder where the pretrained models are saved
    CHECKPOINT_PATH = "../saved_models/tutorial14"
    pretrained_filename = os.path.join(CHECKPOINT_PATH, 'SimCLR.ckpt')
    model = Moco_v2.load_from_checkpoint(pretrained_filename)
    # Setting the seed
    seed_everything(1234)

    img_transforms = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,))])

    train_img_data = STL10(root=DATASET_PATH, split='train', download=True,
                           transform=img_transforms)
    test_img_data = STL10(root=DATASET_PATH, split='test', download=True,
                          transform=img_transforms)

    print("Number of training examples:", len(train_img_data))
    print("Number of test examples:", len(test_img_data))

    train_feats = prepare_data_features(model, train_img_data)
    test_feats = prepare_data_features(model, test_img_data)
    # Data loaders
    train_loader = data.DataLoader(train_feats, batch_size=batch_size, shuffle=True,
                                   drop_last=False, pin_memory=True, num_workers=0)
    test_loader = data.DataLoader(test_feats, batch_size=batch_size, shuffle=False,
                                  drop_last=False, pin_memory=True, num_workers=0)

    model = LogisticRegression(**kwargs)
    trainer.fit(model, train_loader, test_loader)
    model = LogisticRegression.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    
    # args
    parser = ArgumentParser()
    parser = LogisticRegression.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # model
    # model = LogisticRegression(**vars(args))
    model = LogisticRegression(input_dim=4, num_classes=3, l1_strength=0.01, learning_rate=0.01)

    # data
    X, y = load_iris(return_X_y=True)
    loaders = SklearnDataModule(X, y, batch_size=args.batch_size, num_workers=0)

    # train
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, train_dataloader=loaders.train_dataloader(), val_dataloaders=loaders.val_dataloader())


if __name__ == "__main__":
    cli_main()