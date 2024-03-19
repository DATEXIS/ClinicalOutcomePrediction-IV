
from lightning.pytorch.cli import LightningCLI
from bert_models import BertClassificationModel
from datamodule import MIMICClassificationDataModule


cli = LightningCLI(BertClassificationModel, MIMICClassificationDataModule)