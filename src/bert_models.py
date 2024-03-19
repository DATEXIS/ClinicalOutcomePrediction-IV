from typing import Optional, Dict, Any, List, Union
import os
from custom_metrics import PR_AUC
import lightning.pytorch as pl
import torch
import torchmetrics
from torchmetrics.functional.classification.auroc import _multilabel_auroc_compute
from torchmetrics.functional.classification import auroc as auroc_f
from torchmetrics.functional.classification import average_precision, accuracy
import transformers
from transformers import AutoModelForSequenceClassification, BertForSequenceClassification, BertModel, AutoModel
from torchmetrics.functional.retrieval import retrieval_recall, retrieval_precision
import torch.nn.functional as F



class BertClassificationModel(pl.LightningModule):
    def __init__(self,
                 num_classes: int = 1617,
                 encoder_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                 warmup_steps: int = 0,
                 decay_steps: int = 50_000,
                 num_training_steps: int = 50_000,
                 weight_decay: float = 0.01,
                 lr: float = 2e-5,
                 optimizer_name="adam",
                 task: str = "dia",
                 ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_model_name)
        self.encoder.pooler = None
        self.task = task
        self.num_classes = num_classes
        self.classification_layer = torch.nn.Linear(768, num_classes)
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.num_training_steps = num_training_steps
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.auroc = None
        self.mean_precision = None
        self.val_output_list = []
        self.test_output_list = []
        self.running_allocated_memory = 0
        self.running_reserved_memory = 0

    def forward(self,
                input_ids,
                attention_mask):
        encoded = self.encoder(input_ids, attention_mask, return_dict=True)[
            'last_hidden_state'][:, 0]
        logits = self.classification_layer(encoded)
        return logits

    def training_step(self, batch, batch_idx):
        encoded = self.encoder(batch['input_ids'], batch['attention_mask'], return_dict=True)[
            'last_hidden_state'][:, 0]
        logits = self.classification_layer(encoded)
        if self.task in ['dia', 'pro', 'mp']:
            loss = F.binary_cross_entropy_with_logits(logits, batch['labels'])
        elif self.task in ['los', 'pr']:
            _, labels = torch.max(batch['labels'], dim=1)
            loss = F.cross_entropy(logits, labels)
        self.log("Train/Loss", loss)
        return loss

#    def on_train_epoch_end(self) -> None:
#        torch.cuda.empty_cache()

    def test_step(self, batch, batch_idx, **kwargs):
        encoded = self.encoder(batch['input_ids'], batch['attention_mask'], return_dict=True)[
            'last_hidden_state'][:, 0]
        logits = self.classification_layer(encoded)

        self.test_output_list.append(
            {"logits": logits, "labels": batch["labels"]})

        return {"logits": logits,
                "labels": batch["labels"], }

    def on_test_epoch_end(self) -> None:
        logits = torch.cat([x["logits"] for x in self.test_output_list])
        labels = torch.cat([x["labels"] for x in self.test_output_list]).int()
        if self.task in ['los', 'pr']:
            _, labels = torch.max(labels, dim=1)
            auroc_ = auroc_f(logits, labels, num_classes=self.num_classes, 
                                       task="multiclass", average='macro')
            mean_precision_ = average_precision(logits, labels, num_classes=self.num_classes, 
                                       task="multiclass", average='macro')
            accuracy_ = accuracy(logits, labels, task='multiclass', num_classes=self.num_classes,
                                    average='macro') 
            loss = F.cross_entropy(logits, labels)
        else:
            mask = ((labels.sum(dim=0) > 0) +
                    (labels.sum(dim=0) == len(labels)))
            filtered_target = labels[:, mask]
            filtered_preds = torch.sigmoid(logits[:, mask])

            auroc = torchmetrics.AUROC(num_labels=len(filtered_target.T),
                                    task="multilabel",
                                    average='macro')

            mean_precision = torchmetrics.AveragePrecision(num_labels=len(filtered_target.T),
                                                        task="multilabel",
                                                        average='macro')
            auroc_ = auroc(filtered_preds, filtered_target)
            mean_precision_ = mean_precision(filtered_preds, filtered_target)
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        
        self.log("Test/loss", loss)
        self.log("Test/AUROC", auroc_)
        self.log("Test/A_PREC", mean_precision_)
        self.test_output_list = list()

    def validation_step(self, batch, batch_idx, **kwargs):
        encoded = self.encoder(batch['input_ids'], batch['attention_mask'], return_dict=True)[
            'last_hidden_state'][:, 0]
        logits = self.classification_layer(encoded)

        self.val_output_list.append(
            {"logits": logits, "labels": batch["labels"]})

        return {"logits": logits,
                "labels": batch["labels"], }
    
    def on_validation_epoch_end(self) -> None:
        logits = torch.cat([x["logits"] for x in self.val_output_list])
        labels = torch.cat([x["labels"] for x in self.val_output_list]).int()
        if self.task in ['dia', 'pro']: 
            mask = ((labels.sum(dim=0) > 0) +
                    (labels.sum(dim=0) == len(labels)))
            filtered_target = labels[:, mask]
            filtered_preds = torch.sigmoid(logits[:, mask])
            auroc_ = auroc_f(filtered_preds, filtered_target, num_labels=len(
                    filtered_target.T), task="multilabel", average='macro')
            # tt_auroc = tt_auroc(torch.sigmoid(logits), labels)

            mean_precision = average_precision(filtered_preds, filtered_target,num_labels=len(
                    filtered_target.T), task="multilabel", average='macro')
            accuracy_ = accuracy(filtered_preds, filtered_target, task='multilabel', num_labels=len(
                    filtered_target.T), average='macro') 
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        elif self.task in ['los', 'pr']:
            _, labels = torch.max(labels, dim=1)
            auroc_ = auroc_f(logits, labels, num_classes=self.num_classes, 
                                       task="multiclass", average='macro')
            mean_precision = average_precision(logits, labels, num_classes=self.num_classes, 
                                       task="multiclass", average='macro')
            accuracy_ = accuracy(logits, labels, task='multiclass', num_classes=self.num_classes,
                                    average='macro') 
            loss = F.cross_entropy(logits, labels)
        elif self.task in ['mp']:
            auroc_ = auroc_f(logits, labels, num_classes=self.num_classes, 
                                       task="binary", average='macro')
            mean_precision = average_precision(logits, labels, num_classes=self.num_classes, 
                                       task="binary", average='macro')
            accuracy_ = accuracy(logits, labels, task='binary', num_classes=self.num_classes,
                                    average='macro') 
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        # ray.util.pdb.set_trace()
    
        
        self.log("Val/loss", loss)
        self.log("Val/AUROC", auroc_)
        self.log("Val/A_PREC", mean_precision)
        self.log("Val/BalancedAcc", accuracy_)
        self.val_output_list.clear()
        
    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        weight_decay = 0.01
        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
                weight_decay
        }, {
            'params':
                [p for n, p in param_optimizer if any(
                    nd in n for nd in no_decay)],
            'weight_decay':
                0.0
        }]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)

        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, self.warmup_steps, num_training_steps=self.num_training_steps)
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
        }

        return [optimizer], [scheduler]
