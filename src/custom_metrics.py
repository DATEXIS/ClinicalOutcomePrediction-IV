import logging
from typing import Any, Callable, List, Literal, Optional

import numpy as np
import torch
import torchmetrics
from torchmetrics.classification.auroc import AUROC
from torchmetrics.classification.precision_recall_curve import PrecisionRecallCurve
from torchmetrics.functional.classification.auroc import _multilabel_auroc_compute, multilabel_auroc

from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat


class PR_AUC(Metric):
    prauc: torch.Tensor

    def __init__(self, num_classes, compute_on_step=False, dist_sync_on_step=False):
        super().__init__(compute_on_step=compute_on_step,
                         dist_sync_on_step=dist_sync_on_step)
        self.add_state("prauc", default=[], dist_reduce_fx="cat")
        self.pr_curve = PrecisionRecallCurve(
            num_classes=num_classes).to(self.device)
        self.auc = torchmetrics.AUC().to(self.device)

    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        precision, recall, thresholds = self.pr_curve(prediction, target)
        auc_values = [self.auc(r, p) for r, p in zip(recall, precision)]

        pr_auc = torch.mean(torch.tensor(
            [v for v in auc_values if not v.isnan()])).to(self.device)
        self.prauc += [pr_auc.detach()]

    def compute(self):
        self.prauc = torch.as_tensor(self.prauc)
        return torch.mean(self.prauc)


class PR_AUCPerBucket(PR_AUC):
    def __init__(self, num_classes, bucket, compute_on_step=False, dist_sync_on_step=False):
        super().__init__(
            num_classes=len(bucket),
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
        )
        self.bucket = set(bucket)
        self.num_classes = num_classes

    def update(self, prediction: torch.Tensor, target: torch.Tensor):

        mask = np.zeros((self.num_classes), dtype=bool)
        for c in range(self.num_classes):
            if c in self.bucket:
                mask[c] = True
        filtered_target = target[:, mask]
        filtered_preds = prediction[:, mask]

        if len((filtered_target > 0).nonzero()) > 0:
            precision, recall, thresholds = self.pr_curve(
                filtered_preds, filtered_target)
            auc_values = [self.auc(r, p) for r, p in zip(recall, precision)]

            pr_auc = torch.mean(torch.tensor([v for v in auc_values if not v.isnan()])).to(
                self.device
            )
            self.prauc += [pr_auc.detach()]


def calculate_pr_auc(prediction: torch.Tensor, target: torch.Tensor, num_classes, device):
    pr_curve = PrecisionRecallCurve(num_classes=num_classes).to(device)
    auc = torchmetrics.AUC().to(device)

    precision, recall, thresholds = pr_curve(prediction, target)
    auc_values = [auc(r, p) for r, p in zip(recall, precision)]

    pr_auc = torch.mean(torch.tensor(
        [v for v in auc_values if not v.isnan()])).to(device)
    return pr_auc.detach()


class FilteredAUROC(torchmetrics.classification.auroc.MultilabelAUROC):
        
    def compute(self) -> torch.Tensor:

        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)

        # mask = np.ones((self.num_classes), dtype=bool)
        # breakpoint()
        # for c in range(self.num_classes):
        #     if torch.max(target[:, c]) == 0:
        #         mask[c] = False
        mask = ((target.sum(axis=0) > 0) +
                (target.sum(axis=0) == len(target))).cpu().numpy()
        filtered_target = target[:, mask]
        filtered_preds = preds[:, mask]

        num_filtered_cols = np.count_nonzero(mask == False)  # noqa
        logging.info(
            f"{num_filtered_cols} columns not considered for ROC AUC calculation!")

        return multilabel_auroc(
            filtered_preds,
            filtered_target,
            self.num_labels - num_filtered_cols,
            self.average,
        )

class FilteredAUROCPerBucket(AUROC):
    num_classes: int

    def __init__(
        self,
        bucket: List[int],
        num_classes: Optional[int] = None,
        pos_label: Optional[int] = None,
        average: Optional[Literal["macro", "weighted", "none"]] = "macro",
        max_fpr: Optional[float] = None,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Optional[Callable] = None,
    ):
        super().__init__(
            num_classes,
            pos_label,
            average,
            max_fpr,
            compute_on_step,  # type: ignore
            dist_sync_on_step,
            process_group,
            dist_sync_fn,  # type: ignore
        )
        self.bucket = set(bucket)

    def compute(self) -> torch.Tensor:

        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)

        mask = np.zeros((self.num_classes), dtype=bool)
        for c in range(self.num_classes):
            if torch.max(target[:, c]) > 0 and c in self.bucket:
                mask[c] = True
        filtered_target = target[:, mask]
        filtered_preds = preds[:, mask]

        num_filtered_cols = np.count_nonzero(mask == False)  # noqa
        logging.info(
            f"{num_filtered_cols} columns not considered for ROC AUC calculation!")

        return _auroc_compute(
            filtered_preds,
            filtered_target,
            self.mode,
            self.num_classes - num_filtered_cols,
            self.pos_label,
            self.average,
            self.max_fpr,
        )