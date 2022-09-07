import sys, os
import logging

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from datetime import timedelta
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch.nn import Linear
import torch
import numpy as np

from .trigger_utils import build_dataset, load_dataset
from sklearn.metrics import roc_auc_score
from .gnn_base import GNNBase

class GraphLevelClassifierBase(GNNBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """

    def val_dataloader(self):
        if self.valset is not None:
            return DataLoader(
                self.valset, batch_size=100, num_workers=0
            )  # , pin_memory=True, persistent_workers=True)
        else:
            return None
   

    def training_step(self, batch, batch_idx):

        truth = batch[self.hparams["truth_key"]]
            
        sample_indices = torch.arange(batch.edge_index.shape[1])
        edge_sample, _, _ = self.handle_directed(batch, batch.edge_index, batch.y, sample_indices)

        weight = (
            torch.tensor(self.hparams["weight"])
            if ("weight" in self.hparams)
            else torch.tensor((~truth.bool()).sum() / truth.sum())
        )

        input_data = self.get_input_data(batch)
        output = self(input_data, edge_sample, batch.batch).squeeze() 

        positive_loss = F.binary_cross_entropy_with_logits(
            output[truth], torch.ones(truth.sum()).to(self.device)
        )

        negative_loss = F.binary_cross_entropy_with_logits(
            output[~truth], torch.zeros((~truth).sum()).to(self.device)
        )

        loss = positive_loss*weight + negative_loss

        self.log("train_loss", loss, on_step=False, on_epoch=True)

        return loss

    def log_metrics(self, output, batch, loss, log):

        preds = torch.sigmoid(output) > self.hparams["edge_cut"]

        # Positives
        graph_positive = preds.sum().float()

        # Signal true & signal tp
        sig_truth = batch[self.hparams["truth_key"]]
        sig_true = sig_truth.sum().float()
        sig_true_positive = (sig_truth.bool() & preds).sum().float()
        sig_auc = roc_auc_score(
            sig_truth.bool().cpu().detach(), torch.sigmoid(output).cpu().detach()
        )

        # Eff, pur, auc
        sig_eff = sig_true_positive / sig_true
        sig_pur = sig_true_positive / graph_positive
        
        
        if log:
            current_lr = self.optimizers().param_groups[0]["lr"]
            self.log_dict(
                {
                    "val_loss": loss,
                    "current_lr": current_lr,
                    "sig_eff": sig_eff,
                    "sig_pur": sig_pur,
                    "sig_auc": sig_auc,
                },
                sync_dist=True,
            )

        return preds

    def shared_evaluation(self, batch, batch_idx, log=True):

        truth = batch[self.hparams["truth_key"]]
        
        sample_indices = torch.arange(batch.edge_index.shape[1])
        edge_sample, _, _ = self.handle_directed(batch, batch.edge_index, batch.y, sample_indices)

        weight = (
            torch.tensor(self.hparams["weight"])
            if ("weight" in self.hparams)
            else torch.tensor((~truth.bool()).sum() / truth.sum())
        )
        
        input_data = self.get_input_data(batch)
        output = self(input_data, edge_sample, batch.batch).squeeze()

        positive_loss = F.binary_cross_entropy_with_logits(
            output[truth], torch.ones(truth.sum()).to(self.device)
        )

        negative_loss = F.binary_cross_entropy_with_logits(
            output[~truth], torch.zeros((~truth).sum()).to(self.device)
        )

        loss = positive_loss*weight + negative_loss

        preds = self.log_metrics(output, batch, loss, log)

        return {"loss": loss, "preds": preds, "score": torch.sigmoid(output)}

        
class AttentionDeficitBase(GraphLevelClassifierBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """

    def setup(self, stage):
    # Handle any subset of [train, val, test] data split, assuming that ordering

        if self.trainset is None:
            print("Setting up dataset")
            self.trainset, self.valset, self.testset = load_dataset(**self.hparams)

        if (
            (self.trainer)
            and ("logger" in self.trainer.__dict__.keys())
            and ("_experiment" in self.logger.__dict__.keys())
        ):
            self.logger.experiment.define_metric("val_loss", summary="min")
            self.logger.experiment.define_metric("sig_auc", summary="max")
            self.logger.experiment.log({"sig_auc": 0})