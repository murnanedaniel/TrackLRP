import sys, os
import logging
import time
import warnings

import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import DataLoader
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from .utils import load_processed_datasets

class JetGNNBase(LightningModule):
    
    def __init__(self, hparams):
        super().__init__()
        
        """
        Initialise the Lightning Module that can scan over different Equivariant GNN training regimes
        """
        # Assign hyperparameters
        self.save_hyperparameters(hparams)

        self.trainset, self.valset, self.testset = None, None, None

    def setup(self, stage):
        if (self.trainset is None) and (self.valset is None) and (self.testset is None):
            self.trainset, self.valset, self.testset = load_processed_datasets(self.hparams["input_dir"], 
                                                        self.hparams["data_split"],
                                                        self.hparams["graph_construction"]
                                                        )
        
        try:
            print("Defining figures of merit")
            self.logger.experiment.define_metric("val_loss" , summary="min")
            self.logger.experiment.define_metric("auc" , summary="max")
            self.logger.experiment.define_metric("acc" , summary="max")
            self.logger.experiment.define_metric("inv_eps" , summary="max")
        except Exception:
            warnings.warn("Failed to define figures of merit, due to logger unavailable")
            
        print(time.ctime())
        
    def concat_feature_set(self, batch):
        """
        Useful in all models to use all available features of size == len(x)
        """
        
        all_features = []
        for feature in self.hparams["feature_set"]:
            if len(batch[feature]) == len(batch.x):
                all_features.append(batch[feature])
            else:
                all_features.append(batch[feature][batch.batch])
        return torch.stack(all_features).T

    def get_metrics(self, batch, output):
        
        prediction = torch.sigmoid(output)
        tp = (prediction.round() == batch.y).sum().item()
        acc = tp / len(batch.y)
        
        try:
            auc = roc_auc_score(batch.y.bool().cpu().detach(), prediction.cpu().detach())
        except Exception:
            auc = 0
        fpr, tpr, _ = roc_curve(batch.y.bool().cpu().detach(), prediction.cpu().detach())
        
        # Calculate which threshold gives the best signal goal
        signal_goal_idx = abs(tpr - self.hparams["signal_goal"]).argmin()
        
        eps_fpr = fpr[signal_goal_idx]
        eps_tpr = tpr[signal_goal_idx]

        
        return prediction, acc, auc, eps_fpr, eps_tpr
    
    def training_step(self, batch, batch_idx):
                
        output = self(batch).squeeze(-1)
        
        loss = F.binary_cross_entropy_with_logits(output, batch.y.float(), pos_weight=torch.tensor(self.hparams["pos_weight"]))
        
        prediction, acc, auc, eps, eps_eff = self.get_metrics(batch, output)
        
        self.log_dict({"train_loss": loss, "train_acc": acc}, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        
        output = self(batch).squeeze(-1)

        loss = F.binary_cross_entropy_with_logits(output, batch.y.float(), pos_weight=torch.tensor(self.hparams["pos_weight"]))

        prediction, acc, auc, eps, eps_eff = self.get_metrics(batch, output)
        
        current_lr = self.optimizers().param_groups[0]["lr"]
        
        self.log_dict({"val_loss": loss, "acc": acc, "auc": auc, "current_lr": current_lr}, on_step=False, on_epoch=True)
        
        return {
            "loss": loss,
            "preds": prediction,
            "acc": acc,
            "auc": auc,
            "eps": eps,
            "eps_eff": eps_eff,
        }
    
    def validation_epoch_end(self, step_outputs):
        mean_eps = np.mean([output["eps"] for output in step_outputs])
        mean_eps_eff = np.mean([output["eps_eff"] for output in step_outputs])
        
        if mean_eps != 0:
            self.log_dict({"inv_eps": 1/mean_eps, "ant": (1/mean_eps) / self.get_num_params(), "eps_eff": mean_eps_eff})
    
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure=None,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # warm up lr
        if (self.hparams["warmup"] is not None) and (
            self.current_epoch < self.hparams["warmup"]
        ):
            lr_scale = min(
                1.0, float(self.current_epoch + 1) / self.hparams["warmup"]
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams["lr"]

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def train_dataloader(self):
        if self.trainset is not None:
            return DataLoader(self.trainset, batch_size=self.hparams["train_batch"], num_workers=1, shuffle=True)
        else:
            return None

    def val_dataloader(self):
        if self.valset is not None:
            return DataLoader(self.valset, batch_size=self.hparams["val_batch"], num_workers=1)
        else:
            return None

    def test_dataloader(self):
        if self.testset is not None:
            return DataLoader(self.testset, batch_size=1, num_workers=1)
        else:
            return None

    def configure_optimizers(self):
        optimizer = [
            torch.optim.AdamW(
                self.parameters(),
                lr=(self.hparams["lr"]),
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0.0001,
                amsgrad=True,
            )
        ]
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer[0],
                    step_size=self.hparams["patience"],
                    gamma=self.hparams["factor"],
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return optimizer, scheduler
        
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
        