# base_trainer.py
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from src.Apps.FusionGait.GaitRecognition.IdentityModel.DopplerBaseLearning.train_config import TrainConfig
from src.Apps.FusionGait.GaitRecognition.IdentityModel.Utils.metrics import AverageMeter, accuracy

class BaseTrainer(ABC):
    """
    BaseTrainer
    ------------
    - Sole responsibility: training / validation / saving
    - Does NOT care about:
        * single-modality vs. multi-modality
        * input dimensionality
        * model architecture
    - Decouples input differences via prepare_batch / forward
    """

    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            train_cfg: TrainConfig,
            logger=None,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_cfg = train_cfg
        self.logger = logger

        # 1. Automatically build the optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
            betas=train_cfg.betas
        )

        # 2. Automatically build the loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        #self.criterion = nn.CrossEntropyLoss()

        # 3. Learning rate scheduler: supports Warmup + Cosine
        if train_cfg.lr_scheduler == "warmup_cosine":
            # Compute warmup steps (in epochs)
            warmup_epochs = getattr(train_cfg, "warmup_epochs", 5)

            # 1. Warmup: linearly increase from lr*0.1 to lr
            warmup_lr_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=warmup_epochs
            )

            # 2. Annealing: cosine decay from lr to a very small value
            cosine_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=train_cfg.epochs - warmup_epochs
            )

            # 3. Chain the two schedulers
            self.scheduler = optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_lr_scheduler, cosine_lr_scheduler],
                milestones=[warmup_epochs]
            )

        elif train_cfg.lr_scheduler == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=train_cfg.step_size, gamma=train_cfg.gamma
            )
        elif train_cfg.lr_scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=train_cfg.epochs
            )
        else:
            self.scheduler = None

    # ==========================================================
    # Input Adapter Hooks (subclasses only need to care about these two)
    # ==========================================================
    @abstractmethod
    def prepare_batch(self, batch) -> Tuple[Any, torch.Tensor]:
        """
        Returns:
            inputs:  any structure as long as forward can consume it
            targets: torch.Tensor
        """
        pass

    def forward(self, inputs):
        """
        Default forward: model(inputs)
        FusionTrainer can override
        """
        return self.model(inputs)

    # ==========================================================
    #  Train / Eval (fully generic)
    # ==========================================================
    def train_one_epoch(self, dataloader):
        self.model.train()
        loss_meter, acc_meter = AverageMeter(), AverageMeter()

        for batch_idx, batch in enumerate(dataloader):
            inputs, targets = self.prepare_batch(batch)

            self.optimizer.zero_grad()
            logits = self.forward(inputs)
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()

            pred = torch.argmax(logits, dim=1)
            acc = accuracy(pred, targets)

            loss_meter.update(loss.item(), targets.size(0))
            acc_meter.update(acc, targets.size(0))

            if self.logger and batch_idx % self.train_cfg.log_interval == 0:
                self.logger.info(
                    f"[Train] Iter {batch_idx}/{len(dataloader)} "
                    f"Loss={loss_meter.avg:.4f} Acc={acc_meter.avg:.4f}"
                )

        return loss_meter.avg, acc_meter.avg

    @torch.no_grad()
    def eval_one_epoch(self, dataloader):
        self.model.eval()
        loss_meter, acc_meter = AverageMeter(), AverageMeter()

        for batch in dataloader:
            inputs, targets = self.prepare_batch(batch)
            logits = self.forward(inputs)
            loss = self.criterion(logits, targets)

            pred = torch.argmax(logits, dim=1)
            acc = accuracy(pred, targets)

            loss_meter.update(loss.item(), targets.size(0))
            acc_meter.update(acc, targets.size(0))

        if self.logger:
            self.logger.info(
                f"[Eval] Loss={loss_meter.avg:.4f} Acc={acc_meter.avg:.4f}"
            )

        return loss_meter.avg, acc_meter.avg

    # ==========================================================
    #  Fit (full lifecycle)
    # ==========================================================
    def fit(self, train_loader, val_loader, save_path: Optional[str] = None):
        best_acc = 0.0
        patience_counter = 0  # for Early Stopping
        epochs = self.train_cfg.epochs

        for epoch in range(epochs):
            if self.logger:
                self.logger.info(f"======== Epoch {epoch+1}/{epochs} ========")

            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.eval_one_epoch(val_loader)

            # Update learning rate
            if self.scheduler is not None:
                # Record current LR before stepping (optional, for logging)
                current_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step()
                if self.logger:
                    self.logger.info(f"📍 Learning Rate: {current_lr:.6f} -> {self.optimizer.param_groups[0]['lr']:.6f}")

            if self.logger:
                self.logger.info(
                    f"[Epoch {epoch+1}] "
                    f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
                    f"Val Loss={val_loss:.4f} Acc={val_acc:.4f}"
                )

            if save_path and val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
                if self.logger:
                    self.logger.info(f"🔥 Saved best model @ Acc={best_acc:.4f}")

                patience_counter = 0  # reset counter
            else:
                patience_counter += 1

            if self.train_cfg.enable_early_stopping and patience_counter >= self.train_cfg.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if self.logger:
            self.logger.info("Training finished.")

    @torch.no_grad()
    def predict(self, dataloader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference and collect all labels and predictions for later confusion-matrix analysis
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        for batch in dataloader:
            inputs, targets = self.prepare_batch(batch)
            logits = self.forward(inputs)

            preds = torch.argmax(logits, dim=1)

            # Move to CPU and convert to numpy
            all_preds.append(preds.cpu().numpy())
            all_labels.append(targets.cpu().numpy())

        # Concatenate results across all batches
        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)

        return y_true, y_pred


class SingleInputTrainer(BaseTrainer):
    """
    Adapter for single-modality Dataset
    batch = (x, y, mask)
    """

    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            train_cfg: TrainConfig,
            logger=None,
    ):
        super().__init__(model, device, train_cfg, logger)

    def prepare_batch(self, batch):
        # Robust unpacking logic
        if len(batch) == 3:
            x, y, mask = batch
        else:
            x, y = batch
            mask = None

        # Only type conversion and device transfer; no permute involved
        x = x.float().to(self.device)
        y = y.long().to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        return x, y


class FusionInputTrainer(BaseTrainer):
    """
    Adapter for fusion-modality Dataset
    batch = (all_padded, labels, all_masks)
    """

    def prepare_batch(self, batch):
        """
        Unpack according to the return format of our collate_fn
        """
        # 1. Flexible unpacking
        if len(batch) == 3:
            all_x, y, all_masks = batch
        else:
            # Compatible with old versions or special cases without masks
            all_x, y = batch
            all_masks = None

        # 2. Move all modalities' data to device (all_x is a list)
        inputs = [x.float().to(self.device) for x in all_x]
        targets = y.long().to(self.device)

        # If the model needs masks, they can be prepared as well
        # self.current_masks = [m.to(self.device) for m in all_masks] if all_masks else None

        # Return: (inputs_list, targets)
        return inputs, targets

    def forward(self, inputs):
        """
        inputs is now a list of tensors: [tensor_wifi, tensor_acoustic]
        Directly pass it to the model's forward
        """
        # There are two options here, depending on your model forward signature:
        # Option A: model forward(self, x_list) takes a list (recommended)
        return self.model(inputs)

        # Option B: model forward(self, x1, x2) takes multiple args
        # return self.model(*inputs)
