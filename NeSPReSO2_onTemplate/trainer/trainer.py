import json
import os
import tempfile
from datetime import datetime, timezone

import numpy as np
import torch
from base.base_trainer import BaseTrainer
from base.util import MetricTracker
from base.performance import autocast_dtype_from_name


class Trainer(BaseTrainer):
    """Trainer for NeSPReSO (inputs, targets, index) batches."""

    def __init__(
        self,
        model,
        criterion,
        metric_ftns,
        optimizer,
        config,
        device,
        data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
        len_epoch=None,
        checkpoint_extra=None,
        performance=None,
    ):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.device = device
        self.performance = performance or {}
        self._use_autocast = bool(self.performance.get("autocast"))
        self._autocast_dtype = autocast_dtype_from_name(self.performance.get("autocast_dtype", "bfloat16"))
        self._grad_scaler = None
        if self._use_autocast and self.performance.get("autocast_dtype") == "float16" and device.type == "cuda":
            self._grad_scaler = torch.amp.GradScaler("cuda")
        self.data_loader = data_loader
        self.checkpoint_extra = checkpoint_extra or {}
        if len_epoch is None:
            self.len_epoch = len(self.data_loader)
        else:
            from base.util import inf_loop

            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker("loss", *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker("loss", *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self._last_log = {}
        self._last_epoch = 0

    def _dataset_tag(self):
        return self.config.config.get("io", {}).get("dataset_tag", "unknown")

    def _status_payload(self, state, reason=None):
        log = self._last_log
        payload = {
            "tag": self._dataset_tag(),
            "state": state,
            "reason": reason,
            "epoch": self._last_epoch,
            "max_epochs": self.epochs,
            "train_loss": log.get("loss"),
            "val_loss": log.get("val_loss"),
            "mnt_best": None if self.mnt_mode == "off" else self.mnt_best,
            "mnt_metric": getattr(self, "mnt_metric", None),
            "not_improved_count": self._not_improved_count,
            "early_stop": self.early_stop if self.early_stop != float("inf") else None,
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "save_dir": str(self.checkpoint_dir),
        }
        if self.mnt_mode != "off" and payload["mnt_best"] is not None:
            payload["mnt_best"] = float(payload["mnt_best"])
        return payload

    def _atomic_write_status(self, payload):
        fd, tmp = tempfile.mkstemp(dir=self.checkpoint_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, self.checkpoint_dir / "status.json")
        except Exception:
            os.unlink(tmp)
            raise

    def _emit_sentinel(self, kind, extra=None):
        body = {"tag": self._dataset_tag()}
        if extra:
            body.update(extra)
        print(f"NESPRO_TRAIN_{kind} {json.dumps(body, separators=(',', ':'))}", flush=True)

    def _epoch_telemetry(self, epoch, log, not_improved_count):
        self._last_epoch = epoch
        self._last_log = log
        payload = self._status_payload("running")
        self._atomic_write_status(payload)
        if self._should_log_epoch(epoch):
            slim = {"tag": payload["tag"], "epoch": epoch, "val_loss": log.get("val_loss")}
            self._emit_sentinel("EPOCH", slim)

    def _train_finished(self, reason):
        state = "done"
        payload = self._status_payload(state, reason=reason)
        self._atomic_write_status(payload)
        self._emit_sentinel("DONE", {"tag": payload["tag"], "epoch": payload["epoch"], "reason": reason})

    def train(self):
        try:
            super().train()
        except Exception as exc:
            payload = self._status_payload("failed", reason=type(exc).__name__)
            self._atomic_write_status(payload)
            self._emit_sentinel("FAIL", {"tag": payload["tag"], "reason": type(exc).__name__})
            raise

    def _loss(self, output, target, indices, inputs=None):
        if indices is None:
            return self.criterion(output, target)
        if getattr(self.criterion, "needs_inputs", False):
            return self.criterion(output, target, indices, inputs=inputs)
        return self.criterion(output, target, indices)

    def _forward_loss(self, data, target, indices):
        with torch.autocast(
            device_type=self.device.type,
            dtype=self._autocast_dtype,
            enabled=self._use_autocast,
        ):
            output = self.model(data)
            loss = self._loss(output, target, indices, inputs=data)
        return output, loss

    def _optimizer_step(self, loss):
        if self._grad_scaler is not None:
            self._grad_scaler.scale(loss).backward()
            self._grad_scaler.step(self.optimizer)
            self._grad_scaler.update()
            return
        loss.backward()
        self.optimizer.step()

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target, indices) in enumerate(self.data_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            indices = indices.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            output, loss = self._forward_loss(data, target, indices)
            self._optimizer_step(loss)

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update("loss", loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target, indices, self.data_loader))

            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(epoch, self._progress(batch_idx), loss.item()),
                    extra={"console": self._should_log_epoch(epoch)},
                )

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if hasattr(self.model, "gate"):
            grad_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    grad_norm += float(p.grad.data.norm(2) ** 2)
            log["grad_norm"] = grad_norm ** 0.5
            if self.writer is not None:
                self.writer.add_scalar("train/grad_norm", log["grad_norm"], epoch)

        return log

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target, indices) in enumerate(self.valid_data_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                indices = indices.to(self.device)

                output, loss = self._forward_loss(data, target, indices)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, "valid")
                self.valid_metrics.update("loss", loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target, indices, self.valid_data_loader))

        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _save_checkpoint(self, epoch, save_best_val=False, save_best_train=False):
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.mnt_best,
            "best_train_loss": self.best_train_loss,
            "config": self.config,
            **self.checkpoint_extra,
        }
        show = self._should_log_epoch(epoch)
        latest_path = str(self.checkpoint_dir / "checkpoint.pth")
        torch.save(state, latest_path)
        self.logger.debug("Saving checkpoint: {} ...".format(latest_path), extra={"console": show})
        if save_best_val:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            self.logger.debug("Saving current best val: model_best.pth ...", extra={"console": show})
        if save_best_train:
            best_train_path = str(self.checkpoint_dir / "model_best_train.pth")
            torch.save(state, best_train_path)
            self.logger.debug("Saving current best train: model_best_train.pth ...", extra={"console": show})
