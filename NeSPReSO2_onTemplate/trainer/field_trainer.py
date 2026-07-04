"""Trainer for field-to-field U-Net with sparse pixel supervision."""

from __future__ import annotations

import torch

from trainer.trainer import Trainer


class FieldTrainer(Trainer):
    def _loss(self, output, target, indices, inputs=None, pixel_rc=None, batch_of_pixel=None):
        if pixel_rc is None or batch_of_pixel is None:
            return super()._loss(output, target, indices, inputs=inputs)
        bop = batch_of_pixel.to(output.device)
        rc = pixel_rc.to(output.device)
        rows = rc[:, 0].long()
        cols = rc[:, 1].long()
        pred_pcs = output[bop, :, rows, cols]
        if getattr(self.criterion, "needs_inputs", False):
            return self.criterion(pred_pcs, target, indices, inputs=inputs)
        return self.criterion(pred_pcs, target, indices)

    def _forward_loss(self, data, target, indices, pixel_rc=None, batch_of_pixel=None):
        with torch.autocast(
            device_type=self.device.type,
            dtype=self._autocast_dtype,
            enabled=self._use_autocast,
        ):
            output = self.model(data)
            loss = self._loss(
                output, target, indices, inputs=data, pixel_rc=pixel_rc, batch_of_pixel=batch_of_pixel
            )
        return output, loss

    def _gather_output(self, output, pixel_rc, batch_of_pixel):
        bop = batch_of_pixel.to(output.device)
        rc = pixel_rc.to(output.device)
        return output[bop, :, rc[:, 0].long(), rc[:, 1].long()]

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, batch in enumerate(self.data_loader):
            if len(batch) == 5:
                data, target, indices, pixel_rc, batch_of_pixel = batch
            else:
                data, target, indices = batch
                pixel_rc, batch_of_pixel = None, None
            data = data.to(self.device)
            target = target.to(self.device)
            indices = indices.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            output, loss = self._forward_loss(data, target, indices, pixel_rc, batch_of_pixel)
            self._optimizer_step(loss)

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update("loss", loss.item())
            if pixel_rc is not None:
                gathered = self._gather_output(output, pixel_rc, batch_of_pixel)
                for met in self.metric_ftns:
                    self.train_metrics.update(met.__name__, met(gathered, target, indices, self.data_loader))

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
        return log

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_data_loader):
                if len(batch) == 5:
                    data, target, indices, pixel_rc, batch_of_pixel = batch
                else:
                    data, target, indices = batch
                    pixel_rc, batch_of_pixel = None, None
                data = data.to(self.device)
                target = target.to(self.device)
                indices = indices.to(self.device)
                output, loss = self._forward_loss(data, target, indices, pixel_rc, batch_of_pixel)
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, "valid")
                self.valid_metrics.update("loss", loss.item())
                if pixel_rc is not None:
                    gathered = self._gather_output(output, pixel_rc, batch_of_pixel)
                    for met in self.metric_ftns:
                        self.valid_metrics.update(met.__name__, met(gathered, target, indices, self.valid_data_loader))
        return self.valid_metrics.result()
