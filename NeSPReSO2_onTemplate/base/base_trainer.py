import logging

import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter


class _ConsoleEpochFilter(logging.Filter):
    """Drop throttled records on stdout; trainer DEBUG requires extra console=True."""

    def filter(self, record):
        if record.name == "trainer" and record.levelno <= logging.DEBUG:
            return getattr(record, "console", False)
        return getattr(record, "console", True)


_console_filter_attached = False


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')
        self.log_interval = int(cfg_trainer.get('log_interval', 10))

        global _console_filter_attached
        if not _console_filter_attached:
            for handler in logging.root.handlers:
                if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                    handler.addFilter(_ConsoleEpochFilter())
            _console_filter_attached = True

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
            self.early_stop = inf
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1
        self._not_improved_count = 0
        self.best_train_loss = inf

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance                
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    def _should_log_epoch(self, epoch, *, force=False):
        """Console epoch summary: first, last, every log_interval epochs, and first after resume."""
        if force:
            return True
        interval = max(1, self.log_interval)
        if epoch == 1 or epoch == self.epochs:
            return True
        if epoch == self.start_epoch and self.start_epoch != 1:
            return True
        return epoch % interval == 0

    def _format_epoch_line(self, log):
        parts = [f"epoch {log['epoch']}/{self.epochs}"]
        for key, value in log.items():
            if key == 'epoch':
                continue
            if isinstance(value, float):
                parts.append(f"{key}={value:.6g}")
            else:
                parts.append(f"{key}={value}")
        return " ".join(parts)

    def _log_epoch(self, log):
        """Log every epoch to file; print to console on log_interval (and first/last)."""
        line = self._format_epoch_line(log)
        show = self._should_log_epoch(log["epoch"])
        self.logger.info(line, extra={"console": show})
        return show

    def _epoch_telemetry(self, epoch, log, not_improved_count):
        """Override in subclass for machine-readable progress (ponytail: no-op in base)."""
        pass

    def _train_finished(self, reason):
        """Override in subclass; called once when train() loop exits normally."""
        pass

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            printed = self._log_epoch(log)

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            save_best_val = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    # ponytail: strict < / > for early-stop; <= ties reset the counter forever
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] < self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] > self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    save_best_val = True
                else:
                    not_improved_count += 1

                self._not_improved_count = not_improved_count
                self._epoch_telemetry(epoch, log, not_improved_count)

                if not_improved_count > self.early_stop:
                    if not printed:
                        print(self._format_epoch_line(log), flush=True)
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    self._train_finished("early_stop")
                    break

            else:
                self._not_improved_count = not_improved_count
                self._epoch_telemetry(epoch, log, not_improved_count)

            save_best_train = False
            train_loss = log.get('loss')
            if train_loss is not None and train_loss < self.best_train_loss:
                self.best_train_loss = train_loss
                save_best_train = True

            # Always persist best-val / best-train; periodic dump otherwise.
            if save_best_val or save_best_train or (epoch % self.save_period == 0):
                self._save_checkpoint(epoch, save_best_val=save_best_val, save_best_train=save_best_train)
        else:
            self._train_finished("completed")

    def _save_checkpoint(self, epoch, save_best_val=False, save_best_train=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param save_best_val: if True, also save as model_best.pth (lowest monitored val metric)
        :param save_best_train: if True, also save as model_best_train.pth (lowest train loss)
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'best_train_loss': self.best_train_loss,
            'config': self.config
        }
        show = self._should_log_epoch(epoch)
        latest_path = str(self.checkpoint_dir / 'checkpoint.pth')
        torch.save(state, latest_path)
        self.logger.debug("Saving checkpoint: {} ...".format(latest_path), extra={"console": show})
        if save_best_val:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.debug("Saving current best val: model_best.pth ...", extra={"console": show})
        if save_best_train:
            best_train_path = str(self.checkpoint_dir / 'model_best_train.pth')
            torch.save(state, best_train_path)
            self.logger.debug("Saving current best train: model_best_train.pth ...", extra={"console": show})

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        # Trusted local checkpoints may contain ConfigParser; weights_only=False is intentional.
        checkpoint = torch.load(resume_path, weights_only=False)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.best_train_loss = checkpoint.get('best_train_loss', inf)

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        elif checkpoint.get('optimizer') is None:
            self.logger.info("No optimizer state in checkpoint; starting optimizer fresh.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
