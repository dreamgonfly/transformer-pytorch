from typing import Optional

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
import torch

from .checkpointers.checkpointer import (
    Checkpointer,
    ModelInfo,
)
from .feeder import Feeder
from .loggers.logger import Logger
from .progress_bar import ProgressBar
from .runner import Runner
from .training_info import TrainingInfo
from .utils import (
    to_float,
    to_numpy,
    dictionarize_list,
    to_device,
)


class Trainer:
    logger: Logger
    model_checkpoint: Checkpointer
    progress_bar: ProgressBar

    gradient_accumulation_steps: int
    gradient_clip_val: Optional[float]
    epochs: Optional[int]
    # steps: Optional[int]
    device: torch.device

    use_amp: bool
    scaler: torch.cuda.amp.GradScaler
    previous_scale = float

    epoch: int
    global_step: int
    total_num_batches: int

    model: nn.Module
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    optimizer: Optimizer
    lr_scheduler: _LRScheduler
    model_runner: Runner

    def __init__(
        self,
        logger: Logger,
        model_checkpoint: Checkpointer,
        progress_bar: ProgressBar,
        gradient_accumulation_steps: int,
        gradient_clip_val: Optional[float],
        epochs: int,
        # steps: int,  # TODO
        num_sanity_check_steps: int,
        device: torch.device,
        use_amp: bool,
    ):
        self.logger = logger
        self.model_checkpoint = model_checkpoint
        self.progress_bar = progress_bar

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip_val = gradient_clip_val
        self.epochs = epochs
        # self.steps = steps
        self.num_sanity_check_steps = num_sanity_check_steps
        self.device = device

        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.previous_scale = self.scaler.get_scale()
        # TODO: Saving & Loading scaler

        self.epoch = 0
        self.global_step = 0

    def train(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler,
        data_feeder: Feeder,
        model_runner: Runner,
    ):
        self.initialize_trainer(model, optimizer, lr_scheduler, data_feeder, model_runner)
        self.run()

    def initialize_trainer(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler,
        data_feeder: Feeder,
        model_runner: Runner,
    ):
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.train_dataloader = data_feeder.train_dataloader
        self.val_dataloader = data_feeder.val_dataloader

        self.model_runner = model_runner

    def run(self):
        self.sanity_check()
        self.run_epochs(self.epochs)

    def sanity_check(self):
        self.progress_bar.on_sanity_check_start(num_sanity_check_steps=self.num_sanity_check_steps)

        self.model.eval()
        val_results = []
        for batch_index, batch in enumerate(self.val_dataloader):
            if batch_index >= self.num_sanity_check_steps:
                break
            with torch.no_grad():
                val_result = self.run_val_step(batch_index, batch)
            val_results.append(to_numpy(val_result))
        val_results_dict = dictionarize_list(val_results)
        val_result_aggregated = self.model_runner.validation_epoch_end(val_results_dict)
        val_result_float = to_float(val_result_aggregated)
        self.progress_bar.on_validation_end(val_result_float)
        self.progress_bar.on_sanity_check_end()

    def run_epochs(self, epochs: int):
        self.progress_bar.on_train_start()

        for epoch in range(epochs):
            self.epoch = epoch
            self.run_epoch()

        self.progress_bar.on_train_end()

    def run_epoch(self):
        self.progress_bar.on_epoch_start(epoch=self.epoch, total_num_batches=self.total_num_batches)

        self.run_train_epoch()
        val_result_aggregated = self.run_val_epoch()

        self.checkpoint(val_result_aggregated)
        self.progress_bar.on_epoch_end()

    def run_train_epoch(self):
        self.model.train()
        for batch_index, batch in enumerate(self.train_dataloader):
            self.run_train_step(batch_index, batch)

    def run_train_step(self, batch_index, batch):
        self.global_step += 1
        batch = to_device(batch, self.device)
        info = TrainingInfo(self.epoch, self.global_step, batch_index)
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            train_result = self.model_runner.training_step(self.model, batch, info)
            loss = train_result["loss"]

        # Accumulate loss
        loss = loss / self.gradient_accumulation_steps

        self.scaler.scale(loss).backward()

        accumulation_done = (batch_index + 1) % self.gradient_accumulation_steps == 0
        is_final_batch = (batch_index + 1) == len(self.train_dataloader)
        if accumulation_done or is_final_batch:
            if self.gradient_clip_val is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()  # TODO: set_to_none=True
            if self.scaler.get_scale() == self.previous_scale:
                self.lr_scheduler.step()
            else:
                self.previous_scale = self.scaler.get_scale()

            train_result = to_float(train_result)
            train_result["epoch"] = self.epoch
            train_result["global_step"] = self.global_step
            train_result["lr"] = self.optimizer.param_groups[0]["lr"]
            self.progress_bar.on_batch_end(train_result)
            self.logger.log_metrics_at_intervals(train_result)

    def run_val_epoch(self):
        self.progress_bar.on_validation_start(val_num_batches=len(self.val_dataloader))
        val_results = []
        self.model.eval()

        for batch_index, batch in enumerate(self.val_dataloader):
            with torch.no_grad():
                val_result = self.run_val_step(batch_index, batch)
            val_results.append(to_numpy(val_result))
        val_results_dict = dictionarize_list(val_results)
        val_metrics = self.model_runner.validation_epoch_end(val_results_dict)
        val_metrics = to_float(val_metrics)

        val_metrics["epoch"] = self.epoch
        val_metrics["global_step"] = self.global_step
        self.progress_bar.on_validation_end(val_metrics)
        self.logger.log_metrics(val_metrics)
        return val_metrics

    def run_val_step(self, batch_index, batch):
        batch = to_device(batch, self.device)
        info = TrainingInfo(self.epoch, self.global_step, batch_index)
        val_result = self.model_runner.validation_step(self.model, batch, info)
        self.progress_bar.on_validation_batch_end()
        return val_result

    def checkpoint(self, val_metrics):
        self.model_checkpoint.checkpoint(
            model_info=ModelInfo(
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                epoch=self.epoch,
                global_step=self.global_step,
            ),
            metrics=val_metrics,
        )

    @property
    def total_num_batches(self):
        return len(self.train_dataloader) + len(self.val_dataloader)
