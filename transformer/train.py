import torch
from pathlib import Path

from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from transformer.data_feeder import DataFeeder
from transformer.lr_schedulers.noam_lr_scheduler import NoamLRScheduler

from transformer.model.transformer import Transformer

from transformer.model_runner import ModelRunner
from transformer.token_indexers.vocabulary_indexer import PAD_TOKEN_NAME
from transformer.training.checkpointers.checkpointer import MonitorMode
from transformer.training.checkpointers.model_checkpointer import ModelCheckpointer
from transformer.training.loggers.file_logger import FileLogger
from transformer.training.progress_bar import ProgressBar
from transformer.training.trainer import Trainer
from transformer.training.utils import find_next_version_dir, seed


def train(
    train_path: Path,
    val_path: Path,
    source_vocab_path: Path,
    target_vocab_path: Path,
    run_name: str,
    runs_dir: Path = "results/runs",
    num_layers: int = 6,
    epochs: int = 400,
    batch_size: int = 256,
    warmup_steps: int = 128000,
):
    seed(0)

    data_feeder = DataFeeder(
        train_path, val_path, source_vocab_path, target_vocab_path, batch_size, max_length=100
    )
    pad_token_index = data_feeder.source_token_indexer.encode_token_name(PAD_TOKEN_NAME)

    model_runner = ModelRunner(pad_token_index=pad_token_index)

    model = Transformer(
        data_feeder.source_token_indexer.num_tokens(),
        data_feeder.target_token_indexer.num_tokens(),
        pad_token_index=pad_token_index,
        d_model=512,
        d_ff=2048,
        num_layers=num_layers,
        n_heads=8,
        dropout=0.1,
        num_positions=200,
        input_target_weight_sharing=True,
        source_target_weight_sharing=True,
    )

    optimizer = Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-09)
    lr_factor_scheduler = NoamLRScheduler(factor=2.0, d_model=512, wamrup_steps=warmup_steps)
    lr_scheduler = LambdaLR(optimizer, lr_factor_scheduler.get_factor)

    version_dir = find_next_version_dir(runs_dir=runs_dir, run_name=run_name)

    model_logger = FileLogger(log_dir=version_dir)

    name_prefix = f"{run_name.replace('/', '-')}-{version_dir.name}-"
    model_checkpoint = ModelCheckpointer(
        checkpoints_dir=version_dir.joinpath("checkpoints"),
        monitor_metric="val_ppl",
        mode=MonitorMode.MIN,
        top_k=1,
        save_last=True,
        period=1,
        name_format=name_prefix + "epoch-{epoch:0>3}-val-ppl-{val_ppl:.5f}.checkpoint",
        save_weights_only=True,
        config={
            "source_vocab_path": str(source_vocab_path),
            "target_vocab_path": str(target_vocab_path),
            "num_layers": num_layers,
        },
    )

    progress_bar = ProgressBar(
        train_monitor_metrics=["loss", "acc", "lr"],
        val_monitor_metrics=["val_loss", "val_acc", "val_ppl"],
        version=version_dir.name,
        refresh_rate=1,
    )

    trainer = Trainer(
        logger=model_logger,
        model_checkpoint=model_checkpoint,
        progress_bar=progress_bar,
        gradient_accumulation_steps=1,
        gradient_clip_val=None,
        epochs=epochs,
        num_sanity_check_steps=3,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        use_amp=False,
    )
    trainer.train(model, optimizer, lr_scheduler, data_feeder, model_runner)
