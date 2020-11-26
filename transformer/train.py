import torch
from pathlib import Path

from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from typer import Option

from transformer.data_feeder import DataFeeder
from transformer.lr_schedulers.noam_lr_scheduler import NoamLRScheduler
from transformer.old_model.Models import Transformer
from transformer.model_runner import ModelRunner
from transformer.training.checkpointers.checkpointer import MonitorMode
from transformer.training.checkpointers.model_checkpointer import ModelCheckpointer
from transformer.training.loggers.file_logger import FileLogger
from transformer.training.progress_bar import ProgressBar
from transformer.training.trainer import Trainer
from transformer.training.utils import find_next_version_dir


def train(
    train_path: Path,
    val_path: Path,
    source_vocab_path: Path,
    target_vocab_path: Path,
    runs_dir: Path,
    run_name: str,
):

    data_feeder = DataFeeder(train_path, val_path, source_vocab_path, target_vocab_path, 256)
    model_runner = ModelRunner()

    print(
        "data_feeder.source_token_indexer.num_tokens()",
        data_feeder.source_token_indexer.num_tokens(),
    )
    print(
        "data_feeder.target_token_indexer.num_tokens()",
        data_feeder.target_token_indexer.num_tokens(),
    )
    model = Transformer(
        data_feeder.source_token_indexer.num_tokens(),
        data_feeder.target_token_indexer.num_tokens(),
        src_pad_idx=0,
        trg_pad_idx=0,
        trg_emb_prj_weight_sharing=True,
        emb_src_trg_weight_sharing=True,
        d_k=64,
        d_v=64,
        d_model=512,
        d_word_vec=512,
        d_inner=2048,
        n_layers=6,
        n_head=8,
        dropout=0.1,
    )

    optimizer = Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-09)
    lr_factor_scheduler = NoamLRScheduler(d_model=512, wamrup_steps=128000)
    lr_scheduler = LambdaLR(optimizer, lr_factor_scheduler.get_factor)

    version_dir = find_next_version_dir(runs_dir=runs_dir, run_name=run_name)

    model_logger = FileLogger(log_dir=version_dir)

    name_prefix = f"{run_name.replace('/', '-')}-{version_dir.name}-"
    model_checkpoint = ModelCheckpointer(
        checkpoints_dir=version_dir.joinpath("checkpoints"),
        monitor_metric="val_loss",
        mode=MonitorMode.MIN,
        top_k=1,
        save_last=True,
        period=1,
        name_format=name_prefix + "epoch-{epoch:0>3}-val-cer-{val_loss:.5f}.checkpoint",
        save_weights_only=True,
        config={},
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
        epochs=400,
        # steps=config.steps,
        num_sanity_check_steps=3,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        use_amp=False,
    )
    trainer.train(model, optimizer, lr_scheduler, data_feeder, model_runner)
