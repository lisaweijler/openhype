from dataclasses import asdict, dataclass
import json
import os
from typing import Union
from pathlib import Path
import torch


torch.autograd.set_detect_anomaly(True)

from torch.utils.data import DataLoader
from openhype.utils.loggingmanager import LoggingManager, LoggingConfig
from openhype.utils.wandb_logger import WandBLogger, WandBConfig
from openhype.openhype_ae.mask_feature_dataset import (
    MaskFeatCollater,
    MaskFeatureDataset,
)
from openhype.openhype_ae.hyperbolic_ae import (
    HyperEmbedder,
    HyperEmbedderConfig,
)


MAX_NUM_THREADS = 6
torch.set_num_threads(MAX_NUM_THREADS)


@dataclass
class HyperEmbTrainConfig:
    experiment_name: str = "default_exp_name"
    logging_config: LoggingConfig = LoggingConfig()
    epochs: int = 100
    num_workers: int = 2
    wandb_config: WandBConfig = WandBConfig()
    checkpoint_frequency: int = 100
    batch_size: int = 10
    weight_decay: float = 0.0001
    max_lr: float = 0.002
    pct_start: float = 0.05
    div_factor: float = 10.0
    final_div_factor: float = 1000.0
    model_config: HyperEmbedderConfig = HyperEmbedderConfig()
    cache_feature_embeds: bool = False

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}


def train_one_epoch(
    epoch_index,
    model,
    train_dataloader,
    optimizer,
    lr_scheduler,
    wandb_logger,
    device,
):
    running_loss = 0.0
    last_loss = 0.0

    running_n_masks = 0

    logging_metric_dict = {}

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_dataloader):
        # Zero your gradients for every batch
        optimizer.zero_grad()

        cumsum_batchsize = torch.cumsum(
            torch.tensor([0] + [x.item() for x in data["n_masks_used"]][:-1]), dim=0
        )

        # has parent should work here since it is only about
        # what batch it belongs to and how much to add,
        # child should belong to same batch as corresponding parent
        global_parent_id = (
            data["mask_id_parents"]
            + cumsum_batchsize[data["batch_ids"]][data["has_parent"]]
        )  # cleaned global parnet id, mask_id_parent are cleaned ids

        # Make predictions for this batch
        mask_feats = data["mask_feats"].to(device)

        loss_dict = model(
            mask_feats,
            global_parent_id.to(device),
            data["has_parent"].to(device),
            data["keep_for_negs"].to(device),
        )

        # Compute  gradients
        loss_dict["loss"].backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        n_used_masks = torch.sum(data["n_masks_used"])
        running_n_masks += n_used_masks
        running_loss += loss_dict["loss"].item() * n_used_masks

        for k, v in loss_dict["logging"].items():
            if k not in logging_metric_dict:
                logging_metric_dict[k] = v.item() * n_used_masks
            else:
                logging_metric_dict[k] += v.item() * n_used_masks

        lr_scheduler.step()

        if i % 5 == 4:  # % 10 == 9:
            last_loss = running_loss / running_n_masks  # 10# loss per batch
            print("  batch {} loss: {}".format(i + 1, last_loss))
            metrics = {
                "train/train_loss": last_loss,
                "train/epoch": epoch_index
                + 1,  # (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch,
                "lr": lr_scheduler.get_last_lr()[0],
            }

            for k, v in logging_metric_dict.items():
                print(f"  batch {i+1} {k}: {v/running_n_masks}")
                metrics[k] = v / running_n_masks

            step = epoch_index * len(train_dataloader) + i + 1

            running_loss = 0.0
            running_n_masks = 0
            logging_metric_dict = {}

            wandb_logger.log(metrics, step=step)

    return last_loss


def save_checkpoint(
    epoch, model, optimizer, lr_scheduler, checkpoint_dir, save_best=False
) -> None:

    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
    }

    if save_best:
        best_path = str(checkpoint_dir / "model_best.pth")
        torch.save(state, best_path)
    else:
        filename = str(checkpoint_dir / "checkpoint-epoch{}.pth".format(epoch))
        torch.save(state, filename)


def train(
    config: HyperEmbTrainConfig,
    img_dir: Union[Path, str],  # used in dataloader,
    output_dir: Union[Path, str],
    ckpt_dir: Union[Path, str],
    mask_feature_dir: Union[Path, str],
    mask_hierarchy_dir: Union[Path, str],
    mask_feature_suffix: str,
    mask_hierarchy_suffix: str,
    wandb_logger: WandBLogger = None,
):

    device = torch.device(f"cuda:0")
    img_dir = Path(img_dir)
    output_dir = Path(output_dir)
    ckpt_dir = Path(ckpt_dir)
    mask_feature_dir = Path(mask_feature_dir)
    mask_hierarchy_dir = Path(mask_hierarchy_dir)

    # save config
    with open(str(output_dir / "config.json"), "w") as outfile:
        json.dump(config.dict(), outfile)

    # python logger
    logging_manager = LoggingManager(config.logging_config)
    logging_manager.register_handlers(
        name="train_hyper_emb", save_path=output_dir / "log.txt"
    )
    logger = logging_manager.get_logger_by_name(name="train")

    # create dataset and loader
    logger.info("-" * 20 + "Creating data_loader instance..")
    train_data = MaskFeatureDataset(
        mask_feature_dir,
        mask_hierarchy_dir,
        img_dir,
        mask_feature_suffix,
        mask_hierarchy_suffix,
        split="train",
    )

    train_dataloader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        collate_fn=MaskFeatCollater(),
    )

    logger.info("-" * 20 + "Done!")
    # model loading
    logger.info("-" * 20 + "Loading your hot shit model..")
    model = HyperEmbedder(config.model_config)
    logger.info(model)
    logger.info("-" * 20 + "Done!")

    # push model to gpu and get trainable params
    model.to(device)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.max_lr / config.div_factor,
        weight_decay=config.weight_decay,
    )

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.max_lr,
        steps_per_epoch=len(train_dataloader),
        epochs=config.epochs + 1,
        div_factor=config.div_factor,
        final_div_factor=config.final_div_factor,
        pct_start=config.pct_start,
    )

    # WandB logger
    if wandb_logger is None:
        logger.info("-" * 20 + "Initializing W&B Logger..")
        wandb_logger = WandBLogger(
            config.wandb_config,
            model,
            run_name=config.experiment_name,
            run_config=config,
        )
        logger.info("-" * 20 + "Done!")

    best_loss = None
    for epoch in range(config.epochs):
        print("EPOCH {}:".format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(
            epoch,
            model,
            train_dataloader,
            optimizer,
            lr_scheduler,
            wandb_logger,
            device,
        )

        print("LOSS train {}".format(avg_loss))

        if epoch % config.checkpoint_frequency - 1 == 0:
            logger.info("Saving checkpoint..")
            save_checkpoint(epoch, model, optimizer, lr_scheduler, ckpt_dir)
        if best_loss is None:
            best_loss = avg_loss
        elif avg_loss < best_loss:

            logger.info("Saving current best: model_best.pth ...")
            save_checkpoint(
                epoch, model, optimizer, lr_scheduler, ckpt_dir, save_best=True
            )

    wandb_logger.finish()
