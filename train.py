import hydra

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichProgressBar,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate


@hydra.main(version_base=None, config_path="configs", config_name="main")
def train(cfg: DictConfig):
    # conf = OmegaConf.to_yaml(cfg)
    print(cfg)

    model = instantiate(cfg.model)
    logger = CSVLogger(save_dir=cfg.trainer.output_folder, name="experiments")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    dataloader = instantiate(cfg.dataloader)
    ck_callback = ModelCheckpoint(
        filename="model-{epoch}-{val_loss:.5f}",
        mode="min",
        save_last=True,
    )
    pb = RichProgressBar()

    trainer = pl.Trainer(
        accelerator="cuda",
        max_epochs=cfg.trainer.max_epochs,
        logger=logger,
        callbacks=[ck_callback, pb, lr_monitor],
        log_every_n_steps=512,
        gradient_clip_val=5,
        use_distributed_sampler=False,
        accumulate_grad_batches=2,
    )

    restore_path = cfg.trainer.restore_path if cfg.trainer.restore_path else None
    trainer.fit(
        model=model,
        train_dataloaders=dataloader,
        ckpt_path=restore_path,
    )
    trainer.test(
        model=model,
        dataloaders=dataloader,
        ckpt_path="best",
    )


if __name__ == "__main__":
    train()
