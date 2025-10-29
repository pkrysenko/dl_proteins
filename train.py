import hydra

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from data.ligads import BinaryLigadsDL
from arch.ligads import LigadTransformer

F_VOCAB_SIZE = 128
S_VOCAB_SIZE = 128

d_model = 32

if __name__ == "__main__":
    pl.seed_everything(42)
    ck_callback = ModelCheckpoint(
        filename="model-{epoch}-{val_loss:.5f}", mode="max", save_last=True
    )

    logger = CSVLogger("logs", name="experiments")
    dl_ligad = BinaryLigadsDL(
        csv_path="csv_dataset/filtered_binding_finalm.csv",
        f_seq_column="Ligand",
        s_seq_column="Target_Chain",
        out_column="logIC50_scaled",
        f_vocab_size=F_VOCAB_SIZE,
        s_vocab_size=S_VOCAB_SIZE,
        batch_size=16,
        task="regression",
    )

    model = LigadTransformer(
        d_model=d_model,
        d_ff=64,
        f_vocab_size=F_VOCAB_SIZE,
        s_vocab_size=S_VOCAB_SIZE,
        nheads=8,
        num_encoder_layers=5,
        num_decoder_layers=5,
        d_fc_layers=[32, 16],
        task="regression",
    )

    trainer = pl.Trainer(accelerator="gpu", max_epochs=5, callbacks=[ck_callback])
    trainer.fit(model=model, train_dataloaders=dl_ligad)
    trainer.test(model=model, dataloaders=dl_ligad, ckpt_path="best")
