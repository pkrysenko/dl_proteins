import torch
import pl_bolts
from torch.nn import functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import lightning.pytorch as pl
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryRecall,
    BinaryAUROC,
    BinaryPrecision,
)
from torchmetrics.regression import PearsonCorrCoef, R2Score, MeanSquaredError


def masked_mean(
    input,
    mask,
    dim,
    keepdim=False,
):
    return (input * mask).sum(dim=dim, keepdim=keepdim) / mask.broadcast_to(
        input.shape
    ).sum(dim=dim, keepdim=keepdim)


def get_mask(inp_array, padding_value):
    return ~(inp_array == padding_value)


def metrics_calculation_func(task, device):
    output_metrics_func = {}
    if task == "binary":
        output_metrics_func["accuracy"] = BinaryAccuracy().to(device)
        output_metrics_func["recall"] = BinaryRecall().to(device)
        output_metrics_func["auroc"] = BinaryAUROC().to(device)
        output_metrics_func["precision"] = BinaryPrecision().to(device)
    elif task == "regression":
        output_metrics_func["r2"] = R2Score().to(device)
        output_metrics_func["pearson"] = PearsonCorrCoef().to(device)
        output_metrics_func["rmse"] = MeanSquaredError(squared=False).to(device)
        output_metrics_func["mse"] = MeanSquaredError(squared=True).to(device)

    return output_metrics_func


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.0, max_len=2048):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class LigadTransformer(pl.LightningModule):
    def __init__(
        self,
        d_model,
        d_ff,
        f_vocab_size,
        s_vocab_size,
        nheads,
        num_encoder_layers,
        num_decoder_layers,
        d_fc_layers,
        max_epochs,
        task="binary",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.embedding_f = torch.nn.Embedding(f_vocab_size, d_model)
        self.embedding_s = torch.nn.Embedding(s_vocab_size, d_model)

        self.pos_enc = PositionalEncoding(d_model, dropout=0.1)
        # self.pos_enc_s = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nheads,
            dim_feedforward=d_ff,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            batch_first=True,
            dropout=0.25,
        )

        self.pooling = masked_mean
        d_fc_layers.insert(0, d_model)
        self.fc_block = nn.Sequential()
        for in_feat, out_feat in zip(d_fc_layers, d_fc_layers[1:]):
            self.fc_block.append(nn.Dropout(0.25))
            self.fc_block.append(nn.Linear(in_feat, out_feat))
            self.fc_block.append(nn.LeakyReLU())

        self.out_act = F.sigmoid if task == "binary" else F.relu
        self.out_layer = nn.Linear(d_fc_layers[-1], 1)

        self.loss = nn.BCELoss() if task == "binary" else nn.HuberLoss()

        self.train_metrics = ["mse"]
        self.valid_metrics = ["mse"]
        self.test_metrics = ["rmse", "r2", "pearson"]
        if task == "regression":
            self.train_metrics_mc = MetricCollection(
                prefix="train_", metrics=[MeanSquaredError(squared=True)]
            )
            self.val_metrics_mc = MetricCollection(
                prefix="val_", metrics=[MeanSquaredError(squared=True)]
            )
            self.test_metrics_mc = MetricCollection(
                prefix="test_",
                metrics=[MeanSquaredError(squared=False), PearsonCorrCoef(), R2Score()],
            )
        else:
            self.train_metrics_mc = MetricCollection(
                prefix="train_", metrics=[BinaryAccuracy()]
            )
            self.val_metrics_mc = MetricCollection(
                prefix="val_", metrics=[BinaryAccuracy()]
            )
            self.test_metrics_mc = MetricCollection(
                prefix="test_",
                metrics=[
                    BinaryAccuracy(),
                    BinaryAUROC(),
                    BinaryRecall(),
                    BinaryPrecision(),
                ],
            )
        self.task = task
        self.max_epochs = max_epochs

    def forward(self, f_seq, s_seq):
        embed_f_seq = self.embedding_f(f_seq)
        embed_s_seq = self.embedding_s(s_seq)

        embed_f_seq = self.pos_enc(embed_f_seq)
        embed_s_seq = self.pos_enc(embed_s_seq)

        transformer_out = self.transformer(
            embed_f_seq, embed_s_seq, src_is_causal=False, tgt_is_causal=False
        )
        mask_out = get_mask(transformer_out, padding_value=0)

        pooled_out = self.pooling(transformer_out, mask_out, dim=1)

        fc_out = self.fc_block(pooled_out)
        score = self.out_act(self.out_layer(fc_out))

        return score

    def training_step(self, batch, batch_idx):
        f_seq, s_seq, target = batch
        pred = self(f_seq, s_seq)
        loss = self.loss(pred, target)

        metrics_log = self.train_metrics_mc(pred, target)
        metrics_log["train_loss"] = loss

        self.log_dict(metrics_log, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        f_seq, s_seq, target = batch
        pred = self(f_seq, s_seq)
        loss = self.loss(pred, target)

        metrics_log = self.val_metrics_mc(pred, target)

        metrics_log["val_loss"] = loss

        self.log_dict(metrics_log, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        f_seq, s_seq, target = batch
        pred = self(f_seq, s_seq)
        loss = self.loss(pred, target)

        metrics_log = self.test_metrics_mc(pred, target)

        metrics_log["test_loss"] = loss

        self.log_dict(metrics_log, on_epoch=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5, weight_decay=0.001)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
            eta_min=1e-7,
        )
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "name": None,
        }
        return [optimizer], [lr_scheduler_config]


if __name__ == "__main__":
    m = LigadTransformer(16, 16, 128, 128, 2, 5, 5, [64, 32])
    inp_data = torch.randint(1, 128, (2, 32))

    score = m(inp_data, inp_data)
    print(score)
    print(score.size())
