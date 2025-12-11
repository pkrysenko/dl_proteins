import math
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


def get_warmup_cosine_scheduler(
    optimizer, warmup_steps, total_steps, min_lr_ratio=0.001, start_factor=0.001
):
    # Phase 1: Linear Warmup (0 -> Max LR)
    scheduler1 = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_steps
    )

    # Phase 2: Cosine Decay (Max LR -> Min LR)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(total_steps - warmup_steps),
        eta_min=optimizer.param_groups[0]["lr"] * min_lr_ratio,
    )

    # Chain them
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps]
    )


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


class ConvCompressorSmall(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        return x


class ConvCompressorMedium(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size * 2,
            stride=stride,
            padding=padding,
            groups=groups,
        )
        self.conv2 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride // 2,
            padding=padding,
            groups=groups,
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        return x


class AdditiveSequenceAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query_context = nn.Parameter(torch.Tensor(1, hidden_dim))
        nn.init.uniform_(self.query_context, -0.1, 0.1)  # Initialize query parameter
        self.V = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, H, mask=None):
        batch_size, seq_len, _ = H.size()

        Q = self.query_context.unsqueeze(0).repeat(batch_size, seq_len, 1)
        projected_H = self.W_h(H)

        E = self.V(torch.tanh(projected_H + Q))
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            E = E.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(E, dim=1)
        context_vector = torch.sum(attention_weights * H, dim=1)
        return context_vector


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
        conv_compressor=None,
        learning_rate=1e-5,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.embedding_f = torch.nn.Embedding(f_vocab_size, d_model)
        self.embedding_s = torch.nn.Embedding(s_vocab_size, d_model)

        self.pos_enc_f = PositionalEncoding(d_model, dropout=0.0, max_len=2048)
        self.pos_enc_s = PositionalEncoding(d_model, dropout=0.0, max_len=512)
        self.emb_scale = math.sqrt(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nheads,
            dim_feedforward=d_ff,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            batch_first=True,
            dropout=0.05,
        )
        self.conv_compressor = conv_compressor

        self.pooling = AdditiveSequenceAttention(d_model)
        d_fc_layers.insert(0, d_model)
        self.fc_block = nn.Sequential()
        for in_feat, out_feat in zip(d_fc_layers, d_fc_layers[1:]):
            self.fc_block.append(nn.Dropout(0.0))
            self.fc_block.append(nn.Linear(in_feat, out_feat))
            self.fc_block.append(nn.LeakyReLU())

        self.out_act = F.sigmoid if task == "binary" else None
        self.out_layer = nn.Linear(d_fc_layers[-1], 1)

        self.loss = nn.BCELoss() if task == "binary" else nn.HuberLoss(delta=1.0)
        self.learning_rate = learning_rate

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
        self.save_hyperparameters(ignore=["conv_compressor"])

    def forward(self, f_seq, s_seq):
        embed_f_seq = self.embedding_f(f_seq) * self.emb_scale
        embed_s_seq = self.embedding_s(s_seq) * self.emb_scale

        if self.conv_compressor:
            embed_f_seq = self.conv_compressor(embed_f_seq)

        embed_f_seq = self.pos_enc_f(embed_f_seq)
        embed_s_seq = self.pos_enc_s(embed_s_seq)

        transformer_out = self.transformer(
            embed_f_seq, embed_s_seq, src_is_causal=False, tgt_is_causal=False
        )
        mask_out = get_mask(transformer_out, padding_value=0)

        pooled_out = self.pooling(transformer_out, mask_out)

        fc_out = self.fc_block(pooled_out)
        out = self.out_layer(fc_out)

        if self.out_act:
            out = self.out_act(out)

        return out

    def training_step(self, batch, batch_idx):
        f_seq, s_seq, target = batch
        pred = self(f_seq, s_seq)
        loss = self.loss(pred, target)

        metrics_log = self.train_metrics_mc(pred, target)
        metrics_log["train_loss"] = loss

        self.log_dict(metrics_log, on_epoch=True, logger=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        f_seq, s_seq, target = batch
        pred = self(f_seq, s_seq)
        loss = self.loss(pred, target)

        metrics_log = self.val_metrics_mc(pred, target)

        metrics_log["val_loss"] = loss

        self.log_dict(metrics_log, on_epoch=True, logger=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        f_seq, s_seq, target = batch
        pred = self(f_seq, s_seq)
        loss = self.loss(pred, target)

        metrics_log = self.test_metrics_mc(pred, target)

        metrics_log["test_loss"] = loss

        self.log_dict(metrics_log, on_epoch=True, logger=True)

    def get_optimizers(self, weight_decay=0.01, muon_lr=0.02, adam_lr=0.001):
        muon_params = []
        adam_params = []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue

            if p.ndim == 2 and "embed" not in name:
                muon_params.append(p)
            else:
                adam_params.append(p)

        opt_muon = torch.optim.Muon(muon_params, lr=muon_lr, momentum=0.95)

        opt_adam = torch.optim.AdamW(
            adam_params, lr=adam_lr, betas=(0.9, 0.95), weight_decay=weight_decay
        )

        return opt_muon, opt_adam

    def configure_optimizers(self):
        opt_adam = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=0.01
        )

        adam_scheduler = get_warmup_cosine_scheduler(
            opt_adam,
            warmup_steps=40334 * 3,
            total_steps=40334 * (20 - 3),
            min_lr_ratio=0.01,
            start_factor=0.01,
        )

        lr_config = {
            "scheduler": adam_scheduler,
            "interval": "step",
            "frequency": 1,
            "name": None,
        }
        return [opt_adam], [lr_config]


if __name__ == "__main__":
    m = LigadTransformer(16, 16, 128, 128, 2, 5, 5, [64, 32])
    inp_data = torch.randint(1, 128, (2, 32))

    score = m(inp_data, inp_data)
    print(score)
    print(score.size())
