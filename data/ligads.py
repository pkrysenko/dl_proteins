import os

import torch
import pandas as pd
import lightning.pytorch as pl
import sentencepiece as spm

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from data.utils import (
    generate_processed_data_name,
    generate_spm_prefix,
    train_sp_model,
    train_validate_test_split,
)


BPE_MODEL_SAVE_DIR = "bpe_models"
PROCESSED_DATA_DIR = "processed_data"

binary_score_converter = {"yes": 1, "no": 0}


def pad_collate(batch):

    f_seq = [item[0] for item in batch]
    f_seq = pad_sequence(f_seq, batch_first=True)

    s_seq = [item[1] for item in batch]
    s_seq = pad_sequence(s_seq, batch_first=True)

    score = torch.FloatTensor([item[2] for item in batch]).unsqueeze(1)

    return [f_seq, s_seq, score]


class BinaryLigadsDL(pl.LightningDataModule):
    def __init__(
        self,
        csv_path,
        f_seq_column,
        s_seq_column,
        out_column,
        f_vocab_size,
        s_vocab_size,
        batch_size,
        task="binary",
    ):
        super().__init__()
        self.dataset_name = os.path.basename(csv_path[:-4])
        self.csv_path = csv_path
        self.f_seq_column = f_seq_column
        self.s_seq_column = s_seq_column
        self.out_column = out_column
        self.f_vocab_size = f_vocab_size
        self.s_vocab_size = s_vocab_size
        self.batch_size = batch_size

        self.sp_prefix_f = generate_spm_prefix(
            self.dataset_name, self.f_seq_column, self.f_vocab_size
        )
        self.sp_prefix_s = generate_spm_prefix(
            self.dataset_name, self.s_seq_column, self.s_vocab_size
        )
        self.data_processed_name = generate_processed_data_name(
            self.dataset_name, self.sp_prefix_f, self.sp_prefix_s
        )

    def prepare_data(self):
        df = pd.read_csv(self.csv_path).head(1000)

        path_to_vocab_model_f = os.path.join(BPE_MODEL_SAVE_DIR, self.sp_prefix_f)
        if not os.path.exists(path_to_vocab_model_f):
            train_sp_model(
                path_to_vocab_model_f, df[self.f_seq_column], self.f_vocab_size
            )

        path_to_vocab_model_s = os.path.join(BPE_MODEL_SAVE_DIR, self.sp_prefix_s)
        if not os.path.exists(path_to_vocab_model_s):
            train_sp_model(
                path_to_vocab_model_s, df[self.s_seq_column], self.s_vocab_size
            )

        spp_f = spm.SentencePieceProcessor(model_file=path_to_vocab_model_f + ".model")
        spp_s = spm.SentencePieceProcessor(model_file=path_to_vocab_model_s + ".model")

        df[f"{self.f_seq_column}_encoded"] = df[self.f_seq_column].apply(
            lambda x: spp_f.encode(x)
        )
        df[f"{self.s_seq_column}_encoded"] = df[self.s_seq_column].apply(
            lambda x: spp_s.encode(x)
        )

        df[
            [
                f"{self.f_seq_column}_encoded",
                f"{self.s_seq_column}_encoded",
                f"{self.out_column}",
            ]
        ].to_json(os.path.join(PROCESSED_DATA_DIR, self.data_processed_name))

    def setup(self, stage):
        processed_data = pd.read_json(
            os.path.join(PROCESSED_DATA_DIR, self.data_processed_name)
        )
        train_df, val_df, test_df = train_validate_test_split(processed_data)

        if stage == "fit":
            self.ligad_train = [
                (torch.IntTensor(f_seq), torch.IntTensor(s_seq), float(score))
                for f_seq, s_seq, score in zip(
                    train_df[f"{self.f_seq_column}_encoded"],
                    train_df[f"{self.s_seq_column}_encoded"],
                    train_df[f"{self.out_column}"],
                )
            ]
            self.ligad_valid = [
                (torch.IntTensor(f_seq), torch.IntTensor(s_seq), float(score))
                for f_seq, s_seq, score in zip(
                    val_df[f"{self.f_seq_column}_encoded"],
                    val_df[f"{self.s_seq_column}_encoded"],
                    val_df[f"{self.out_column}"],
                )
            ]
        if stage == "test":
            self.ligad_test = [
                (torch.IntTensor(f_seq), torch.IntTensor(s_seq), float(score))
                for f_seq, s_seq, score in zip(
                    test_df[f"{self.f_seq_column}_encoded"],
                    test_df[f"{self.s_seq_column}_encoded"],
                    test_df[f"{self.out_column}"],
                )
            ]

    def train_dataloader(self):
        return DataLoader(
            self.ligad_train,
            batch_size=self.batch_size,
            collate_fn=pad_collate,
            num_workers=16,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ligad_valid,
            batch_size=self.batch_size,
            collate_fn=pad_collate,
            num_workers=16,
        )

    def test_dataloader(self):
        return DataLoader(
            self.ligad_test,
            batch_size=self.batch_size,
            collate_fn=pad_collate,
            num_workers=16,
        )


if __name__ == "__main__":

    spm.SentencePieceTrainer()
