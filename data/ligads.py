import os

import torch
import random
import pandas as pd
import numpy as np
import lightning.pytorch as pl
import sentencepiece as spm

from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader, Sampler

from data.utils import (
    generate_processed_data_name,
    generate_spm_prefix,
    train_sp_model,
    train_validate_test_split,
)

torch.set_float32_matmul_precision("high")


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


class SequenceLengthSampler(Sampler):

    def __init__(
        self,
        data_source,
        lengths,
        batch_size,
        shuffle=True,
        bin_size=0,
        auto_bins=False,
    ):
        self.data_source = data_source
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bin_size = bin_size
        self.auto_bins = auto_bins

        if auto_bins:
            self.buckets = self._build_auto_bins()
        else:
            self.buckets = self._build_fixed_or_exact_bins()

        if self.shuffle:
            for b in self.buckets:
                random.shuffle(b)
            random.shuffle(self.buckets)

    def __iter__(self):
        batches = []
        for bucket in self.buckets:

            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i : i + self.batch_size]
                batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        return sum(
            (len(bucket) + self.batch_size - 1) // self.batch_size
            for bucket in self.buckets
        )

    def _build_auto_bins(self):
        lengths = np.array(self.lengths)
        indices = np.arange(len(lengths))

        order = np.argsort(lengths)
        sorted_indices = indices[order]

        bins = np.array_split(sorted_indices, self.auto_bins)
        return [list(b.tolist()) for b in bins if len(b) > 0]

    def _build_fixed_or_exact_bins(self):
        buckets = defaultdict(list)

        for idx, l in enumerate(self.lengths):
            if self.bin_size:
                key = l // self.bin_size  # fixed bins
            else:
                key = l  # exact lengths
            buckets[key].append(idx)

        return list(buckets.values())


class LigadsDL(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        f_seq_column,
        s_seq_column,
        out_column,
        f_vocab_size,
        s_vocab_size,
        batch_size,
        task="binary",
    ):
        super().__init__()
        self.dataset_name = os.path.basename(data_dir)
        self.data_dir = data_dir
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
        if os.path.exists(os.path.join(self.data_dir, PROCESSED_DATA_DIR)):
            return

        train, val, test = (
            pd.read_csv(os.path.join(self.data_dir, "train.csv")),
            pd.read_csv(os.path.join(self.data_dir, "val.csv")),
            pd.read_csv(os.path.join(self.data_dir, "test.csv")),
        )

        os.makedirs(os.path.join(self.data_dir, BPE_MODEL_SAVE_DIR), exist_ok=True)
        path_to_vocab_model_f = os.path.join(
            self.data_dir, BPE_MODEL_SAVE_DIR, self.sp_prefix_f
        )
        if not os.path.exists(path_to_vocab_model_f + ".model"):
            combined_dataset = pd.concat([train, val, test])
            train_sp_model(
                path_to_vocab_model_f,
                combined_dataset[self.f_seq_column],
                self.f_vocab_size,
            )

        path_to_vocab_model_s = os.path.join(
            self.data_dir, BPE_MODEL_SAVE_DIR, self.sp_prefix_s
        )
        if not os.path.exists(path_to_vocab_model_s + ".model"):
            combined_dataset = pd.concat([train, val, test])
            train_sp_model(
                path_to_vocab_model_s,
                combined_dataset[self.s_seq_column],
                self.s_vocab_size,
            )

        spp_f = spm.SentencePieceProcessor(model_file=path_to_vocab_model_f + ".model")
        spp_s = spm.SentencePieceProcessor(model_file=path_to_vocab_model_s + ".model")

        os.makedirs(os.path.join(self.data_dir, PROCESSED_DATA_DIR), exist_ok=True)

        for df, df_name in zip([train, val, test], ["train", "val", "test"]):

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
            ].to_json(
                os.path.join(self.data_dir, PROCESSED_DATA_DIR, df_name + ".json")
            )

    def setup(self, stage):
        cached_dir = os.path.join(self.data_dir, PROCESSED_DATA_DIR)
        train_df, val_df, test_df = (
            pd.read_json(os.path.join(cached_dir, "train.json")),
            pd.read_json(os.path.join(cached_dir, "val.json")),
            pd.read_json(os.path.join(cached_dir, "test.json")),
        )

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
        # [print(row[0].size()) for row in self.ligad_train]
        lengthes = [row[0].size()[0] for row in self.ligad_train]
        sampler = SequenceLengthSampler(
            data_source=self.ligad_train,
            lengths=lengthes,
            batch_size=self.batch_size,
            auto_bins=False,
            bin_size=32,
        )
        return DataLoader(
            self.ligad_train,
            collate_fn=pad_collate,
            pin_memory=True,
            batch_sampler=sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ligad_valid,
            batch_size=self.batch_size,
            collate_fn=pad_collate,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.ligad_test,
            batch_size=self.batch_size,
            collate_fn=pad_collate,
            drop_last=True,
            pin_memory=True,
        )


if __name__ == "__main__":

    spm.SentencePieceTrainer()
