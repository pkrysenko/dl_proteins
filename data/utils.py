import numpy as np
import sentencepiece as sp


def iter_pipeline(text):
    for line in text:
        line = line.rstrip()
        yield line


def generate_processed_data_name(csv_name, sp_prefix_f, sp_prefix_s):
    return f"{csv_name}_{sp_prefix_f}_{sp_prefix_s}.json"


def generate_spm_prefix(csv_name, column_name, vocab_size):
    return f"{csv_name}_{column_name}_{vocab_size}"


def train_sp_model(path_to_save, text, vocab_size):
    sp.SentencePieceTrainer.train(
        sentence_iterator=iter_pipeline(text),
        model_prefix=path_to_save,
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type="bpe",
    )


def train_validate_test_split(df, train_percent=0.7, validate_percent=0.2, seed=None):
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test
