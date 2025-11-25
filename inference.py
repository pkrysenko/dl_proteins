import sys
import sentencepiece as spm
import torch

from hydra.utils import instantiate
from omegaconf import OmegaConf
from arch.ligads import LigadTransformer


# This function is neede to load weights without additional info from pl.LightningModule
def load_model_with_weights(cfg, weight_path):
    model = instantiate(cfg.model)
    checkpoint = torch.load(weight_path)

    model.embedding_f.load_state_dict(
        {
            k.replace("embedding_f.", ""): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("embedding_f.")
        }
    )
    model.embedding_s.load_state_dict(
        {
            k.replace("embedding_s.", ""): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("embedding_s.")
        }
    )
    model.transformer.load_state_dict(
        {
            k.replace("transformer.", ""): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("transformer.")
        }
    )
    model.fc_block.load_state_dict(
        {
            k.replace("fc_block.", ""): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("fc_block.")
        }
    )
    model.out_layer.load_state_dict(
        {
            k.replace("out_layer.", ""): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("out_layer.")
        }
    )
    model.eval()
    return model


def inference(cfg, model, sp_model_f, sp_model_s):
    while True:
        seq_f = input(f"Enter sequence {cfg.dataloader.f_seq_column}: ")
        seq_s = input(f"Enter sequence {cfg.dataloader.s_seq_column}: ")
        tokens_f = sp_model_f.encode(seq_f, out_type=int)
        tokens_s = sp_model_s.encode(seq_s, out_type=int)

        tensor_f = torch.tensor(tokens_f).unsqueeze(0)  # Add batch dimension
        tensor_s = torch.tensor(tokens_s).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = model(tensor_f, tensor_s)
        print(f"Model output: {output.item()}")


if __name__ == "__main__":
    cfg = OmegaConf.load(sys.argv[1])
    weight_path = sys.argv[2]
    model = load_model_with_weights(cfg, weight_path)
    sp_model_f = spm.SentencePieceProcessor(model_file=sys.argv[3])
    sp_model_s = spm.SentencePieceProcessor(model_file=sys.argv[4])
    try:
        inference(cfg, model, sp_model_f, sp_model_s)
    except KeyboardInterrupt:
        print("\nExiting inference.")
