import os
import sys
import time
import math
import numpy as np
from typing import Tuple, NamedTuple
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

import optax
import tiktoken
import equinox as eqx

from model import GPT
from dataset import SimpleDataLoader

from utils import is_layer
from utils import is_leaf
from utils import set_mask
from utils import get_weight_and_bias
from utils import set_weight_and_bias
from utils import scaled_dot_product_attention


@dataclass
class GPTConfig:
    block_size: int = 1024  # sequence length
    vocab_size: int = 50304
    num_layers: int = 12
    num_heads: int = 12
    embed_dim: int = 768 # embedding dimension for the tokens



def main(text_file_path):
    if not os.path.exists(text_file_path):
        print("Given file not found in the path. Please provide a correct file path.")
        sys.exit(1)
    
    config = GPTConfig()
    model = GPT(config, key=jax.random.PRNGKey(1))
    tokenizer = tiktoken.get_encoding("gpt2")

    batch_size = 8
    total_batch_size = 524288 # ideal batch size as in GPT2 paper
    num_devices = len(jax.device_count("gpu"))
    grad_accum_steps = total_batch_size // (batch_size * config.block_size * num_devices)

    assert total_batch_size % (batch_size * config.block_size * num_devices) == 0, \
        "make sure total_batch_size is divisible by B * T * num_devices"

    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 715
    total_train_steps = 19073
    b1 = 0.9
    b2 = 0.95
    weight_decay = 0.1
    grad_clip_norm = 1.0


    print("\n", "="*30, "Model config values", "="*30, "\n")
    print(f"Block size (Sequence length)           : {config.block_size}")
    print(f"Vocabulary size                        : {config.vocab_size}")
    print(f"Number of transformer layers           : {config.num_layers}")
    print(f"Number of heads                        : {config.num_heads}")
    print(f"Embedding dimension size               : {config.embed_dim}")

    print("\n", "="*30, "Hyper params values", "="*30, "\n")
    print(f"Batch size as per paper                : {total_batch_size}")
    print(f"Max batch size that can be fitted here : {batch_size}")
    print(f"Number of gradient accumulation steps  : {grad_accum_steps}")
    print(f"Number of training steps               : {total_train_steps}")
    print(f"Weight decay                           : {weight_decay}")
    print(f"Minimum learning rate                  : {min_lr:.6f}")
    print(f"Maximum learning rate                  : {max_lr:.6f}")
    print(f"Warmup steps                           : {warmup_steps}")
    print(f"Decay steps                            : {total_train_steps - warmup_steps}")
    print(f"Adam betas values                      : {b1=} {b2=}\n")
    print(f"Number of devices                      : {num_devices}")
    
    dl = SimpleDataLoader(
        batch_size=batch_size,
        seqlen=config.block_size,
        tokenizer=tokenizer,
        file_path=text_file_path,
        split="train"
    )


    # Filter out the parameters so that we apply weight decay to selected
    # parameters only
    param_mask = jtu.tree_map(
        set_mask, eqx.filter(model, eqx.is_array), is_leaf=is_layer)
    
    
