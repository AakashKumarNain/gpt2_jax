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
    
    dl = SimpleDataLoader(
        batch_size=batch_size,
        seqlen=config.block_size,
        tokenizer=tokenizer,
        file_path=text_file_path,
        split="train"
    )

    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 715
    max_steps = 19073

    # Filter out the parameters so that we apply weight decay to selected
    # parameters only
    param_mask = jtu.tree_map(
        set_mask, eqx.filter(model, eqx.is_array), is_leaf=is_layer)
    
    