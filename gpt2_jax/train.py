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


@eqx.filter_jit(donate="all")
def train_step(
    flat_model,
    treedef_model,
    flat_opt_state,
    treedef_opt_state,
    optim,
    data,
    targets
):
    model = jtu.tree_unflatten(treedef_model, flat_model)
    opt_state = jtu.tree_unflatten(treedef_opt_state, flat_opt_state)

    loss, grads = compute_loss(model, data, targets)

    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)

    flat_update_model = jtu.tree_leaves(model)
    flat_update_opt_state = jtu.tree_leaves(opt_state)
    return loss, flat_update_model, flat_update_opt_state


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
    
    # Learning rate schedule with cosine decay
    schedule = optax.warmup_cosine_decay_schedule(
        max_lr,
        min_lr,
        warmup_steps=warmup_steps,
        decay_steps=(total_train_steps - warmup_steps),
    )

    optim = optax.chain(
        optax.adamw(schedule, mask=param_mask, b1=b1, b2=b2, weight_decay=weight_decay),
        optax.clip_by_global_norm(grad_clip_norm)
    )
    optim = optax.MultiSteps(optim, every_k_schedule=grad_accum_steps)
    
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    flat_model, treedef_model = jtu.tree_flatten(model)
    flat_opt_state, treedef_opt_state = jtu.tree_flatten(opt_state)
