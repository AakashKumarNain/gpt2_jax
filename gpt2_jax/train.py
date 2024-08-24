import os
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    # '--xla_gpu_triton_gemm_any=True '
    # '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
    '--xla_gpu_enable_pipelined_all_gather=true '
    '--xla_gpu_enable_pipelined_reduce_scatter=true '
    '--xla_gpu_enable_pipelined_all_reduce=true '
    '--xla_gpu_enable_pipelined_collectives=false '
    '--xla_gpu_enable_all_gather_combine_by_dim=false '
    '--xla_gpu_enable_reduce_scatter_combine_by_dim=false '
    '--xla_gpu_all_gather_combine_threshold_bytes=8589934592 '
    '--xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592 '
    '--xla_gpu_all_reduce_combine_threshold_bytes=8589934592 '
    '--xla_gpu_multi_streamed_windowed_einsum=true '
    '--xla_gpu_threshold_for_windowed_einsum_mib=0 '
)

import sys
import time
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
from utils import set_mask
from utils import count_params


@dataclass
class GPTConfig:
    block_size: int = 1024  # sequence length
    vocab_size: int = 50304
    num_layers: int = 12
    num_heads: int = 12
    embed_dim: int = 768 # embedding dimension for the tokens


@eqx.filter_value_and_grad
def compute_loss(model, inputs, labels):
    """Computes cross entropy loss for a batch of preds and targets."""
    logits = eqx.filter_vmap(model)(inputs).astype(jnp.float32)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return jnp.mean(loss)


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
    tokenizer = tiktoken.get_encoding("gpt2")

    batch_size = 8 # batch size that can fit on a single A100 40G GPU
    total_batch_size = 524288 # ideal batch size as in GPT2 paper
    num_devices = jax.device_count("gpu")
    grad_accum_steps = total_batch_size // (batch_size * config.block_size * num_devices)

    assert total_batch_size % (batch_size * config.block_size * num_devices) == 0, \
        "make sure total_batch_size is divisible by B * T * num_devices"

    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 10 #715
    total_train_steps = 200 #19073
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
    print(f"AdamW betas values                     : {b1=} {b2=}\n")
    print(f"Number of devices                      : {num_devices}")
    
    dl = SimpleDataLoader(
        batch_size=batch_size,
        seqlen=config.block_size,
        tokenizer=tokenizer,
        file_path=text_file_path,
        split="train"
    )

    print("\nLoading GPT2 model...")
    model = GPT(config, key=jax.random.PRNGKey(1))
    print(f"Number of parameters in the model       : {(count_params(model)/1e6):.2f} M")
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

    # Flatten the model and optimizer to avoid the little
    # overhead we get every time we do a forward and a backward pass.
    flat_model, treedef_model = jtu.tree_flatten(model)
    flat_opt_state, treedef_opt_state = jtu.tree_flatten(opt_state)

    print("Training...\n")
    for step in range(total_train_steps):
        start = time.time()

        for _ in range(grad_accum_steps):
            batch_inputs, batch_targets = dl.next_batch()
            loss, flat_model, flat_opt_state = train_step(
                    flat_model,
                    treedef_model,
                    flat_opt_state,
                    treedef_opt_state,
                    optim,
                    batch_inputs,
                    batch_targets,
                )

        end = time.time()
        dt = end - start
        tokens_processed = config.block_size * batch_size * grad_accum_steps * num_devices
        tokens_per_sec = int(tokens_processed / dt)
        print(f"Step: {step:<5d} | Loss: {loss:<10.4f}  |  time_taken: {dt :<5.2f} s  |  tok/sec: {tokens_per_sec:,}")


if __name__ == "__main__":
    main(text_file_path="input.txt")