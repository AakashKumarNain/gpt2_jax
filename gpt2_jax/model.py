import os
import numpy as np
import jax
import math
import jax.numpy as jnp
import jax.tree_util as jtu
from typing import Tuple
from typing import NamedTuple

import equinox as eqx
from equinox._misc import default_floating_dtype

from utils import count_params
from utils import get_weight_and_bias
from utils import set_weight_and_bias
from utils import scaled_dot_product_attention


class MLP(eqx.Module):
    fc1: eqx.nn.Linear
    proj: eqx.nn.Linear
    
    def __init__(self, config, key, dtype=jnp.bfloat16):
        dtype = default_floating_dtype() if dtype is None else dtype
        key1, key2 = jax.random.split(key, 2)
        std = 0.02
        
        self.fc1 = eqx.nn.Linear(
            config.embed_dim, config.embed_dim * 4, key=key1, dtype=dtype
        )
        self.proj = eqx.nn.Linear(
            config.embed_dim * 4, config.embed_dim, key=key2, dtype=dtype
        )

        # Set the weights and bias of the linear layer as per the paper
        self.fc1 = eqx.tree_at(
            get_weight_and_bias,
            self.fc1,
            set_weight_and_bias(self.fc1.weight, self.fc1.bias, key1, std=std)
        )
        # Set the weights and bias of the projection layer as per the paper
        self.proj = eqx.tree_at(
            get_weight_and_bias,
            self.proj,
            set_weight_and_bias(
                self.proj.weight,
                self.proj.bias,
                key2,
                std * (2 * config.num_layers) ** -0.5,
            )
        )

    def __call__(self, x):
        x = jax.vmap(self.fc1)(x)
        x = jax.nn.gelu(x.astype(jnp.float32))
        x = jax.vmap(self.proj)(x.astype(jnp.bfloat16))
        return x


class CausalSelfAttention(eqx.Module):
    num_heads: int
    num_layers: int
    wqkv: eqx.nn.Linear
    proj: eqx.nn.Linear
    scale: float
    
    def __init__(self, config, key, dtype=jnp.bfloat16):
        assert config.embed_dim  % config.num_heads == 0
        dtype = default_floating_dtype() if dtype is None else dtype
        key1, key2 = jax.random.split(key, 2)

        self.num_heads = config.num_heads
        self.num_layers = config.num_layers
        self.scale = 1./ math.sqrt(config.embed_dim)
        
        self.wqkv = eqx.nn.Linear(config.embed_dim, 3 * config.embed_dim, key=key1, dtype=dtype) # 3 for qkv
        self.proj = eqx.nn.Linear(config.embed_dim, config.embed_dim, key=key2, dtype=dtype)

        self.wqkv = eqx.tree_at(
            get_weight_and_bias,
            self.wqkv,
            set_weight_and_bias(self.wqkv.weight, self.wqkv.bias, key1, std=0.02,)
        )
        self.proj = eqx.tree_at(
            get_weight_and_bias,
            self.proj,
            set_weight_and_bias(
                self.proj.weight,
                self.proj.bias,
                key2,
                std = 0.02 * (2 * config.num_layers) ** -0.5,
            )
        )

    def __call__(self, x, mask=None):
        # x is of shape [seqlen, embed_dim]
        # batch size will be handled by vmap
        T, C = x.shape
        x_dtype = x.dtype

        # 1. Calculate qkv
        qkv = jax.vmap(self.wqkv)(x)
        
        # 2. Split qkv into three vectors of equal depth
        q, k, v = jnp.split(qkv, 3, axis=1)

        # 3. Reshape q, k,v to move the heads to the batch dimension
        # so that we can calculate the attention for all heads in one go
        q = jnp.reshape(q, (T, self.num_heads, C // self.num_heads))
        k = jnp.reshape(k, (T, self.num_heads, C // self.num_heads))
        v = jnp.reshape(v, (T, self.num_heads, C // self.num_heads))

        # 4. Compute attention
        # TODO: Implement causal attention function
        attn = scaled_dot_product_attention(q, k, v, is_causal=True).astype(x_dtype)
        attn = jnp.reshape(jnp.transpose(attn, (1, 0, 2)), (T, -1))

        # 5. Projection
        out = jax.vmap(self.proj)(attn)
        return out


class TransformerBlock(eqx.Module):
    norm_1: eqx.nn.LayerNorm
    norm_2: eqx.nn.LayerNorm
    attn: CausalSelfAttention
    mlp: MLP

    def __init__(self, config, key, dtype=jnp.bfloat16):
        key1, key2 = jax.random.split(key, 2)
        self.norm_1 = eqx.nn.LayerNorm(config.embed_dim)
        self.attn = CausalSelfAttention(config, key=key1, dtype=dtype)
        self.norm_2 = eqx.nn.LayerNorm(config.embed_dim)
        self.mlp = MLP(config, key=key2, dtype=dtype)

    def __call__(self, x, mask=None):
        x_dtype = x.dtype
        x = jax.vmap(self.norm_1)(x.astype(jnp.float32)).astype(x_dtype)
        x = x + self.attn(x, mask=mask)
        x = jax.vmap(self.norm_2)(x.astype(jnp.float32)).astype(x_dtype)
        x = x + self.mlp(x)
        return x



class GPT(eqx.Module):
    block_size: int
    num_layers: int
    num_heads: int
    vocab_size: int
    tok_embed_and_head: eqx.nn.Shared
    pos_embed: eqx.nn.Embedding
    tf_blocks: TransformerBlock
    norm: eqx.nn.LayerNorm

    def __init__(self, config, key, dtype=jnp.bfloat16):
        self.block_size = config.block_size
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.vocab_size = config.vocab_size

        keys = jax.random.split(key, config.num_layers + 3)
        key1, key2, key3, tf_keys = keys[0], keys[1], keys[2], keys[3:]

        self.norm = eqx.nn.LayerNorm(config.embed_dim)

        make_layers = lambda k: TransformerBlock(config, key=k, dtype=dtype)
        self.tf_blocks = eqx.filter_vmap(make_layers)(tf_keys)
        del make_layers
        
        self.pos_embed = eqx.nn.Embedding(config.block_size, config.embed_dim, key=key1)
        self.pos_embed = eqx.tree_at(get_weight_and_bias,
            self.pos_embed, set_weight_and_bias(self.pos_embed.weight, None, key1)
        )

        tok_embed = eqx.nn.Embedding(config.vocab_size, config.embed_dim, key=key2)
        tok_embed = eqx.tree_at(get_weight_and_bias,
            tok_embed, set_weight_and_bias(tok_embed.weight, None, key2)
        )

        lm_head = eqx.nn.Linear(config.embed_dim, config.vocab_size, use_bias=False, key=key3)
        dst = lambda embed_and_linear: embed_and_linear[1].weight
        src = lambda embed_and_linear: embed_and_linear[0].weight
        self.tok_embed_and_head = eqx.nn.Shared((tok_embed, lm_head), dst, src)

    def __call__(self, idx, mask=None):
        tok_embed, lm_head = self.tok_embed_and_head()
        seqlen = idx.shape[-1]
        pos = jnp.arange(0, seqlen, dtype=jnp.int32)
        
        # idx is of shape (seqlen,)
        pos_embed = jax.vmap(self.pos_embed)(pos)
        tok_embed = jax.vmap(tok_embed)(idx)

        # 2. Add position to token embeddings
        x = pos_embed + tok_embed

        # 3. Partition the TransformerLayers into static and dynamic parts
        # and pass the previous output through transformer blocks
        dynamic_layers, static_layers = eqx.partition(self.tf_blocks, eqx.is_array)
        layer_idx = 0

        def f(_x, _dynamic_l):
            layer = eqx.combine(_dynamic_l, static_layers)
            x, layer_idx = _x
            x = layer(x)
            return (x, layer_idx + 1), None

        (x, layer_idx), _ = jax.lax.scan(f, (x, layer_idx), dynamic_layers)

        # 4. Final pre-layer norm
        x = jax.vmap(self.norm)(x)

        # 5. Classification head
        logits = jax.vmap(lm_head)(x)
        return logits