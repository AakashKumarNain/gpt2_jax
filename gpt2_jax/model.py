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


def count_params(model):
    return sum(x.size for x in jtu.tree_leaves(eqx.filter(model, eqx.is_array)))


def get_weight_and_bias(module):
    if hasattr(module, "bias") and module.bias is not None:
        return module.weight, module.bias
    return module.weight


def set_weight_and_bias(weight, bias, key, mean=0.0, std=0.02):
    init = jax.nn.initializers.normal(stddev=std)
    weight = init(key=key, shape=weight.shape).astype(weight.dtype)
    if bias is not None:
        bias = jnp.zeros_like(bias, dtype=bias.dtype)
        return weight, bias
    return weight


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
        x = jax.nn.gelu(x)
        x = jax.vmap(self.proj)(x)
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
        
        self.wqkv = eqx.nn.Linear(config.embed_dim, 3 * config.embed_dim, key=key1) # 3 for qkv
        self.proj = eqx.nn.Linear(config.embed_dim, config.embed_dim, key=key2)

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
        attn = self.compute_attention(q, k, v, mask)
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
        x = x + self.attn(jax.vmap(self.norm_1)(x), mask=mask)
        x = x + self.mlp(jax.vmap(self.norm_2)(x))
        return x