import os
import numpy as np
import jax
import math
import jax.numpy as jnp
import jax.tree_util as jtu
from typing import Tuple
from typing import NamedTuple

import equinox as eqx


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