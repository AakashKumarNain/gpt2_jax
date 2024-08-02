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