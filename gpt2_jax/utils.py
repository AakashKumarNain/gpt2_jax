from typing import NamedTuple

import optax
import jax
import equinox as eqx
from jax import tree_util as jtu


######################## Equinox model utils ################################

def is_layer(x):
    """Check if the current pytree is an instance of any Equinox layer."""
    return isinstance(x, (eqx.nn.Linear, eqx.nn.Embedding, eqx.nn.LayerNorm))


def is_leaf(x):
    return x is None


def set_mask(x):
    """Sets the mask for certain parameters.
    
    There are scenarios where you want to filter out the parameters of the
    model for applying some specialized op. For example, in this case we
    are filtering our pytrees and masking certain parameters to avoid applying 
    `weight_decay` to these parameters. These parameters are:

    1. Linear layer -> Weight decay is only applied to the weights and not the bias
    2. Embedding -> Weight decay applied to the weights
    3. Any other layer e.g. LayerNorm -> No weight decay is applied
    """

    if isinstance(x, eqx.nn.Linear):
        # Decay has to be applied only on the weights, and not the biases
        mask = jtu.tree_map(lambda _: True, x)
        mask = eqx.tree_at(lambda m: m.bias, mask, False, is_leaf=is_leaf)
        return mask
    elif isinstance(x, eqx.nn.Embedding):
        return jtu.tree_map(lambda _: True, x)
    else:
        return jtu.tree_map(lambda _: False, x)



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
        bias = jnp.zeros_like(bias, dtype=weight.dtype)
        return weight, bias
    return weight


def scaled_dot_product_attention(query, key, value, mask=None, bias=None, is_causal=False, scale=None):
    attn_dtype = jnp.promote_types(query.dtype, jnp.float32)
    attn_weight = jnp.matmul(query, jnp.transpose(key, (0, 2, 1))).astype(attn_dtype)
    
    scale_factor = 1 / jnp.sqrt(query.shape[-1]) if scale is None else scale
    scale_factor = jnp.array(scale_factor, dtype=attn_weight.dtype)

    attn_weight *= scale_factor
    
    if bias is not None:
        attn_weight = (attn_weight + bias).astype(attn_weight.dtype)

    if mask is not None:
        assert mask.dtype == jnp.bool_
        large_negative_number = _get_large_negative(attn_weight.dtype)
        padded_attn_weight = jnp.where(mask, attn_weight, large_negative_number)
    else:
        padded_attn_weight = attn_weight

    def add_causal_mask(padded_attn_weight):
        S = query.shape[-2]
        T = key.shape[-2]
        mask = _get_causal_mask(S, T, attn_weight.dtype)
        mask = jnp.broadcast_to(mask, padded_attn_weight.shape)
        padded_attn_weight += mask
        return padded_attn_weight

    def no_casual_mask(padded_attn_weight):
        return padded_attn_weight


    padded_attn_weight = jax.lax.cond(
        is_causal,
        add_causal_mask,
        no_casual_mask,
        operand=padded_attn_weight
    )

    padded_attn_weight = padded_attn_weight.astype(jnp.float32)
    probs = jax.nn.softmax(padded_attn_weight, axis=-1).astype(key.dtype)
    out = jnp.matmul(probs, value)
    return out


############################################################################


class RecordNormState(NamedTuple):
    """Holds the norm of th gradients as jax arrays."""
    grad_norm: jax.Array


def record_norm():
    """Records the norm of th gradients in optax multi-transform chaining."""
    def init_fn(params):
        return RecordNormState(grad_norm=jnp.asarray(0.0))

    def update_fn(updates, state, params=None):
        norm = optax.tree_utils.tree_l2_norm(updates)
        return updates, RecordNormState(grad_norm=norm)

    return optax.GradientTransformation(init_fn, update_fn)