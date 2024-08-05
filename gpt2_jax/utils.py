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


############################################################################


class RecordNormState(NamedTuple):
    grad_norm: jax.Array

def record_norm():
    def init_fn(params):
        return RecordNormState(grad_norm=jnp.asarray(0.0))

    def update_fn(updates, state, params=None):
        norm = optax.tree_utils.tree_l2_norm(updates)
        # jax.debug.print("grad_norm = {norm}", norm=norm)
        return updates, RecordNormState(grad_norm=norm)

    return optax.GradientTransformation(init_fn, update_fn)