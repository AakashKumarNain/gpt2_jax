import equinox as eqx
from jax import tree_util as jtu


def is_layer(x):
    return isinstance(x, (eqx.nn.Linear, eqx.nn.Embedding, eqx.nn.LayerNorm))

def is_leaf(x):
    return x is None

def set_mask(x):
    if isinstance(x, eqx.nn.Linear):
        # Decay has to be applied only on the weights, and not the biases
        mask = jtu.tree_map(lambda _: True, x)
        mask = eqx.tree_at(lambda m: m.bias, mask, False, is_leaf=is_leaf)
        return mask
    elif isinstance(x, eqx.nn.Embedding):
        return jtu.tree_map(lambda _: True, x)
    else:
 
        return jtu.tree_map(lambda _: False, x)