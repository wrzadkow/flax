# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from flax.core import Scope, init, apply, unfreeze, lift, nn
from typing import Any, Sequence, Callable

import jax
from jax import lax, random, numpy as jnp

from functools import partial


Array = Any

def mlp_vmap(scope: Scope, x: Array,
             sizes: Sequence[int] = (8, 1),
             act_fn: Callable[[Array], Array] = nn.relu,
             share_params: bool = False):
  if share_params:
    dense_vmap = lift.vmap(nn.dense,
                           in_axes=(0, None),
                           variable_in_axes={'param': None},
                           variable_out_axes={'param': None},
                           split_rngs={'param': False})
  else:
    dense_vmap = lift.vmap(nn.dense,
                           in_axes=(0, None),
                           variable_in_axes={'param': 0},
                           variable_out_axes={'param': 0},
                           split_rngs={'param': True})

  # hidden layers
  for size in sizes[:-1]:
    x = scope.child(dense_vmap)(x, size)
    x = act_fn(x)

  # output layer
  return scope.child(dense_vmap)(x, sizes[-1])

if __name__ == "__main__":
  x = random.normal(random.PRNGKey(0), (1, 4))
  x = jnp.concatenate([x, x], 0)

  print('shared params: (same inputs, same outputs)')
  y, params = init(mlp_vmap)(random.PRNGKey(1), x, share_params=True)
  print(y)
  print(jax.tree_map(jnp.shape, unfreeze(params)))

  print('unshared params: (sampe inputs, different outputs, extra dim in params)')
  y, params = init(mlp_vmap)(random.PRNGKey(1), x, share_params=False)
  print(y)
  print(jax.tree_map(jnp.shape, unfreeze(params)))