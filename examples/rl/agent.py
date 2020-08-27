import jax
import numpy as onp
import jax.numpy as jnp
import math

@jax.jit
def greedy_action(model, state):
  out = model(state)
  action = jnp.argmax(out, axis = 1).astype(jnp.int32)
  return action

def eps_greedy_action(state, optimizer, 
                      steps_done, eps_start, eps_end, eps_decay):
  decay_factor = math.exp(-1. * steps_done / eps_decay)
  eps_threshold = eps_end + (eps_start - eps_end) * decay_factor
  batch_size = state.shape[0]
  r = onp.random.uniform()
  if r < eps_threshold:
    rand_actions = onp.random.randint(low=0, high=4, size=(batch_size, ))
    return rand_actions
  else:
    greedy_actions = greedy_action(optimizer.target, state)
    greedy_actions = jax.device_get(greedy_actions)
    return greedy_actions
    # return greedy_actions.block_until_ready()
  # r = onp.random.uniform(size=(batch_size, ))
  # random_greedy_mask =  r < eps_threshold
  # rand_actions = onp.random.randint(low=0, high=4, size=(batch_size, ))
  # greedy_actions, out = greedy_action(optimizer.target, state)
  # actions = onp.where(random_greedy_mask, rand_actions, greedy_actions)
  # return actions