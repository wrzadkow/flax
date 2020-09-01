import jax.numpy as jnp
import numpy as onp
import jax
import jax.random
import flax
from flax import nn
import time
import math

from numpy_memory import NumpyMemory
from model import create_model, create_optimizer
from agent import eps_greedy_action
from remote import RemoteSimulator
from test_episodes import test

@jax.jit
def train_step(
  optimizer : flax.optim.base.Optimizer,
  target_model : nn.base.Model,
  cur_states: onp.ndarray,
  next_states: onp.ndarray,
  actions: onp.ndarray,
  rewards: onp.ndarray,
  terminal_mask : onp.ndarray,
  gamma : float):
  """
  states: shape (minibatch, 84, 84, 4)
  actions: shape (minibatch, )
  rewards: shape (minibatch, )
  """
  print("compile")
  batch = cur_states, next_states, actions, rewards, terminal_mask
  batch_size = BATCH_SIZE
  iterations = cur_states.shape[0] // batch_size
  batch = jax.tree_map(
    lambda x: x.reshape((iterations, batch_size) + x.shape[1:]), batch)
  loss = 0.0
  for row in zip(*batch):
    cur_states, next_states, actions, rewards, terminal_mask = row

    def loss_fn(
      policy_model,
      target_model,
      cur_states,
      next_states,
      actions,
      rewards,
      terminal_mask,
      gamma):
      out_cur, out_next = policy_model(cur_states), target_model(next_states)
      best_continuations = jnp.max(out_next, axis=1)
      # zero best continuation for terminal_states using mask
      best_continuations = jnp.where(terminal_mask, 0.0, best_continuations)
      # right-hand-side Bellman equation
      targets = rewards + gamma * best_continuations
      current_values = jax.vmap(lambda x, a: x[a])(out_cur, actions)

      def hubber_loss(x, y):
        error = x - y
        loss = jnp.where(
          jnp.abs(error) > 1.0, jnp.abs(error) - 0.5, 0.5 * jnp.square(error))
        return loss

      return jnp.mean(hubber_loss(targets, current_values), axis=0)

    grad_fn = jax.value_and_grad(loss_fn)
    l, grad = grad_fn(
      optimizer.target,
      target_model,
      cur_states,
      next_states,
      actions,
      rewards,
      terminal_mask,
      gamma)
    loss += l
    optimizer = optimizer.apply_gradient(grad)
  return optimizer, loss


@jax.jit
def train_step_prioritized(
  optimizer : flax.optim.base.Optimizer,
  target_model : nn.base.Model,
  cur_states: onp.ndarray,
  next_states: onp.ndarray,
  actions: onp.ndarray,
  rewards: onp.ndarray,
  terminal_mask : onp.ndarray,
  gamma : float,
  beta : float,
  buffer_len : int,
  probs: onp.ndarray):
  """
  Main compilable train step. The loop over num_agents batches is included here
  (as opposed to the training loop) for increased speed.
  Also computes importance sampling weights and uses them.

  Args:
    optimizer: optimizer for the policy model
    target_model: target model
    cur_states: shape (batch_size*num_agents, 84, 84, 4)
    next_states: shape (batch_size*num_agents, 84, 84, 4)
    actions: shape (batch_size*num_agents, )
    rewards: shape (batch_size*num_agents, )
    beta: exponent in importance sampling
    buffer_len: current length of the buffer that samples come from, needed for
                importance sampling
    probs: probabilities with which the samples have been drawn from the 
           replay buffer, shape (batch_size*num_agents, )

  Returns:
    optimizer: new optimizer after parameters update 
    loss: loss summed over training steps
    d_priorities: new priorities used later to update replay buffer 
    
  """
  print("compile")
  # print(f" probs shape {probs.shape}")
  batch = cur_states, next_states, actions, rewards, terminal_mask, probs
  batch_size = BATCH_SIZE
  iterations = cur_states.shape[0] // batch_size
  batch = jax.tree_map(
    lambda x: x.reshape((iterations, batch_size) + x.shape[1:]), batch)
  loss = 0.0
  d_priorities = []
  for row in zip(*batch):
    cur_states, next_states, actions, rewards, terminal_mask, probs = row

    def loss_fn(
      policy_model,
      target_model,
      cur_states,
      next_states,
      actions,
      rewards,
      terminal_mask,
      gamma):
      out_cur, out_next = policy_model(cur_states), target_model(next_states)
      best_continuations = jnp.max(out_next, axis=1)
      # zero best continuation for terminal_states using mask
      best_continuations = jnp.where(terminal_mask, 0.0, best_continuations)
      # right-hand-side Bellman equation
      targets = rewards + gamma * best_continuations
      current_values = jax.vmap(lambda x, a: x[a])(out_cur, actions)

      def hubber_loss(x, y):
        error = x - y
        loss = jnp.where(
          jnp.abs(error) > 1.0, jnp.abs(error) - 0.5, 0.5 * jnp.square(error))
        return loss

      huber_losses = hubber_loss(targets, current_values)
      weights = ((1/buffer_len) * (1/probs))**beta
      max_weight = jnp.amax(weights)
      weights /= max_weight #normalize to avoid huge updates
      # print(f"weights shape {weights.shape}, weights {weights}")
      total_loss = jnp.sum(huber_losses * weights, axis=0)
      # print(f"total loss {total_loss}")
      cur_target_differences = jnp.abs(targets - current_values)
      # return total_loss, (cur_target_differences, weights)
      return total_loss, cur_target_differences

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (l, d), grad = grad_fn(
      optimizer.target,
      target_model,
      cur_states,
      next_states,
      actions,
      rewards,
      terminal_mask,
      gamma)
    d_priorities.append(d)
    loss += l
    # print(f'loss {l}, shape {l.shape}')  #good
    # print(f'd {d} shape {d.shape}') #good
    # print(f'grad type {type(grad)}, grad {grad}') #good
    optimizer = optimizer.apply_gradient(grad)
    # print(f"priorities before concatenating len {len(d_priorities)}, shape of first {d_priorities[0].shape}")
  return optimizer, loss, jnp.concatenate(d_priorities)


def sample_and_train(
  memory : NumpyMemory, 
  policy_optimizer : flax.optim.base.Optimizer, 
  target_model : nn.base.Model,
  batch_size : int, 
  gamma : float,
  prioritized : bool = False,
  step : int = None,
  beta_start : float = 0.,
  beta_end : float = 0.,
  beta_decay : int = 0):

  if prioritized:
    with jax.profiler.TraceContext("sampling"):
      exponent = ALPHA_0
      transitions = memory.sample(batch_size, exponent)
      cur_states, next_states, actions, rewards, terminal_mask, \
      probs, indices = transitions

    beta = beta_end + (beta_start - beta_end)*math.exp(-step/beta_decay)

    policy_optimizer, loss, d_priorities = train_step_prioritized(
    policy_optimizer,
    target_model,
    cur_states,
    next_states,
    actions,
    rewards,
    terminal_mask,
    gamma,
    beta,
    len(memory),
    probs)
    memory.update_priorities(indices, d_priorities)

  else:
    with jax.profiler.TraceContext("sampling"):
      transitions = memory.sample(batch_size)
      cur_states, next_states, actions, rewards, terminal_mask = transitions
    
    policy_optimizer, loss = train_step(
      policy_optimizer,
      target_model,
      cur_states,
      next_states,
      actions,
      rewards,
      terminal_mask,
      gamma)
  return policy_optimizer, loss


def train(
  policy_optimizer : flax.optim.base.Optimizer, 
  target_model : nn.base.Model,
  steps_total : int,
  num_agents : int,
  prioritized : bool = False):
  scores = []
  print(f"Using {num_agents} environments")
  if prioritized:
    memory = NumpyMemory(MEMORY_SIZE, prioritized=True)
  else:
    memory = NumpyMemory(MEMORY_SIZE)
  simulators = [RemoteSimulator() for i in range(num_agents)]
  t1 = time.time()
  for s in range(steps_total // num_agents):
    # print(s)
    if (s + 1) % (10000 // num_agents) == 0:
      print(f"Frames processed {s*num_agents}, time elapsed {time.time()-t1}")
      t1 = time.time()
    if (s + 1) % (50000 // num_agents) == 0:
      test(1, policy_optimizer.target, render=False)

    # 1. collect the states from simulators
    with jax.profiler.TraceContext("running simulators"):
      states = []
      for sim in simulators:
        state = sim.conn.recv()
        states.append(state)
      states = onp.concatenate(states, axis=0)

    # 2. epsilon - greedy actions based on collected states
    with jax.profiler.TraceContext("eps_greedy_actions"):
      actions = eps_greedy_action(
        states,
        policy_optimizer,
        s * num_agents,
        EPS_START,
        EPS_END,
        EPS_DECAY)
      # block_until_ready() for profiling used in greedy_actions()
      for i, sim in enumerate(simulators):
        action = actions[i]
        sim.conn.send(action)

    # 3. run num_agents train steps
    if len(memory) > INITIAL_MEMORY:
      # losses = []
      # for i in range(num_agents):
      for i in range(1):
        with jax.profiler.TraceContext("train_step"):
          if not prioritized:
            policy_optimizer, loss = sample_and_train(
              memory,
              policy_optimizer,
              target_model,
              num_agents * BATCH_SIZE,
              GAMMA)
            # loss.block_until_ready()
          else:
            policy_optimizer, loss = sample_and_train(
              memory,
              policy_optimizer,
              target_model,
              num_agents * BATCH_SIZE,
              GAMMA,
              prioritized,
              s*num_agents,
              BETA_START,
              BETA_END,
              BETA_DECAY)

      if s * num_agents % TARGET_UPDATE <= num_agents:
        # copy policy model parameters to target model
        target_model = policy_optimizer.target

    # 4. collect memories
    # print("4. collect memories")
    with jax.profiler.TraceContext("collecting experience"):
      for sim in simulators:
        sample = sim.conn.recv()
        memory.push(*sample)

  return policy_optimizer, scores, memory


BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.02
EPS_DECAY = 1000000
TARGET_UPDATE = 1000
LR = 1e-4
INITIAL_MEMORY = 10000
MEMORY_SIZE = 10 * INITIAL_MEMORY

#exponent in sampling:
# prob[i] = priorities[i]**\alpha/(\sum_j priorities[j]**\alpha)
ALPHA_0 = 0.6 #arXiv:1511.0595
#\beta used in importance sampling
BETA_START = 0.5 #arXiv:1511.0595
BETA_END =  1. #arXiv:1511.0595
#for simplicity, let's anneal \beta
#at the same speed as \epsilon
BETA_DECAY = EPS_DECAY
LR_REDUCTION = 4 # PER typically yields larger gradients (selects higher-error
                 # samples from the buffer)
LR /= LR_REDUCTION

key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
policy_model = create_model(subkey)
policy_optimizer = create_optimizer(policy_model, learning_rate=LR)
target_model = policy_model
del policy_model

if __name__ == "__main__":
  num_agents = 16
  total_frames = 4000000
  prioritized = True
  print(f"prioritized: {prioritized}")
  policy_optimizer, scores, mem = train(policy_optimizer, target_model, 
                  total_frames, num_agents, prioritized)