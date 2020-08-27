import jax.numpy as jnp
import numpy as onp
import jax
import jax.random
import flax
from flax import nn
import time
from typing import Tuple

from numpy_memory import NumpyMemory
from model import create_model, create_optimizer
from agent import eps_greedy_action
from remote import RemoteSimulator
from test_episodes import test

@jax.jit
def train_step(
  optimizer : flax.optim.base.Optimizer,
  target_model : nn.base.Model,
  transitions : Tuple[onp.ndarray, onp.ndarray, onp.ndarray, 
                      onp.ndarray, onp.ndarray],
  # target_model : nn.base.Model,
  # cur_states: onp.ndarray,
  # next_states: onp.ndarray,
  # actions: onp.ndarray,
  # rewards: onp.ndarray,
  # terminal_mask : onp.ndarray,
  gamma : float):
  """
  Compilable train step. The loop over num_agents batches is included here
  (as opposed to the Python training loop) for increased speed.

  Args:
    optimizer: optimizer for the policy model
    target_model: target model
    transitions: Tuple of the following five elements forming the experience: 
                  cur_states: shape (batch_size*num_agents, 84, 84, 4)
                  next_states: shape (batch_size*num_agents, 84, 84, 4)
                  actions: shape (batch_size*num_agents, )
                  rewards: shape (batch_size*num_agents, )
                  terminal_mask: (batch_size*num_agents, )
    gamma: discount factor

  Returns:
    optimizer: new optimizer after the parameters update 
    loss: loss summed over training steps
  """
  print("compile")
  # batch = cur_states, next_states, actions, rewards, terminal_mask
  batch_size = BATCH_SIZE
  iterations = transitions[0].shape[0] // batch_size
  transitions = jax.tree_map(
    lambda x: x.reshape((iterations, batch_size) + x.shape[1:]), transitions)
  loss = 0.0
  for batch in zip(*transitions):
    # cur_states, next_states, actions, rewards, terminal_mask = row

    def loss_fn(policy_model, target_model, batch, gamma):
      cur_states, next_states, actions, rewards, terminal_mask = batch
      out_cur, out_next = policy_model(cur_states), target_model(next_states)
      best_continuations = jnp.max(out_next, axis=1)
      # zero best continuation for terminal_states using mask
      best_continuations = jnp.where(terminal_mask, 0.0, best_continuations)
      # right-hand-side Bellman equation
      targets = rewards + gamma * best_continuations
      current_values = jax.vmap(lambda x, a: x[a])(out_cur, actions)

      def huber_loss(x, y):
        error = x - y
        loss = jnp.where(
          jnp.abs(error) > 1.0, jnp.abs(error) - 0.5, 0.5 * jnp.square(error))
        return loss

      return jnp.mean(huber_loss(targets, current_values), axis=0)

    grad_fn = jax.value_and_grad(loss_fn)
    l, grad = grad_fn(optimizer.target, target_model, batch, gamma)
    loss += l
    optimizer = optimizer.apply_gradient(grad)
  return optimizer, loss


def train(
  policy_optimizer : flax.optim.base.Optimizer, 
  target_model : nn.base.Model,
  steps_total : int,
  num_agents : int):
  scores = []
  print(f"Using {num_agents} environments")
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
      with jax.profiler.TraceContext("train_step"):
        with jax.profiler.TraceContext("sampling"):
          transitions = transitions = memory.sample(num_agents * BATCH_SIZE)
        policy_optimizer, loss = train_step(policy_optimizer, target_model,
                                            transitions, GAMMA)
        # optimizer.step.block_until_ready()
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


key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
policy_model = create_model(subkey)
policy_optimizer = create_optimizer(policy_model, learning_rate=LR)
target_model = policy_model
del policy_model

if __name__ == "__main__":
  num_agents = 128
  total_frames = 4000000
  policy_optimizer, scores, mem = train(policy_optimizer, target_model, 
                  total_frames, num_agents)