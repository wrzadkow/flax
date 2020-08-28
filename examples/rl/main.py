import jax.numpy as jnp
import numpy as onp
import jax
import jax.random
import flax
from flax import nn
import time
from typing import Tuple, List
from queue import Queue
import threading

from numpy_memory import NumpyMemory
from model import create_model, create_optimizer
from agent import eps_greedy_action, greedy_action
from remote import RemoteSimulator
from test_episodes import test

def train_step(
  optimizer : flax.optim.base.Optimizer,
  target_model : nn.base.Model,
  transitions : Tuple[onp.ndarray, onp.ndarray, onp.ndarray, 
                      onp.ndarray, onp.ndarray],
  gamma : float):
  """Compilable train step. The loop over num_agents batches is included here
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
  batch_size = BATCH_SIZE
  iterations = transitions[0].shape[0] // batch_size
  transitions = jax.tree_map(
    lambda x: x.reshape((iterations, batch_size) + x.shape[1:]), transitions)
  loss = 0.0
  for batch in zip(*transitions):

    def loss_fn(policy_model, target_model, batch, gamma):
      """Compute loss from the difference between LHS and RHS of Bellman eq."""
      cur_states, next_states, actions, rewards, terminal_mask = batch
      out_cur, out_next = policy_model(cur_states), target_model(next_states)
      best_continuations = jnp.max(out_next, axis=1)
      #best continuation for terminal_states is zero
      best_continuations = jnp.where(terminal_mask, 0.0, best_continuations)
      targets = rewards + gamma * best_continuations #RHS of Bellman eq.
      current_values = jax.vmap(lambda x, a: x[a])(out_cur, actions) #LHS

      def huber_loss(x, y):
        """Squared loss for |x-y| < 1, linear loss elsewhere."""
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

def thread_inference(q1 : Queue, q2: Queue, simulators : List[RemoteSimulator]):
  """Worker function for a separate thread used for inference and running
  the simulators in order to maximize the GPU/TPU usage."""
  while(True):
    # collect states from simulators
    states = []
    for sim in simulators:
      state = sim.conn.recv()
      states.append(state)
    states = onp.concatenate(states, axis=0)

    # perform inference
    policy_optimizer, step = q1.get()
    actions = eps_greedy_action(states, policy_optimizer, step,
                                EPS_START, EPS_END, EPS_DECAY)
    for i, sim in enumerate(simulators):
      action = actions[i]
      sim.conn.send(action)

    # get experience from simulators
    experiences = []
    for sim in simulators:
      sample = sim.conn.recv()
      experiences.append(sample)
    q2.put(experiences)

def train(
  policy_optimizer : flax.optim.base.Optimizer, 
  target_model : nn.base.Model,
  steps_total : int,
  num_agents : int,
  train_device, 
  inference_device):
  """Main training loop.

  Args:
    optimizer: optimizer for the policy model
    target_model: target model
    steps total: total number of frames (env steps) to train on
    num_agents: number of separate processes with agents running the envs
    train_device : device used for training
    inference_device :  device used for inference

  Returns:
    policy_optimizer: optimizer for the policy model, containing the trained
                      parameters
    memory: NumpyMemory object containing MEMORY_SIZE last frames seen
  """
  print(f"Using {num_agents} environments")
  memory = NumpyMemory(MEMORY_SIZE)
  simulators = [RemoteSimulator() for i in range(num_agents)]
  q1, q2 = Queue(maxsize=1), Queue(maxsize=1)
  inference_thread = threading.Thread(target=thread_inference, 
                                      args=(q1, q2, simulators), daemon=True)
  inference_thread.start()
  t1 = time.time()
  for s in range(steps_total // num_agents):
    # print(s)
    if (s + 1) % (10000 // num_agents) == 0:
      print(f"Frames processed {s*num_agents}, time elapsed {time.time()-t1}")
      t1 = time.time()
    if (s + 1) % (50000 // num_agents) == 0:
      test(1, policy_optimizer.target, render=False)
    
    # send the up-to-date policy model and current step to inference thread 
    with jax.profiler.TraceContext("eps_greedy_actions"):
      step = s*num_agents
      q1.put((policy_optimizer, step))

    # perform training: run num_agents train steps
    if len(memory) > INITIAL_MEMORY:
      with jax.profiler.TraceContext("train_step"):
        with jax.profiler.TraceContext("sampling"):
          transitions = transitions = memory.sample(num_agents * BATCH_SIZE)
        policy_optimizer, loss = train_step(policy_optimizer, target_model,
                                            transitions, GAMMA)
        # policy_optimizer.step.block_until_ready()
      if s * num_agents % TARGET_UPDATE <= num_agents:
        # copy policy model parameters to target model
        target_model = policy_optimizer.target
        # copy them also to inference_device
        jax.device_put(policy_optimizer.target, device=inference_device)

    # collect experience from the inference thread and add them to memory
    with jax.profiler.TraceContext("collecting experience"):
      samples = q2.get()
      for s in samples:
        memory.push(*s)

  return policy_optimizer, memory


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
  train_device = jax.devices()[0]
  inference_device = jax.devices()[1]
  train_step = jax.jit(train_step, device=train_device)
  greedy_action = jax.jit(greedy_action, device=inference_device)
  jax.device_put(policy_optimizer.target, device=train_device)
  policy_optimizer, mem = train(policy_optimizer, target_model, 
                  total_frames, num_agents, train_device, inference_device)
