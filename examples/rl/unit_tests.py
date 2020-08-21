import jax
import flax
from flax import nn
import numpy as onp

from absl.testing import absltest

#test replay buffer
from numpy_memory import NumpyMemory
class TestMemory(absltest.TestCase):
  def test_create(self):
    capacity = 100000
    mem = NumpyMemory(capacity)
    frame_shape = (84, 84, 4)
    memory_states_shape = (capacity, ) + frame_shape
    self.assertTrue(
      mem.states.shape == memory_states_shape)
    self.assertTrue(
      mem.next_states.shape == memory_states_shape)
    self.assertTrue(
      mem.rewards.shape == (capacity, ))
    self.assertTrue(
      mem.rewards.shape == (capacity, ))

  def test_push_and_sample(self):
    capacity = 100000
    mem = NumpyMemory(capacity)
    batch_size = 32
    frame_shape = (84, 84, 4)
    pushed_shape = (1, ) + frame_shape
    dummy_data = (onp.ones(shape=pushed_shape), 2,
                  onp.ones(shape=pushed_shape), 1.0)
    for _ in range(2*batch_size):
      mem.push(*dummy_data)
    states, next_states, actions, rewards, terminal_mask = mem.sample(batch_size)
    self.assertTrue(states.shape == next_states.shape)
    self.assertTrue(states.shape == (batch_size, ) + frame_shape)
    self.assertTrue(actions.shape == rewards.shape == terminal_mask.shape)
    self.assertTrue(actions.shape == (batch_size, ))
    self.assertTrue(states.dtype == next_states.dtype)
    self.assertTrue(states.dtype == onp.uint8)

#test environment and preprocessing
from remote import RemoteSimulator, rcv_action_send_exp
from env import create_env
class TestEnvironmentPreprocessing(absltest.TestCase):
  def test_creation(self):
    frame_shape = (84, 84, 4)
    env = create_env()
    obs = env.reset()
    self.assertTrue(obs.shape == frame_shape)
  
  def test_step(self):
    frame_shape = (84, 84, 4)
    env = create_env()
    obs = env.reset()
    actions = [1, 2, 3, 0]
    for a in actions:
      obs, reward, done, info = env.step(a)
      self.assertTrue(obs.shape == frame_shape)
      self.assertTrue(reward <= 1. and reward >= -1.)
      self.assertTrue(isinstance(done, bool))
      self.assertTrue(isinstance(info, dict))

#test creation of the model and optimizer
from model import create_model, create_optimizer
class TestCreation(absltest.TestCase):
  def test_create(self):
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    policy_model = create_model(subkey)
    policy_optimizer = create_optimizer(policy_model, learning_rate=1e-3)
    self.assertTrue(isinstance(policy_model, nn.base.Model))
    self.assertTrue(isinstance(policy_optimizer, flax.optim.base.Optimizer))
  
if __name__ == '__main__':
  absltest.main()