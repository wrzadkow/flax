import flax
from flax import nn
import jax.numpy as jnp

class Q(flax.nn.Module):
  ''' Same as DQN on @jmichaux Github:
https://github.com/jmichaux/dqn-pytorch/blob/master/models.py
different than DeepMind's one
 '''
  def apply(self, x):
    x = x.astype(jnp.float32) / 255.
    dtype = jnp.float32
    x = nn.Conv(x, features=32, kernel_size=(8, 8), 
                strides=(4, 4), name='conv1',
                dtype=dtype)  
    # x = nn.relu(x)
    x = jnp.maximum(0, x)
    x = nn.Conv(x, features=64, kernel_size=(4, 4), 
                strides=(2, 2), name='conv2',
                dtype=dtype)  
    # x = nn.relu(x)
    x = jnp.maximum(0, x)
    x = nn.Conv(x, features=64, kernel_size=(3, 3), 
                strides=(1, 1), name='conv3',
                dtype=dtype)  
    # x = nn.relu(x)
    x = jnp.maximum(0, x)
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(x, features=512, name='hidden',
                 dtype=dtype)
    # x = nn.relu(x)
    x = jnp.maximum(0, x)
    x = nn.Dense(x, features=4, name='output', dtype=dtype)
    return x

def create_model(key):
  input_dims = (1, 84, 84, 4) #(minibatch, height, width, stacked frames)
  _, initial_params = Q.init_by_shape(key, [(input_dims, jnp.float32)])
  model = flax.nn.Model(Q, initial_params)
  return model

def create_optimizer(model, learning_rate):
  optimizer_def = flax.optim.Adam(learning_rate)
  optimizer = optimizer_def.create(model)
  return optimizer