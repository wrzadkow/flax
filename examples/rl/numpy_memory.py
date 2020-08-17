import numpy as onp

class NumpyMemory:
  '''
  Circular replay buffer built using preallocated NumPy arrays.
  '''
  def __init__(self, capacity : int):
    self.states = onp.zeros(shape=(capacity, 84, 84, 4), dtype=onp.uint8)
    self.actions = onp.zeros(shape=(capacity, ), dtype=onp.int32)
    self.next_states = onp.zeros(shape=(capacity, 84, 84, 4), dtype=onp.uint8)
    self.rewards = onp.zeros(shape=(capacity, ), dtype=onp.float32)
    self.terminal = onp.zeros(shape=(capacity, ), dtype=bool)
    self.counter = 0 #indicates position for next data to be inserted
    self.capacity = capacity
    self.full = False #set to True after the entire buffer has been filled

  def __len__(self):
    if not self.full:
        return self.counter
    return self.capacity

  def _push_batch(self, states, actions, next_states, rewards):
    # we need to handle the case when the batch spans across the end of array
    # then part of the batch is inserted at the end of the array, part at 
    # the beginning
    batch_size = states.shape[0]
    free_slots = self.capacity - self.counter #free slots at the end of the array
    put_end = min(free_slots, batch_size) 
    put_beginning = batch_size - put_end
    self.states[self.counter : (self.counter + put_end)] = states[:put_end]
    self.actions[self.counter : (self.counter + put_end)] = actions[:put_end]
    self.next_states[self.counter : 
                     (self.counter + put_end)] = next_states[:put_end]
    self.rewards[self.counter : (self.counter + put_end)] = rewards[:put_end]
    self.counter += put_end
    if put_beginning > 0:
      self.full = True
      self.states[:put_beginning] = states[put_end:]
      self.actions[:put_beginning] = actions[put_end:]
      self.next_states[:put_beginning] = next_states[put_end:]
      self.rewards[:put_beginning] = rewards[put_end:]
      self.counter = put_beginning

  def _push_single(self, state, action, next_state, reward):
    self.states[self.counter] = onp.squeeze(state, axis=0)
    self.actions[self.counter] = action
    if next_state is not None:
      self.next_states[self.counter] = onp.squeeze(next_state, axis=0)
    else:
      self.terminal[self.counter] = True 
    self.rewards[self.counter] = reward 
    self.counter += 1
    if self.counter == self.capacity: # "roll" the counter
      self.counter = 0
      self.full = True

  def push(self, state, action, next_state, reward):
    if state.shape[0] == 1:
      self._push_single(state, action, next_state, reward)
    else:
      self._push_batch(state, action, next_state, reward)

  def sample(self, batch_size : int):
    ub = len(self)
    indices = onp.random.randint(0, ub, size=(batch_size,))
    cur_states_batch = self.states[indices]
    next_states_batch = self.next_states[indices]
    actions_batch = self.actions[indices]
    rewards_batch = self.rewards[indices]
    terminal_mask = self.terminal[indices]
    return cur_states_batch, next_states_batch, actions_batch, rewards_batch, terminal_mask