import numpy as onp

class NumpyMemory:
  '''
  Circular replay buffer built using preallocated NumPy arrays.
  '''
  def __init__(self, capacity : int, prioritized : bool = False):
    self.states = onp.zeros(shape=(capacity, 84, 84, 4), dtype=onp.uint8)
    self.actions = onp.zeros(shape=(capacity, ), dtype=onp.int32)
    self.next_states = onp.zeros(shape=(capacity, 84, 84, 4), dtype=onp.uint8)
    self.rewards = onp.zeros(shape=(capacity, ), dtype=onp.float32)
    self.terminal = onp.zeros(shape=(capacity, ), dtype=bool)
    self.counter = 0 #indicates position for next data to be inserted
    self.capacity = capacity
    self.full = False #set to True after the entire buffer has been filled
    self.prioritized = prioritized
    if self.prioritized:
      #initialize to zeros, small number added in sample() for stability
      self.priorities = onp.ones(shape=(capacity, ), dtype=onp.float32)

  def __len__(self):
    if not self.full:
        return self.counter
    return self.capacity

  def _push_batch(self, states, actions, next_states, rewards):
    # we need to handle the case when the batch spans across the end of array
    # then part of the batch is inserted at the end of the array, part at 
    # the beginning
    batch_size = states.shape[0]
    free_slots = self.capacity - self.counter #free slots at arrray end
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

  def _push_single(self, state, action, next_state, reward, priority=None):
    self.states[self.counter] = onp.squeeze(state, axis=0)
    self.actions[self.counter] = action
    if next_state is not None:
      self.next_states[self.counter] = onp.squeeze(next_state, axis=0)
    else:
      self.terminal[self.counter] = True 
    self.rewards[self.counter] = reward
    if priority is not None:
      self.priorities[self.counter] = priority
    self.counter += 1
    if self.counter == self.capacity: # "roll" the counter
      self.counter = 0
      self.full = True

  def push(self, state, action, next_state, reward, priority=None):
    if state.shape[0] == 1:
      if priority is not None:
        self._push_single(state, action, next_state, reward, priority)
      else:
        max_pr = onp.amax(self.priorities)
        # print("Pushing with maximal priority")
        self._push_single(state, action, next_state, reward, max_pr)

    else:
      assert priority is None, "Batched pushing with priorities not implemented"
      self._push_batch(state, action, next_state, reward)

  def sample(self, batch_size : int, priority_exponent : float = None):
    """
    Sample from the replay buffer. If `priority exponent` is not specified,
    uniform sampling is used. Otherwise, each sample is selected with 
    probability: priorities[i]**a/(\sum_j priorities[j]**a)
    where a, for short, is the priorty exponent
    """
    ub = len(self)
    if priority_exponent is None:
      indices = onp.random.randint(0, ub, size=(batch_size,))
    elif priority_exponent == 0.:
      indices = onp.random.randint(0, ub, size=(batch_size,))
    else:
      l = len(self)
      probs = self.priorities[:l].copy()
      probs += 1e-12 #add small number to avoid division by zero
      probs **= priority_exponent
      probs /= onp.sum(probs)
      indices  = onp.random.choice(a=l, size=batch_size, replace=False, p=probs)
    # print(f"Sample: len of entire buffer {l} {len(self)}, batch_size")
    cur_states_batch = self.states[indices]
    next_states_batch = self.next_states[indices]
    actions_batch = self.actions[indices]
    rewards_batch = self.rewards[indices]
    terminal_mask = self.terminal[indices]
    if priority_exponent is None: 
      return cur_states_batch, next_states_batch, actions_batch, \
            rewards_batch, terminal_mask
    else:
      return cur_states_batch, next_states_batch, actions_batch, \
            rewards_batch, terminal_mask, probs[indices], indices

  def update_priorities(self, indices: onp.ndarray, priorities: onp.ndarray):
    assert self.prioritized == True, "Can't update priorities in uniform buffer"
    # print(f"updating priorities, indices {indices}, priorities {priorities}")
    self.priorities[indices] = priorities
    