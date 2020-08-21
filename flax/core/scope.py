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

"""Flax functional core."""

import enum
import functools
import hashlib
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, TypeVar, Union, Generic

from . import tracers
from .frozen_dict import freeze
from .frozen_dict import FrozenDict
from .frozen_dict import unfreeze

import jax
from jax import lax
from jax import random

T = TypeVar('T')


PRNGKey = Any
Array = Any


KindFilter = Union[bool, str, Sequence[str]]

MaybeFrozenKind = Union[Dict[str, Any], FrozenDict[str, Any]]

Variables = Dict[str, MaybeFrozenKind]


def _fold_in_str(rng: PRNGKey, data: str) -> PRNGKey:
  """Fold a string into a jax.random.PRNGKey using its SHA-1 hash."""
  m = hashlib.sha1()
  m.update(data.encode('utf-8'))
  d = m.digest()
  hash_int = int.from_bytes(d[:4], byteorder='big')
  return random.fold_in(rng, hash_int)


def in_kind_filter(kind_filter: KindFilter, kind: str) -> bool:
  if isinstance(kind_filter, str):
    return kind == kind_filter
  if isinstance(kind_filter, Sequence) and not isinstance(kind_filter, str):
    return kind in kind_filter
  if isinstance(kind_filter, bool):
    return kind_filter
  raise TypeError('Invalid KindFilter')


def group_kinds(xs: Variables,
                kind_filters: Sequence[KindFilter]) -> Sequence[Variables]:
  """Group variables by kind filters."""
  kinds = xs.keys()
  groups = []
  for kind_filter in kind_filters:
    remaining_kinds = []
    group = {}
    for kind in kinds:
      if in_kind_filter(kind_filter, kind):
        group[kind] = jax.tree_map(lambda x: x, xs[kind])
      else:
        remaining_kinds.append(kind)
    kinds = remaining_kinds
    groups.append(group)
  return tuple(groups)


class Variable(Generic[T]):

  def __init__(self, scope: 'Scope', kind: str, name: str):
    self.scope = scope
    self.kind = kind
    self.name = name

  @property
  def value(self) -> T:
    return self.scope.get_variable(self.kind, self.name)

  @value.setter
  def value(self, value: T):
    self.scope.put_variable(self.kind, self.name, value)

import contextlib

class Scope:
  """Scope."""

  def __init__(self,
               variables: Variables,
               rngs: Optional[Dict[str, PRNGKey]] = None,
               name: Optional[str] = None,
               parent: Optional['Scope'] = None):
    self._variables = variables
    self.parent = parent
    self.name = name
    self.rngs = rngs if rngs else {}

    self.root = parent.root if parent else self
    self.trace_level = tracers.trace_level(tracers.current_trace())

    self.rng_counters = {key: 0 for key in self.rngs}
    self.reservations = set()

    self._invalid = False

  @property
  def invalid(self):
    return self._invalid

  def _check_valid(self):
    if self._invalid:
      raise ValueError('This scope is no longer valid.')

  @contextlib.contextmanager
  def temporary(self):
    try:
      yield self
    finally:
      self.invalidate()

  def invalidate(self):
    self._invalid = True

  def variables(self):
    self._populate_kinds()
    return freeze(self._variables)

  def _validate_trace_level(self):
    tracers.check_trace_level(self.trace_level)

  def transformed(self, fn, kind):
    variables = unfreeze(self.get_kind(kind))
    variables = freeze(fn(variables))
    scope = self.rewound(rewind_rngs=True)
    scope._variables[kind] = variables
    return scope

  def rewound(self, rewind_rngs=False):
    self._check_valid()
    scope = Scope(self._variables, self.rngs, self.name, self.parent)
    if not rewind_rngs:
      scope.rng_counters = self.rng_counters
    return scope

  def reserve(self, name: str):
    if name in self.reservations:
      raise ValueError(f'Duplicate use of name: "{name}"')
    self.reservations.add(name)

  def default_name(self, prefix: str) -> str:
    i = 0
    while True:
      name = f'{prefix}{i}'
      if name not in self.reservations:
        return name
      i += 1

  def push(self, name: Optional[str] = None, prefix: str = '') -> 'Scope':
    self._check_valid()
    self._validate_trace_level()
    if name is None:
      name = self.default_name(prefix)
    self.reserve(name)
    rngs = {key: _fold_in_str(rng, name) for key, rng in self.rngs.items()}
    scope = Scope({}, name=name, rngs=rngs, parent=self)
    return scope

  def child(self,
            fn: Callable[..., Any],
            name: Optional[str] = None,
            prefix: Optional[str] = None,
            named_call: bool = True,
            **partial_kwargs) -> Callable[..., Any]:
    """Partially applies a child scope to fn."""
    if name is None:
      if prefix is None:
        prefix = fn.__name__ + '_' if hasattr(fn, '__name__') else ''
      name = self.default_name(prefix)
    scope = self.push(name)
    if named_call:
      from . import lift
      fn = lift.named_call(fn, name)
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
      kwargs = dict(partial_kwargs, **kwargs)
      return fn(scope.rewound(), *args, **kwargs)
    return wrapper

  def get_kind(self, kind: str, mutable: bool = False) -> MaybeFrozenKind:
    """Returns all variable of a given kind."""
    if kind not in self._variables:
      if self.parent:
        parent_kind = self.parent.get_kind(kind, mutable)
        if self.name not in parent_kind:
          if isinstance(parent_kind, FrozenDict) or not mutable:
            return FrozenDict()
          parent_kind[self.name] = {}
        self._variables[kind] = parent_kind[self.name]
      elif mutable:
        self._variables[kind] = {}
      else:
        return FrozenDict()
    return self._variables[kind]

  def has_rng(self, kind: str) -> bool:
    return kind in self.rngs

  def make_rng(self, kind: str) -> PRNGKey:
    assert self.has_rng(kind), f"Need RNG for kind {kind}"
    self._check_valid()
    self._validate_trace_level()
    self.rng_counters[kind] += 1
    return random.fold_in(self.rngs[kind], self.rng_counters[kind])

  def get_variable(self, kind: str, name: str, default: T = None) -> T:
    variables = self.get_kind(kind)
    if name in variables:
      return variables[name]
    else:
      return default

  def has_variable(self, kind: str, name: str) -> bool:
    variables = self.get_kind(kind)
    return name in variables

  def put_variable(self, kind: str, name: str, value: Any):
    self._check_valid()
    self._validate_trace_level()
    variables = self.get_kind(kind, mutable=True)
    variables[name] = value

  def variable(self, kind: str, name: str, init_fn: Callable[..., T],
               *init_args) -> Variable[T]:
    self.reserve(name)
    if not self.has_variable(kind, name):
      init_value = init_fn(*init_args)
      self.put_variable(kind, name, init_value)
    return Variable(self, kind, name)

  def param(self, name: str, init_fn: Callable[..., T], *init_args) -> T:
    s_init_fn = lambda *args: init_fn(self.make_rng('param'), *init_args)
    v = self.variable('param', name, s_init_fn, *init_args)
    return v.value

  def _populate_kinds(self):
    kinds = self.root._variables.keys()
    for kind in kinds:
      self.get_kind(kind)

def _unfreeze_variables(variables, mutable):
  new_variables = {}
  for key, value in variables.items():
    if in_kind_filter(mutable, key):
      new_variables[key] = unfreeze(value)
    else:
      new_variables[key] = value
  return new_variables


def apply(fn: Callable[..., Any],
          mutable: KindFilter = False) -> Callable[..., Any]:
  """Functionalize a module."""
  @functools.wraps(fn)
  def wrapper(variables, *args, rngs=None, **kwargs):
    new_variables = _unfreeze_variables(variables, mutable)
    with Scope(new_variables, rngs=rngs).temporary() as root:
      y = fn(root, *args, **kwargs)
    if mutable:
      return y, freeze(new_variables)
    else:
      return y
  return wrapper


def init(fn: Callable[..., Any], mutable: KindFilter = True) -> Callable[..., Any]:
  @functools.wraps(fn)
  def wrapper(rngs, *args, **kwargs):
    if not isinstance(rngs, dict):
      assert rngs.shape == (2,)
      rngs = {'param': rngs}
    return apply(fn, mutable=mutable)({}, *args, rngs=rngs, **kwargs)
  return wrapper
