# Copyright 2022 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Substrate builder."""

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from meltingpot.configs import substrates as substrate_configs
from meltingpot.utils.substrates import substrate
from meltingpot.utils.substrates import substrate_factory
from ml_collections import config_dict

SUBSTRATES = substrate_configs.SUBSTRATES


def get_config(name: str) -> config_dict.ConfigDict:
  """Returns the configs for the specified substrate."""
  return substrate_configs.get_config(name).lock()


def _set_path_value(
    target: Any,
    path: Sequence[str],
    value: Any,
) -> None:
  """Sets a nested value on mapping/list containers using a dotted path."""
  node = target
  for index, segment in enumerate(path):
    is_last = index == len(path) - 1
    try:
      key: str | int = int(segment)
    except ValueError:
      key = segment

    if isinstance(node, (dict, config_dict.ConfigDict)):
      if is_last:
        node[key] = value
        return

      if key not in node or node[key] is None:
        node[key] = [] if path[index + 1].isdigit() else {}
      node = node[key]
      continue

    if isinstance(node, list):
      if not isinstance(key, int):
        raise TypeError(f"List path segment must be an integer: {segment!r}")
      while len(node) <= key:
        node.append(None)

      if is_last:
        node[key] = value
        return

      if node[key] is None:
        node[key] = [] if path[index + 1].isdigit() else {}
      node = node[key]
      continue

    raise TypeError(f"Cannot apply override to non-container type: {type(node)!r}")


def _deep_override(target: Any, overrides: Mapping[str, Any]) -> None:
  """Recursively applies overrides onto nested dict/list structures."""
  for key, value in overrides.items():
    if "." in key:
      _set_path_value(target, key.split("."), value)
      continue

    if (
        isinstance(value, Mapping)
        and isinstance(target, (dict, config_dict.ConfigDict))
        and key in target
        and isinstance(target[key], (dict, config_dict.ConfigDict))
    ):
      _deep_override(target[key], value)
      continue

    target[key] = value


def build(
    name: str,
    *,
    roles: Sequence[str],
    config_overrides: Mapping[str, Any] | None = None,
    lab2d_settings_overrides: Mapping[str, Any] | None = None,
) -> substrate.Substrate:
  """Builds an instance of the specified substrate.

  Args:
    name: name of the substrate.
    roles: sequence of strings defining each player's role. The length of
      this sequence determines the number of players.
    config_overrides: optional overrides for substrate config fields.
    lab2d_settings_overrides: optional overrides for generated lab2d settings.
      Dotted keys such as `simulation.map` and list indices like
      `simulation.gameObjects.0.name` are supported.

  Returns:
    The training substrate.
  """
  return get_factory(
      name,
      config_overrides=config_overrides,
      lab2d_settings_overrides=lab2d_settings_overrides).build(roles)


def build_from_config(
    config: config_dict.ConfigDict,
    *,
    roles: Sequence[str],
    config_overrides: Mapping[str, Any] | None = None,
    lab2d_settings_overrides: Mapping[str, Any] | None = None,
) -> substrate.Substrate:
  """Builds a substrate from the provided config.

  Args:
    config: config resulting from `get_config`.
    roles: sequence of strings defining each player's role. The length of
      this sequence determines the number of players.
    config_overrides: optional overrides for substrate config fields.
    lab2d_settings_overrides: optional overrides for generated lab2d settings.

  Returns:
    The training substrate.
  """
  return get_factory_from_config(
      config,
      config_overrides=config_overrides,
      lab2d_settings_overrides=lab2d_settings_overrides).build(roles)


def get_factory(
    name: str,
    *,
    config_overrides: Mapping[str, Any] | None = None,
    lab2d_settings_overrides: Mapping[str, Any] | None = None,
) -> substrate_factory.SubstrateFactory:
  """Returns the factory for the specified substrate."""
  config = substrate_configs.get_config(name)
  return get_factory_from_config(
      config,
      config_overrides=config_overrides,
      lab2d_settings_overrides=lab2d_settings_overrides)


def get_factory_from_config(
    config: config_dict.ConfigDict,
    *,
    config_overrides: Mapping[str, Any] | None = None,
    lab2d_settings_overrides: Mapping[str, Any] | None = None,
) -> substrate_factory.SubstrateFactory:
  """Returns a factory from the provided config."""
  if config_overrides:
    config = copy.deepcopy(config)
    with config.unlocked():
      _deep_override(config, config_overrides)

  def lab2d_settings_builder(roles):
    lab2d_settings = config.lab2d_settings_builder(roles=roles, config=config)
    if lab2d_settings_overrides:
      lab2d_settings = copy.deepcopy(lab2d_settings)
      _deep_override(lab2d_settings, lab2d_settings_overrides)
    return lab2d_settings

  return substrate_factory.SubstrateFactory(
      lab2d_settings_builder=lab2d_settings_builder,
      individual_observations=config.individual_observation_names,
      global_observations=config.global_observation_names,
      action_table=config.action_set,
      timestep_spec=config.timestep_spec,
      action_spec=config.action_spec,
      valid_roles=config.valid_roles,
      default_player_roles=config.default_player_roles)
