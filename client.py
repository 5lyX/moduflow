# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Moduflow Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import ModuflowAction, ModuflowObservation
except ImportError:
    from models import ModuflowAction, ModuflowObservation


class ModuflowEnv(
    EnvClient[ModuflowAction, ModuflowObservation, State]
):
    """
    Client for the Moduflow Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with ModuflowEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(ModuflowAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = ModuflowEnv.from_docker_image("moduflow-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(ModuflowAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: ModuflowAction) -> Dict:
        """
        Convert ModuflowAction to JSON payload for step message.

        Args:
            action: ModuflowAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[ModuflowObservation]:
        """
        Parse server response into StepResult[ModuflowObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with ModuflowObservation
        """
        obs_data = payload.get("observation", {})
        observation = ModuflowObservation(**obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
