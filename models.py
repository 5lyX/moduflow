# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Moduflow Environment.

The moduflow environment is a simple test environment that echoes back messages.
"""

from typing import List, Dict, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class ActionPayload(BaseModel):
    policy_id: Optional[str] = Field(default=None, description="The policy ID selected")
    reasoning: Optional[str] = Field(default=None, description="Reasoning note for ANALYZE")
    prediction: Optional[int] = Field(default=None, description="Final binary prediction (1=violation, 0=safe)")
    confidence: Optional[float] = Field(default=None, description="Confidence score")


class ModuflowAction(Action):
    """Action for the ModuFlow environment."""
    action_type: str = Field(default="REVISE", description="Type of action: READ, SELECT_POLICY, ANALYZE, FINAL_ANSWER, REVISE")
    payload: ActionPayload = Field(default_factory=ActionPayload, description="Payload data for the action")


class ModuflowObservation(Observation):
    """Observation from the Moduflow environment."""
    content: str = Field(default="", description="The content to moderate")
    user_metadata: Dict = Field(default_factory=dict, description="Metadata about the user (e.g. account age, followers)")
    context_posts: List[str] = Field(default_factory=list, description="Previous posts in the thread or user history")
    available_policies: List[str] = Field(default_factory=list, description="Policies available to choose from")
    selected_policies: List[str] = Field(default_factory=list, description="Policies agent has flagged")
    analysis_notes: List[str] = Field(default_factory=list, description="Running notes")
    history: List[Dict] = Field(default_factory=list, description="Action trajectory history")
    step_id: int = Field(default=0, description="Current step in the environment")
    max_steps: int = Field(default=10, description="Maximum steps allowed")
    task_type: str = Field(default="easy", description="The difficulty of the task")
