# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Moduflow Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

import json
import os
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ModuflowAction, ModuflowObservation
except ImportError:
    from models import ModuflowAction, ModuflowObservation

from .graders import AccuracyGrader, FormatGrader, PolicySelectionGrader, EfficiencyGrader, ReasoningGrader, LengthGrader


class ModuflowEnvironment(Environment):
    """
    ModuFlow Environment
    A multi-step content moderation simulator.
    """
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.accuracy_grader = AccuracyGrader()
        self.format_grader = FormatGrader()
        self.policy_grader = PolicySelectionGrader()
        self.efficiency_grader = EfficiencyGrader()
        self.reasoning_grader = ReasoningGrader()
        self.length_grader = LengthGrader() # R_len

        # State Initialization (for safe usage before reset)
        self.history = []
        self.analysis_notes = []
        self.selected_policies = []
        self._content_fetched = False
        self.content = ""
        self.available_policies = []
        self.true_label = 0
        self.true_policies = []
        self.user_metadata = {}
        self.context_posts = []
        self.max_steps = 6
        self.current_task = {}
        
        self.tasks = []
        self._load_tasks()
        
        import random
        random.shuffle(self.tasks)
        self._task_idx = 0
        
    def _load_tasks(self):
        tasks_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tasks")
        for filename in ["easy.json", "medium.json", "hard.json"]:
            filepath = os.path.join(tasks_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.tasks.extend(data)
                    else:
                        self.tasks.append(data)

    def reset(self) -> ModuflowObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        if self.tasks:
            self.current_task = self.tasks[self._task_idx % len(self.tasks)]
            self._task_idx += 1
        else:
            self.current_task = {
                "id": "fallback_1", "type": "easy",
                "content": "Buy cheap meds now!", "available_policies": ["spam"],
                "true_label": 1, "true_policies": ["spam"]
            }
        
        self.content = self.current_task["content"]
        self.available_policies = self.current_task["available_policies"]
        self.true_label = self.current_task.get("true_label", 0)
        self.true_policies = self.current_task.get("true_policies", [])
        self.user_metadata = self.current_task.get("metadata", {})
        self.context_posts = self.current_task.get("context_posts", [])
        
        self.selected_policies = []
        self.analysis_notes = []
        self.history = []
        self._content_fetched = False
        self.max_steps = 6
        
        return self._build_obs(0.0, False)

    def _build_obs(self, reward: float, done: bool) -> ModuflowObservation:
        display_content = self.content if self._content_fetched else "[CONTENT_LOCKED] Call READ action to fetch content."
        display_metadata = self.user_metadata if self._content_fetched else {}
        display_context = self.context_posts if self._content_fetched else []
        
        return ModuflowObservation(
            content=display_content,
            user_metadata=display_metadata,
            context_posts=display_context,
            available_policies=self.available_policies,
            selected_policies=self.selected_policies,
            analysis_notes=self.analysis_notes,
            history=self.history,
            step_id=self._state.step_count,
            max_steps=self.max_steps,
            task_type=self.current_task.get("id", "unknown"),
            done=done,
            reward=reward,
        )

    def step(self, action: ModuflowAction) -> ModuflowObservation:
        action_type = action.action_type
        payload = action.payload

        self._state.step_count += 1
        
        format_score = self.format_grader.score(action)
        
        # Consistent Step Penalty to encourage efficiency (-0.02 per action)
        step_reward = -0.02
        
        if format_score == 0.0:
            step_reward -= 0.1 # Additional penalty for invalid format
            self.history.append({"action": action_type, "note": "invalid format"})
        else:
            is_redundant = False
            
            # Action Space logic: Allow any action anytime
            if action_type == "READ":
                if any(h["action"] == "READ" for h in self.history):
                    is_redundant = True
                else:
                    self._content_fetched = True
                    # READ is informational, small positive if non-redundant
                    step_reward += 0.05
                    
            elif action_type == "SELECT_POLICY":
                policy_id = payload.policy_id
                if not policy_id:
                    # Penalty for SELECT_POLICY without a specific policy ID (-0.2)
                    step_reward -= 0.2
                elif policy_id in self.selected_policies:
                    is_redundant = True
                else:
                    self.selected_policies.append(policy_id)
                    # Reward partial progress for correct identification (+0.2)
                    if policy_id in self.true_policies:
                        step_reward += 0.2
                    else:
                        step_reward -= 0.1 # Minor penalty for wrong flag
            
            elif action_type == "ANALYZE":
                reasoning = payload.reasoning
                if not reasoning or len(reasoning.strip()) < 5:
                    # Penalty for ANALYZE with empty or trivial reasoning (-0.2)
                    step_reward -= 0.2
                elif reasoning in self.analysis_notes:
                    is_redundant = True
                else:
                    self.analysis_notes.append(reasoning)
                    # Reward partial progress for non-redundant reasoning (+0.15)
                    step_reward += 0.15
            
            elif action_type == "FINAL_ANSWER":
                if not self.analysis_notes:
                    # Penalty for attempting a FINAL_ANSWER without any prior analysis (-0.3)
                    step_reward -= 0.3
            
            elif action_type == "REVISE":
                # REVISE is neutral, incurs standard step penalty unless it improves prediction (handled in R_total)
                pass

            if is_redundant:
                step_reward -= 0.2 # Redundancy penalty increased to -0.2
            
            # Record trajectory
            self.history.append({"action": action_type, "payload": payload.model_dump()})

        # Terminal conditions
        is_timeout = (self._state.step_count >= self.max_steps)
        done = (action_type == "FINAL_ANSWER" or is_timeout)
        
        if done:
            # If terminated via timeout without a FINAL_ANSWER, apply timeout penalty (-0.2)
            if is_timeout and action_type != "FINAL_ANSWER":
                step_reward -= 0.2
                
            prediction = payload.prediction if format_score > 0 else None
            
            # terminal reward components
            accuracy = self.accuracy_grader.score(prediction, self.true_label)
            
            # Combine content and context for structural grading
            full_content = self.content
            if self.context_posts:
                full_content = "\n".join(self.context_posts + [self.content])
                
            reference_reasoning = self.current_task.get("reference_reasoning", full_content)
                
            reasoning_score = self.reasoning_grader.score(
                reference_reasoning, 
                self.analysis_notes,
                true_policies=self.true_policies,
                context_posts=self.context_posts,
                prediction=prediction
            )
            length_score = self.length_grader.score(self.analysis_notes)
            efficiency = self.efficiency_grader.score(self._state.step_count, self.max_steps)
            
            # Policy grading with F1 support
            policy_stats = self.policy_grader.score(self.selected_policies, self.true_policies)
            policy_score = policy_stats["f1_score"] if isinstance(policy_stats, dict) else policy_stats
            missed_count = policy_stats.get("missed_count", 0) if isinstance(policy_stats, dict) else 0
            
            # Unified weighting for all task types
            w_acc, w_pol, w_reas = 0.40, 0.40, 0.20
            w_format, w_len, w_eff = 0.20, 0.15, 0.10
            
            # Task correctness score based on F1 and semantic reasoning
            task_reward = (w_acc * accuracy) + (w_pol * policy_score) + (w_reas * reasoning_score)
            
            # Apply a uniform missing policy penalty to the base task reward
            task_reward -= (missed_count * 0.1)
            
            # Penalize overconfidence if confidence far exceeds F1 policy score
            confidence = payload.confidence if payload.confidence is not None else 0.0
            overconfidence_penalty = max(0.0, confidence - policy_score) * 0.3
            task_reward -= overconfidence_penalty
            
            # R_total: Optimized weighting integrating form and function
            # 55% task completion + 45% process discipline
            R_total = (0.55 * task_reward) + (w_format * format_score) + (w_len * length_score) + (w_eff * efficiency)
            
            # Extra Penalties for Process Discipline
            if not self.analysis_notes:
                R_total -= 0.3 # no reasoning penalty
                
            has_analyze = any(h["action"] == "ANALYZE" for h in self.history)
            if not has_analyze:
                R_total -= 0.2 # missing ANALYZE penalty
                
            has_read = any(h["action"] == "READ" for h in self.history)
            if not has_read:
                R_total -= 0.1 # missing READ penalty
            
            # If the format is entirely invalid, force standard negative behavior or zero heavily
            if format_score == 0.0:
                R_total = -0.1 # Heavily punish non-JSON responses terminally
            
            # Final Clamping [0.0, 1.0]
            step_reward = max(0.0, min(1.0, R_total))
            
        return self._build_obs(step_reward, done)

    @property
    def state(self) -> State:
        return self._state
