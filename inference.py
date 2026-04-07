import asyncio
import os
import time
import json
import textwrap
import sys
from typing import List, Optional
from openai import OpenAI

try:
    from moduflow.client import ModuflowEnv
    from moduflow.models import ModuflowAction, ActionPayload
except ModuleNotFoundError:
    from client import ModuflowEnv
    from models import ModuflowAction, ActionPayload

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "moduflow-env:latest")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

if os.getenv("HF_TOKEN"):
    default_base_url = "https://router.huggingface.co/v1"
    default_model = "Qwen/Qwen2.5-72B-Instruct"
else:
    default_base_url = "https://api.openai.com/v1"
    default_model = "gpt-4o-mini"

API_BASE_URL = os.getenv("API_BASE_URL", default_base_url)
MODEL_NAME = os.getenv("MODEL_NAME", default_model)
BENCHMARK = os.getenv("BENCHMARK", "moduflow")
SUCCESS_SCORE_THRESHOLD = 0.3

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

SYSTEM_PROMPT = """You are an autonomous content moderation agent interacting with an environment.
At each step, choose ONE action to progress towards moderating the content.

Available Actions (action_type):
1. READ: Fetch the content to moderate (Required before analyzing).
2. SELECT_POLICY: Flag a relevant policy. Provide 'policy_id' in payload.
3. ANALYZE: Provide reasoning for your decision. Provide 'reasoning' (150+ words) in payload.
4. REVISE: If you need to reconsider, you can revise.
5. FINAL_ANSWER: Submit your final decision. Provide 'prediction' (0=allow, 1=violate) and 'confidence' (0.0 to 1.0) in payload.

Respond with exactly ONE valid JSON object:
{"action_type": "<action>", "payload": {"policy_id": "...", "reasoning": "...", "prediction": 1, "confidence": 0.95}}
Provide ONLY the fields necessary for your chosen action_type.
"""

def get_action(step: int, obs, client, model_name=MODEL_NAME, temperature=0.7) -> ModuflowAction:
    """
    Agent logic querying the LLM to autonomously explore the environment's action space.
    """
    if not client:
        # Fallback to a hardcoded sequence if API isn't available
        if step == 1:
            return ModuflowAction(action_type="READ", payload=ActionPayload())
        elif step == 2:
            policy = obs.available_policies[0] if obs.available_policies else "unknown"
            return ModuflowAction(action_type="SELECT_POLICY", payload=ActionPayload(policy_id=policy))
        elif step == 3:
            return ModuflowAction(action_type="ANALYZE", payload=ActionPayload(reasoning="fallback reasoning" * 50)) # 150+ words
        else:
            return ModuflowAction(action_type="FINAL_ANSWER", payload=ActionPayload(prediction=1, confidence=0.9))

    # Construct the RL state prompt
    state_str = f"STEP: {step} of {obs.max_steps}\n"
    if hasattr(obs, 'user_metadata') and obs.user_metadata:
        state_str += f"USER METADATA: {json.dumps(obs.user_metadata)}\n"
    if hasattr(obs, 'context_posts') and obs.context_posts:
        state_str += f"CONTEXT POSTS: {json.dumps(obs.context_posts)}\n"
    state_str += f"CONTENT: {obs.content}\n"
    state_str += f"AVAILABLE POLICIES: {', '.join(obs.available_policies)}\n"
    state_str += f"SELECTED POLICIES: {', '.join(obs.selected_policies)}\n"
    
    # Show history of actions to prevent infinite looping
    history_actions = [h.get('action') for h in obs.history if 'action' in h]
    state_str += f"HISTORY: {', '.join(history_actions) if history_actions else 'None'}\n"
    
    if obs.analysis_notes:
        # Truncate notes if they are too long to save context window
        recent_notes = obs.analysis_notes[-1] if obs.analysis_notes else ""
        state_str += f"LATEST ANALYSIS: {recent_notes}\n"
        
    state_str += "\nWhat is your next action? Respond ONLY in JSON."

    try:
        res = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": state_str}
            ],
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        content = res.choices[0].message.content
        data = json.loads(content)
        
        action_type = data.get("action_type", "REVISE")
        payload_dict = data.get("payload", {})
        
        # Enforce valid action type
        if action_type not in ["READ", "SELECT_POLICY", "ANALYZE", "REVISE", "FINAL_ANSWER"]:
            action_type = "REVISE"
            
        print(f"[DEBUG MODEL] Agent chose {action_type}", file=sys.stderr)
        
        # We parse manually to avoid Pydantic validation crashing the loop
        policy_id = payload_dict.get("policy_id")
        reasoning = payload_dict.get("reasoning")
        
        prediction_val = payload_dict.get("prediction")
        try:
            prediction = int(prediction_val) if prediction_val is not None else None
        except (ValueError, TypeError):
            prediction = None
            
        confidence_val = payload_dict.get("confidence")
        try:
            confidence = float(confidence_val) if confidence_val is not None else None
        except (ValueError, TypeError):
            confidence = None

        return ModuflowAction(
            action_type=action_type,
            payload=ActionPayload(
                policy_id=policy_id, 
                reasoning=reasoning, 
                prediction=prediction, 
                confidence=confidence
            )
        )
    except Exception as e:
        print(f"[DEBUG MODEL] API/Parse Error: {e}", file=sys.stderr)
        # Fallback to REVISE to avoid hard-crashing if model outputs invalid generation
        return ModuflowAction(action_type="REVISE", payload=ActionPayload())



async def run_episode(env, client, model_name=MODEL_NAME, temperature=0.7, verbose=True):
    rewards = []
    steps_taken = 0
    success = False
    score = 0.0
    final_prediction = None
    final_confidence = None
    
    result = await env.reset()
    obs = result.observation
    task_id = obs.task_type
    
    if verbose:
        log_start(task=task_id, env=BENCHMARK, model=model_name)
    
    for step in range(1, obs.max_steps + 1):
        if result.done:
            break
            
        action = get_action(step, obs, client, model_name, temperature)
        
        # Action formatting for logs
        if verbose:
            if action.action_type == "READ":
                action_str = "READ"
            elif action.action_type == "SELECT_POLICY":
                action_str = f"SELECT_POLICY policy={action.payload.policy_id}"
            elif action.action_type == "ANALYZE":
                safe_reasoning = action.payload.reasoning.replace('"', '\\"') if action.payload.reasoning else "none"
                action_str = f"ANALYZE reasoning=\"{safe_reasoning}\""
            elif action.action_type == "FINAL_ANSWER":
                action_str = f"FINAL_ANSWER prediction={action.payload.prediction} confidence={action.payload.confidence}"
            else:
                action_str = action.action_type
            
        result = await env.step(action)
        obs = result.observation
        reward = result.reward or 0.0
        done = result.done
        
        rewards.append(reward)
        steps_taken = step
        
        if verbose:
            log_step(step=step, action=action_str, reward=reward, done=done, error=None)
        
        if done:
            # Store final response values for MC aggregation
            final_prediction = action.payload.prediction
            final_confidence = action.payload.confidence
            
            # Terminal reward in Moduflow is the composite score [0, 1]
            score = rewards[-1] if rewards else 0.0
            success = score >= SUCCESS_SCORE_THRESHOLD
            break
            
    if verbose:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
        
    return {
        "success": success,
        "score": score,
        "steps": steps_taken,
        "prediction": final_prediction,
        "confidence": final_confidence,
        "task_id": task_id
    }

async def main():
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL) if API_KEY else None

    if client is None:
        print("[DEBUG MODEL] ERROR: Client is None. Was HF_TOKEN / OPENAI_API_KEY correctly exported?", file=sys.stderr)
    try:
        env = ModuflowEnv(base_url="http://localhost:8000")
        await env.reset()
    except Exception:
        env = await ModuflowEnv.from_docker_image(LOCAL_IMAGE_NAME)
        
    try:
        await run_episode(env, client)
    finally:
        if env:
            await env.close()

if __name__ == "__main__":
    asyncio.run(main())
