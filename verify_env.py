import sys
import os
import json

# Add current directory to path to import models and server
sys.path.append(os.getcwd())

from server.moduflow_environment import ModuflowEnvironment
from models import ModuflowAction, ActionPayload

def test_reward_system():
    env = ModuflowEnvironment()
    
    print("--- Test 1: Empty Final Answer Penalty ---")
    env.reset()
    # Action: FINAL_ANSWER without any analysis
    action = ModuflowAction(
        action_type="FINAL_ANSWER",
        payload=ActionPayload(prediction=1, confidence=0.9)
    )
    obs = env.step(action)
    print(f"Step 1 Reward: {obs.reward:.4f} (Expected penalty for hollow answer)")
    print(f"Done: {obs.done}")
    
    print("\n--- Test 2: Valid Process Flow ---")
    env.reset()
    # Step 1: READ
    action = ModuflowAction(action_type="READ", payload=ActionPayload())
    obs = env.step(action)
    print(f"Step 1 (READ) Reward: {obs.reward:.4f} (Expected -0.02 + 0.05 = 0.03)")
    
    # Step 2: ANALYZE
    action = ModuflowAction(action_type="ANALYZE", payload=ActionPayload(reasoning="The content clearly demonstrates a pattern of coordinated spamming behavior. It includes predatory links and repetitive commercial solicitations that violate the platform's community standards regarding unsolicited commercial content."))
    obs = env.step(action)
    print(f"Step 2 (ANALYZE) Reward: {obs.reward:.4f} (Expected -0.02 + 0.15 = 0.13)")
    
    # Step 3: SELECT_POLICY
    action = ModuflowAction(action_type="SELECT_POLICY", payload=ActionPayload(policy_id="spam"))
    obs = env.step(action)
    print(f"Step 3 (SELECT_POLICY) Reward: {obs.reward:.4f} (Expected partial progress if correct)")
    
    # Step 4: FINAL_ANSWER
    action = ModuflowAction(action_type="FINAL_ANSWER", payload=ActionPayload(prediction=1, confidence=0.95))
    obs = env.step(action)
    print(f"Step 4 (FINAL_ANSWER) Reward: {obs.reward:.4f} (Final R_total clamped between 0 and 1)")
    print(f"Done: {obs.done}")
    
    print("\n--- Test 3: Timeout Penalty ---")
    env.reset()
    env.max_steps = 3
    for i in range(3):
        action = ModuflowAction(action_type="READ", payload=ActionPayload())
        obs = env.step(action)
        print(f"Step {i+1} Reward: {obs.reward:.4f}")
    
    print(f"Done: {obs.done} (Expected True due to max_steps)")
    # Last reward should have included timeout penalty if not FINAL_ANSWER

if __name__ == "__main__":
    try:
        test_reward_system()
    except Exception as e:
        print(f"Error during verification: {e}")
        import traceback
        traceback.print_exc()
