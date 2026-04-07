import asyncio
import os
import json
import sys
import numpy as np
import time
from typing import Dict, List, Any
from openai import OpenAI

try:
    from moduflow.client import ModuflowEnv
    from moduflow.inference import run_episode, API_KEY, API_BASE_URL, MODEL_NAME, LOCAL_IMAGE_NAME
except ImportError:
    from client import ModuflowEnv
    from inference import run_episode, API_KEY, API_BASE_URL, MODEL_NAME, LOCAL_IMAGE_NAME

# MC Parameters based on user configuration
N_ROLLOUTS = int(os.getenv("N_ROLLOUTS", 2))
TEMPERATURE = 0.8
DIFFICULTIES = ["easy", "medium", "hard"]
EPISODE_DELAY = float(os.getenv("EPISODE_DELAY", 15.0)) # Seconds between episodes

# Added throttle to be inside rate limitation of huggingface
class ThrottledClient:
    def __init__(self, client):
        self._client = client
        self.chat = ThrottledCompletions(client.chat)

class ThrottledCompletions:
    def __init__(self, chat):
        self._chat = chat
        self.completions = ThrottledMessages(chat.completions)

class ThrottledMessages:
    def __init__(self, completions):
        self._completions = completions

    def create(self, *args, **kwargs):
        max_retries = 5
        base_delay = 5
        for attempt in range(max_retries):
            try:
                return self._completions.create(*args, **kwargs)
            except Exception as e:
                err_msg = str(e).lower()
                # 402 is Payment Required, 429 is Too Many Requests
                if "402" in err_msg or "429" in err_msg or "rate limit" in err_msg:
                    delay = base_delay * (2 ** attempt)
                    print(f"\n[THROTTLE] Rate limit/Credit burst (402/429) hit. Retrying in {delay}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(delay)
                else:
                    raise e
        return self._completions.create(*args, **kwargs) # Last try

async def run_benchmark():
    raw_client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL) if API_KEY else None
    client = ThrottledClient(raw_client) if raw_client else None
    
    if client is None:
        print("ERROR: OpenAI client could not be initialized. Check HF_TOKEN or OPENAI_API_KEY.")
        return

    # Initialize environment
    try:
        env = ModuflowEnv(base_url="http://localhost:8000")
        await env.reset()
    except Exception:
        print("Local server not found, starting via Docker...")
        env = await ModuflowEnv.from_docker_image(LOCAL_IMAGE_NAME)

    # Load tasks to know what we are looking for
    tasks_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks")
    task_map = {} # task_id -> difficulty
    all_tasks_data = {} # task_id -> task_dict
    for diff in DIFFICULTIES:
        with open(os.path.join(tasks_dir, f"{diff}.json"), "r", encoding="utf-8") as f:
            tasks_data = json.load(f)
            for t in tasks_data:
                task_map[t["id"]] = diff
                all_tasks_data[t["id"]] = t

    # Results storage
    # { difficulty: { task_id: { "scores": [], "confidences": [], "predictions": [] } } }
    all_results = {diff: {} for diff in DIFFICULTIES}
    for tid, diff in task_map.items():
        all_results[diff][tid] = {"scores": [], "confidences": [], "predictions": []}

    total_tasks = len(task_map)
    print(f"Starting benchmark for {total_tasks} tasks with {N_ROLLOUTS} rollouts each (T={TEMPERATURE}).")

    # Since the server cycles through tasks, we just keep running episodes until all tasks have N samples
    max_episodes = total_tasks * N_ROLLOUTS * 2 # Safety multiplier
    episodes_run = 0
    seen_tasks = set()
    
    while True:
        # Check if we are done
        done_sampling = True
        for diff in DIFFICULTIES:
            for tid in all_results[diff]:
                if len(all_results[diff][tid]["scores"]) < N_ROLLOUTS:
                    done_sampling = False
                    break
        
        if done_sampling or episodes_run >= max_episodes:
            break
            
        print(f"\n--- Episode {episodes_run + 1} ---")
        # run_episode already calls env.reset()
        res = await run_episode(env, client, temperature=TEMPERATURE, verbose=False)
        task_id = res["task_id"]
        
        if task_id in task_map:
            seen_tasks.add(task_id)
            diff = task_map[task_id]
            if len(all_results[diff][task_id]["scores"]) < N_ROLLOUTS:
                all_results[diff][task_id]["scores"].append(res["score"])
                all_results[diff][task_id]["confidences"].append(res["confidence"] or 0.0)
                all_results[diff][task_id]["predictions"].append(res["prediction"] if res["prediction"] is not None else 0)
                print(f"Collected sample for {task_id} ({diff}). Progress: {len(all_results[diff][task_id]['scores'])}/{N_ROLLOUTS}")
            else:
                print(f"Already have enough samples for {task_id}. Skipping.")
        else:
            print(f"Unknown task_id: {task_id}")
            
        episodes_run += 1
        
        if not done_sampling:
            print(f"Waiting {EPISODE_DELAY}s between episodes...")
            await asyncio.sleep(EPISODE_DELAY)

    await env.close()

    # Calculate Aggregated Stats
    print("\n" + "="*40)
    print("MODUFLOW BASELINE RESULTS (MC Aggregated)")
    print("="*40)
    print(f"| Difficulty | Avg Score | Success Rate |")
    print(f"| :--- | :--- | :--- |")
    
    overall_summary = {}

    for diff in DIFFICULTIES:
        diff_scores = []
        diff_successes = []
        
        for tid, data in all_results[diff].items():
            if not data["scores"]:
                continue
                
            # MC Aggregation logic per task
            avg_score = sum(data["scores"]) / len(data["scores"])
            
            # Calculate mean probability of violation across all rollouts
            # If prediction was 0 (allow), its prob-of-violation is (1 - confidence)
            # If prediction was 1 (violation), its prob-of-violation is confidence
            probs = []
            for p, c in zip(data["predictions"], data["confidences"]):
                prob_v = c if p == 1 else (1.0 - c)
                probs.append(prob_v)
            
            avg_prob_violation = sum(probs) / len(probs)
            
            # Final prediction is thresholded mean probability of violation
            final_prediction = 1 if avg_prob_violation >= 0.5 else 0
            
            # Use ground truth for success rate calculation
            true_label = all_tasks_data[tid]["true_label"]
            is_success = (final_prediction == true_label)
            
            diff_scores.append(avg_score)
            diff_successes.append(1 if is_success else 0)
            
        if diff_scores:
            mean_diff_score = sum(diff_scores) / len(diff_scores)
            success_rate = sum(diff_successes) / len(diff_successes)
            print(f"| {diff.capitalize():<10} | {mean_diff_score:.2f} | {success_rate:.0%} |")
            overall_summary[diff] = mean_diff_score
        else:
            print(f"| {diff.capitalize():<10} | N/A | N/A |")

    print("="*40)
    
    print(f"\nBenchmark coverage: {len(seen_tasks)}/{total_tasks} tasks sampled at least once.")
    if len(seen_tasks) < total_tasks:
        missing = set(task_map.keys()) - seen_tasks
        print(f"Missing tasks: {', '.join(list(missing)[:5])}...")
    print("="*40)

if __name__ == "__main__":
    asyncio.run(run_benchmark())
