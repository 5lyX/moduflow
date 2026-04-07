---
title: ModuFlow Environment Server
emoji: 🧠
colorFrom: green
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# ModuFlow: Multi-Step Content Moderation RL Environment

## 1. Motivation

Real-world content moderation is not a one-step classification problem. Human moderators:
- read content,
- identify relevant policies,
- reason about context,
- and then make a final decision.

Most ML environments reduce this to a **single classification step**, ignoring reasoning and workflow.

**ModuFlow** is designed to simulate **real moderation pipelines** as a **multi-step RL environment**, where agents must:
- explore,
- reason,
- and act sequentially.

This makes it suitable for:
- training reasoning-capable agents
- evaluating LLM decision-making workflows
- studying process-aware reinforcement learning

---

## 2. Environment Overview

Each episode consists of a **multi-step trajectory**:

1. `READ` → unlock content
2. `SELECT_POLICY` → identify relevant violations
3. `ANALYZE` → generate reasoning
4. `FINAL_ANSWER` → produce classification + confidence
5. `REVISE` → optional correction

The environment maintains a **stateful trajectory**, enabling agents to learn:
- exploration vs efficiency
- reasoning quality
- decision calibration

---

## 3. Observation Space

At each step, the agent receives a structured observation:

```json
{
  "content": "string | [CONTENT_LOCKED]",
  "user_metadata": { "account_age_days": int, ... },
  "context_posts": ["string"],
  "available_policies": ["string"],
  "selected_policies": ["string"],
  "analysis_notes": ["string"],
  "history": [{"action": "..."}],
  "step_id": int,
  "max_steps": int,
  "task_type": "easy | medium | hard",
  "reward": float,
  "done": bool
}
```

### Key Design Choices
- **Partial observability**: content is locked until `READ`
- **Trajectory awareness**: history prevents repeated actions
- **Context grounding**: includes metadata + previous posts

---

## 4. Action Space

Agents interact using structured actions:

### Valid Actions
| Action | Description |
| :--- | :--- |
| `READ` | Unlock content |
| `SELECT_POLICY` | Choose a policy |
| `ANALYZE` | Add reasoning |
| `FINAL_ANSWER` | Submit prediction |
| `REVISE` | Modify previous reasoning |

### Action Schema
```json
{
  "action_type": "ANALYZE",
  "payload": {
    "policy_id": "hate_speech",
    "reasoning": "text",
    "prediction": 1,
    "confidence": 0.9
  }
}
```

---

## 5. Reward Design

ModuFlow uses a dense + terminal hybrid reward system.

### Step-Level Rewards
- **+0.20** → correct policy selection
- **+0.15** → meaningful reasoning
- **+0.05** → first `READ`
- **-0.02** → step penalty
- **-0.2 to -0.3** → invalid / hollow actions
- **-0.2** → redundancy
- **-0.2** → timeout

### Final Reward (Core Scoring)

Final reward is computed as:

$R_{total} = 0.55 \times \text{TaskScore} + 0.20 \times \text{Format} + 0.15 \times \text{Length} + 0.10 \times \text{Efficiency}$

Where:

### TaskScore (55%)
$$
\text{TaskScore} = 0.40 \times \text{Accuracy} + 0.40 \times \text{Policy\_F1} + 0.20 \times \text{Reasoning}
$$

**Components**
- **Accuracy** → correct label (0/1)
- **Policy F1** → overlap with true policies
- **Reasoning Score** → programmatic evaluation:
    - linguistic quality
    - semantic relevance
    - policy grounding
    - context usage

**Length Score**
- optimal: 100–300 words

**Efficiency**
- fewer steps = higher reward

**Additional Penalties**
- Missing reasoning: -0.3
- No `ANALYZE`: -0.2
- No `READ`: -0.1
- **Overconfidence penalty**:
$$
\max(0, \text{confidence} - \text{policy\_score}) \times 0.3
$$

---

## 6. Tasks & Difficulty

### Easy
- Clear violation
- Single policy
- Minimal reasoning required
- **Example**: Explicit hate speech

### Medium
- Multi-policy conflict
- Requires tradeoffs
- **Example**: Misinformation + spam overlap

### Hard
- Ambiguous / contextual
- Requires:
    - implicit bias detection
    - contextual reasoning
    - non-explicit signals
- **Example**: Dog-whistle stereotyping

---

## 7. Setup & Usage

### Run with Docker
```bash
docker build -t moduflow -f Dockerfile .
docker run -p 8000:8000 moduflow
```

### Run Inference
**Using OpenAI**
```bash
export OPENAI_API_KEY=your_key
python inference.py
```

**Using HuggingFace**
```bash
export HF_TOKEN=your_token
python inference.py
```

---

## 8. Baseline Agent

We provide a LLM-based autonomous agent:

**Features**
- step-by-step reasoning
- history-aware decision making
- structured JSON actions

**Model Options**
- `gpt-4o-mini`
- `Qwen2.5-72B-Instruct`

---

## 9. Baseline Results

| Difficulty | Avg Score |
| :--- | :--- |
| Easy | ~0.75–0.90 |
| Medium | ~0.50–0.70 |
| Hard | ~0.30–0.55 |

**Success Threshold**: score >= 0.30
