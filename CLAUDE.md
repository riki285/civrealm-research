# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a research baseline for LLM-driven agents playing the Freeciv strategy game via the [CivRealm](https://github.com/bigai-nlco/civrealm) Gymnasium environment. Agents receive game observations and produce actions of the form `(ctrl_type, actor_id, action_name)`.

## Setup & Running

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Required environment variables** (Azure OpenAI path):
```bash
export OPENAI_API_TYPE="azure"
export OPENAI_API_VERSION="<version>"
export OPENAI_API_BASE="<base>"
export OPENAI_API_KEY="<key>"
export DEPLOYMENT_NAME="<deployment>"
# Optional: Pinecone vector DB (used by GPT workers)
export MY_PINECONE_API_KEY="<key>"
export MY_PINECONE_ENV="<env>"
```

**For local LLM inference (Qwen via vLLM):**
```bash
# Start the vLLM server first:
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8001
# Then run:
python main.py
```

**Run the agent:**
```bash
python main.py
```

To switch agents, edit `main.py` and change which agent class is instantiated (see Agent Types below).

## Architecture

### Game Loop (`main.py`)
1. Create `gymnasium.make('civrealm/FreecivLLM-v0')` environment
2. Initialize an agent
3. `env.reset()` → `observations, info`
4. Loop: `agent.act(observations, info)` → `(ctrl_type, actor_id, action_name)` → `env.step(action)`

### Agent Hierarchy

```
BaseAgent (civrealm library)
  └── LanguageAgent  (agents/language_agent.py — abstract, owns entity lifecycle + action deconfliction)
        ├── BaselineLanguageAgent  — sequential, single LLM decision per entity
        ├── BaseLangAgent          — one thread per unit (parallel decisions)
        ├── MastabaAgent           — advisor LLM sets strategy, per-unit LLMs execute
        ├── AutoGPTAgent           — legacy; single centralized AzureGPT worker
        └── QwenAgent              — single centralized Qwen worker (local inference)
```

**`LanguageAgent`** is the core base. It tracks entity births/deaths across turns, validates actions against `info['available_actions']`, and retries on conflicts up to `max_deconflict_depth`.

### Worker Hierarchy

Workers handle all LLM communication. Each worker manages one entity (unit/city) or serves as an advisor.

```
BaseWorker  (agents/workers/base_worker.py — abstract)
  ├── AzureGPTWorker   — Azure OpenAI + LangChain + optional Pinecone
  │     └── MastabaWorker  — extends GPT worker; can act as "advisor" or "controller"
  └── QwenWorker       — HTTP requests to local vLLM server (localhost:8001)
```

**`BaseWorker`** interface:
- `init_prompts()` / `init_llm()` / `init_index()` — setup
- `generate_command(prompt)` — call LLM
- `process_command(response, input_prompt, avail_actions)` — parse LLM output into action
- `choose_action()` — retry loop with timeout fallback to random action
- Dialogue history is persisted per-turn to `saved_dialogues/`

### Prompt System

Prompts live in `prompt_collections/*/`:
- `*_instruction.txt` — system-level instructions
- `*_task.txt` — per-turn task descriptions
- `advisor_*.txt` — used by MastabaWorker in advisor role
- `city_obs_action.txt` — city-specific context

`config.py` maps entity types (e.g., `"Settlers"`, `"Warriors"`) to prompt collection subdirectories via `PROMPT_SOLUTIONS_DICT`. The `"_final"` key is the fallback for unlisted types.

Prompt loading is handled by `BasePromptHandler` in `agents/prompt_handlers/`.

## Key Design Decisions

- **Worker–Agent separation**: Agents orchestrate turn flow and entity tracking; workers own all LLM I/O and dialogue state.
- **Token limiting**: Workers truncate dialogue history to stay within LLM context limits.
- **Retry + random fallback**: `choose_action()` retries on parse errors; falls back to a random valid action on timeout.
- **Parallelism**: `BaseLangAgent` spawns one thread per living unit for concurrent decisions.
- **MastabaAgent**: Calls an advisor worker first to produce a strategy, then passes that strategy as context to per-unit controller workers.
