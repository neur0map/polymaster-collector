# Prowl Bot — ML & Model Architecture Plan

Model selection, training strategy, and architecture decisions for the polymaster Prowl Bot prediction market system.

## System Overview

```
User (Telegram)
    │
    ▼
Orchestrator LLM (Qwen3.5-35B-A3B, no fine-tune)
    │
    ├──► Researcher Agent ──► web search, news gathering
    ├──► Scanner Agent ──► reads whale alerts from wwatcher
    ├──► Analyst Agent ──► market context, order book analysis
    │
    ▼
All context assembled
    │
    ├──► Tabular Predictor (XGBoost) ──► "whale accuracy score"
    ├──► Reasoning Predictor (Qwen3.5-27B LoRA) ──► "72% YES" + reasoning
    │
    ▼
Orchestrator formats response + trade recommendation
    │
    ▼
User approves/rejects → execute on Kalshi / Polymarket
```

## Model Roles

### 1. Orchestrator — Qwen3.5-35B-A3B

- **Job:** Conversation, agent routing, tool-use, memory management, trade approval flow
- **Why this model:** Best-in-class agent benchmarks (TAU2-Bench 81.2), native tool-use, thinking mode, 262K context, Apache 2.0
- **Why MoE here:** Only 3B active params per token = fast inference for conversational use. 35B total knowledge base. Multimodal (can read charts/screenshots if needed)
- **Fine-tuning:** None. Use off-the-shelf via Ollama or MLX
- **Hardware:** Runs on 8GB+ VRAM / 21GB Mac unified memory at 4-bit quantization
- **Source:** `ollama run qwen3.5:35b-a3b` or GGUF from `unsloth/Qwen3.5-35B-A3B-GGUF`

### 2. Tabular Predictor — XGBoost

- **Job:** Fast probability estimation from structured whale alert features
- **Why not an LLM:** Tree-based models beat LLMs on tabular data. Faster, smaller, more interpretable, no GPU needed
- **Input features:**
  - value, price, size (trade fundamentals)
  - whale_profile.win_rate, leaderboard_rank
  - wallet_activity.is_repeat_actor, is_heavy_actor, total_value_day
  - market category, platform
  - order_book.bid_depth, spread
  - price_at_entry vs market consensus
  - days_to_resolution
- **Output:** probability (0.0–1.0) that whale's position is correct
- **Training data:** wwatcher alerts + resolved outcomes from collector
- **Training time:** Seconds on CPU
- **When to build:** Month 1-2 (Phase 5). Becomes the day-one baseline predictor
- **Library:** scikit-learn or xgboost Python package
- **Validation:** Brier score on held-out 20% test set. Must beat random (0.250) and ideally beat market consensus (~0.197)

### 3. Reasoning Predictor — Qwen3.5-27B (LoRA fine-tune)

- **Job:** Read market question + whale signals + news context → output calibrated probability with reasoning chain
- **Why dense 27B, not MoE 35B-A3B:** MoE models are hard to LoRA fine-tune (256 experts, only 8 active per token = most adapters never see gradients). Dense model = every training sample updates all weights = efficient LoRA
- **Why not 8B/14B:** The 27B is the strongest dense model in the Qwen3.5 lineup (SWE-bench 72.4). If hardware is too tight, fall back to DeepSeek-R1-Distill-Qwen-14B
- **Fine-tuning approach:**
  - Stage 1 — SFT (Supervised Fine-Tuning): Train on resolved market questions with reasoning traces bootstrapped from a cloud model
  - Stage 2 — GRPO (Group-Relative Policy Optimization): RL with Brier score as reward on resolved outcomes. Modified GRPO per Turtel et al. 2025 (remove per-question variance scaling)
- **Training data:** 5k-10k real resolved markets + up to 100k synthetic augmented questions (Turtel used 10k real + 100k synthetic)
- **Training hardware:** Mac with MLX (unified memory) or rented A100 for a few hours. LoRA at 4-bit base = ~24GB VRAM
- **Inference:** Quantized (Q4) via Ollama on server. ~16-20GB VRAM
- **When to build:** Month 3-4 (Phase 6). Only after enough resolved data exists from the collector
- **Source:** `Qwen/Qwen3.5-27B` base weights from HuggingFace

### 4. Sentiment Model — FinBERT or similar (Phase 5 ONNX)

- **Job:** Classify news headlines as positive/negative/neutral toward a market outcome
- **Why separate:** Tiny model (110M params), runs as ONNX on CPU. Feeds sentiment scores as features into XGBoost and as context for the reasoning predictor
- **Training:** Use off-the-shelf FinBERT. Optional fine-tune on prediction market headlines if accuracy is low
- **When to build:** Month 2-3, after news collection is active in the collector

## Training Pipeline

### Data sources (all from polymaster-collector + wwatcher)

1. **Resolved markets** — question + outcome (YES/NO) from collector
2. **Price history** — time-series snapshots from collector
3. **Whale alerts** — trade signals from wwatcher, linked by market_id
4. **News headlines** — causally masked context from collector (Phase 2)

### XGBoost training flow

```
wwatcher.db + collector.db
    │
    ▼
collector export (Parquet)
    │
    ▼
Feature engineering (pandas)
    │
    ▼
XGBoost.fit() → model.json
    │
    ▼
Validate: Brier score, calibration curve, paper trading sim
```

### LLM training flow (Turtel et al. approach)

```
collector export --format=prompts
    │
    ▼
Stage 1: SFT on resolved market Q&A pairs
    │ (bootstrap reasoning traces with cloud model)
    ▼
Stage 2: GRPO with Brier score reward
    │ (Modified GRPO: no per-question variance scaling)
    │ (Guardrails: penalize gibberish, non-English, missing rationales)
    │ (Guardrails: penalize extreme 0/1 probabilities)
    ▼
Validate: Brier score, calibration curve, paper trading sim
    │
    ▼
Compare vs XGBoost baseline
    │ If LLM wins → deploy as primary predictor
    │ If XGBoost wins → keep XGBoost, retrain LLM with more data
```

## Validation Gates (must pass before real money)

1. **Brier score < 0.250** — better than random guessing
2. **Brier score ≤ 0.197** — matches Polymarket crowd consensus (Turtel's target)
3. **Calibration curve** — when model says 70%, outcome is YES ~70% of the time
4. **Paper trading ROI > 0%** — simulated bets on held-out resolved markets are profitable
5. **Model disagrees with market** — predictions must not just parrot market prices (correlation < 0.95 with market consensus)
6. **Edge detection** — model identifies spots where market is wrong and is right when it disagrees

## Hardware Requirements

### Mac (development + training)

- Qwen3.5-27B LoRA fine-tune via MLX or Unsloth: 24-32GB unified memory
- XGBoost training: CPU only, any Mac
- Model testing/iteration: same machine

### Prowl Server (24/7 inference + collection)

- Orchestrator (Qwen3.5-35B-A3B Q4): ~8-12GB VRAM
- Reasoning predictor (Qwen3.5-27B Q4 LoRA): ~16-20GB VRAM
- If no GPU: both models can run CPU-only with GGUF quantization, slower but functional
- XGBoost inference: CPU, negligible
- FinBERT ONNX: CPU, ~200MB RAM
- polymaster-collector daemon: ~100MB RAM
- wwatcher daemon: ~50MB RAM
- Total VRAM if GPU available: ~20-24GB (one 3090/4090 fits both models sequentially, or two smaller GPUs)
- Total RAM if CPU-only: 32-64GB recommended

## Build Timeline

### Month 1-2 (Foundation)

- polymaster-collector running 24/7, backfill historical markets
- Prowl Bot Telegram scaffold with Qwen3.5-35B-A3B orchestrator
- XGBoost baseline on existing wwatcher data + collector resolutions
- Bot collects predictions from day one

### Month 3-4 (First ML)

- Enough resolved data for LLM training (~5k-10k labeled markets)
- First SFT fine-tune on Qwen3.5-27B
- First GRPO training run
- Compare LLM vs XGBoost → pick winner
- Paper trading validation

### Month 5+ (Continual Learning)

- GRPO retraining loop as new markets resolve
- Model improves over time with more data
- Expand to synthetic data augmentation if needed
- Graduate from paper trading to real execution (with user approval gate)

## Key References

- Turtel et al. 2025 — "Outcome-based Reinforcement Learning to Predict the Future" (arxiv:2505.17989)
  - 14B model matched o1 accuracy on Polymarket
  - Modified GRPO + ReMax for forecasting
  - 10k real + 100k synthetic training examples
  - >10% ROI in Polymarket trading simulation
  - Brier score as reward signal
- Qwen3.5 Medium Series (Feb 24, 2026)
  - 35B-A3B: MoE, 3B active, best for inference/agents
  - 27B: Dense, best for fine-tuning, SWE-bench 72.4
