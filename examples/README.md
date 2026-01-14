# SPGG Examples

This directory contains example scripts demonstrating how to use the SPGG framework for multi-agent collaborative reasoning.

## Local Model Examples (Recommended)

These examples run all models locally without requiring API keys:

- **`gsm8k_local_partial.py`**: Main example using Partial Observation mode
  - Agent sequence: Llama → SmolLM2 → Qwen (all local)
  - Each agent sees only the previous agent's solution
  - Requires ~30GB GPU memory

- **`gsm8k_local_full.py`**: Full Observation mode (ablation test)
  - Each agent sees all previous agents' solutions
  - Uses the same checkpoint trained on Partial Observation

## API-Based Examples (Alternative)

If you prefer using API services (Together AI, DashScope):

- **`api/gsm8k_api_partial.py`**: Partial Observation with API models
- **`api/gsm8k_api_full.py`**: Full Observation with API models

### Required API Keys

Set these environment variables before running API examples:

```bash
export TOGETHER_API_KEY=your_together_api_key
export QWEN_API_KEY=your_dashscope_api_key
export HUGGINGFACE_TOKEN=your_hf_token
```

## Notebooks

- **`notebooks/gsm8k_demo.ipynb`**: Interactive demonstration

## Usage

```bash
# Run local model example (Partial Observation)
python gsm8k_local_partial.py

# Run API-based example
python api/gsm8k_api_partial.py
```

## Notes

- Local examples require sufficient GPU memory (30GB+ recommended)
- API examples require valid API keys
- The checkpoint was trained under Partial Observation protocol
- Full Observation scripts serve as ablation/generalization tests
