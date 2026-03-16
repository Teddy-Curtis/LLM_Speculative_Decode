# llm_speculativeDecoding

This repository implements speculative decoding from scratch in Python using PyTorch and Hugging Face Transformers.

The project compares:

- standard autoregressive decoding with a single target model
- speculative decoding with a small draft model and a larger target model

It includes scripts for:

- baseline generation
- manual speculative decoding
- benchmarking and block-size sweeps
- token-by-token trace capture
- animated GIF generation for qualitative demos

## Colab

The easiest way to run the project on GPU is with the Colab notebook:

- https://colab.research.google.com/drive/1qiNuh9T63f10LSPZqQ3qsOnt8bX2Whp6?usp=sharing

## Dependencies

The repository uses the following Python packages:

- `torch>=2.2.0`
- `transformers>=4.46.0`
- `accelerate>=0.34.0`
- `Pillow>=10.0.0`

Install them with:

```bash
pip install -r requirements.txt
```

## Repository Layout

- `scripts/` main implementation and benchmarking scripts
- `notebooks/` Colab smoke-test notebook
- `artifacts/` saved benchmark outputs, traces, and GIFs

## Main Scripts

- `scripts/baseline_decode.py` runs standard cached autoregressive decoding
- `scripts/speculative_decode.py` runs manual speculative decoding
- `scripts/benchmark.py` compares baseline vs speculative decoding
- `scripts/benchmark_sweep.py` sweeps draft block sizes
- `scripts/render_trace_gif.py` renders measured trace JSON files into a GIF
- `scripts/render_illustrative_gif.py` builds an illustrative GIF with synthetic timing from existing trace JSON files

## Quick Start

Run a baseline decode:

```bash
python scripts/baseline_decode.py \
  --model gpt2 \
  --prompt "Speculative decoding speeds up inference by" \
  --max-new-tokens 32
```

Run speculative decoding:

```bash
python scripts/speculative_decode.py \
  --draft-model distilgpt2 \
  --target-model gpt2 \
  --prompt "Speculative decoding speeds up inference by" \
  --max-new-tokens 32 \
  --num-draft-tokens 4
```

Benchmark the two methods:

```bash
python scripts/benchmark.py \
  --draft-model distilgpt2 \
  --target-model gpt2-xl \
  --max-new-tokens 300 \
  --num-draft-tokens 6 \
  --greedy \
  --warmup-runs 1 \
  --benchmark-repeats 3
```

## Notes

- Different model pairs behave very differently. Some pairs produce clear speedups, while others remain slower than the baseline because of low acceptance rates or implementation overhead.
- For clean side-by-side trace comparisons, `--greedy` is often the easiest mode to reason about.
- For higher-quality text demos, newer same-family model pairs can work better than GPT-2-family checkpoints.
