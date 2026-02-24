# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is CS336 Spring 2025 Assignment 1: Basics - a from-scratch implementation of core LLM components including BPE tokenization and transformer architecture. The assignment implements components matching PyTorch interfaces without using high-level PyTorch primitives like `nn.Linear` or `nn.MultiheadAttention` directly for core algorithms.

## Development Commands

All commands use `uv` for environment management:

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_model.py

# Run single test
uv run pytest tests/test_model.py::test_linear -v

# Run with snapshot update (if test outputs change intentionally)
uv run pytest tests/test_model.py --snapshot-update

# Run specific test category
uv run pytest tests/test_tokenizer.py -v

# Lint with ruff
uv run ruff check .

# Format code
uv run ruff format .

# Type check
uv run pyright

# Run a script
uv run python scripts/train_bpe_benchmark.py tinystories
```

## Architecture

### Test-Driven Development Flow

Tests drive implementation through `tests/adapters.py`, which contains function stubs that connect your implementations to the test suite:

1. **Implement modules** in `cs336_basics/` (e.g., `model.py`, `tokenizer.py`)
2. **Wire up in adapters** - Import your classes and instantiate them in `tests/adapters.py` functions
3. **Tests compare against snapshots** - Expected outputs are stored in `tests/_snapshots/` as `.npz` files

The adapter pattern allows the test suite to remain agnostic to your specific class APIs while requiring consistent function signatures for validation.

### Key Components

**BPE Tokenization** (`cs336_basics/tokenizer.py`, `cs336_basics/train_bpe.py`):

- `BPE_Tokenizer`: Inference-time tokenizer with vocab/merges lookup
- `FastBPETrainer`: Parallel training using multiprocessing with heap-optimized pair counting
- Uses GPT-2 pretokenization regex pattern (`PAT`)
- Special tokens are handled via regex pattern matching before BPE encoding

**Transformer Model** (`cs336_basics/model.py`):
Custom PyTorch modules implementing:

- `Embedding`: Token embedding lookup
- `Linear`: Matrix multiplication layer
- SwiGLU feedforward (to be implemented)
- Scaled dot-product attention (to be implemented)
- Multi-head self-attention with RoPE (to be implemented)
- RMSNorm (to be implemented)
- Full Transformer LM (to be implemented)

**Weight Loading Convention**: Tests provide weights via state_dict loading. Your modules should define parameters that match the expected weight shapes from `ts_state_dict` fixture.

### Test Infrastructure

**Snapshot Testing** (`tests/conftest.py`):

- `numpy_snapshot` fixture compares array outputs against stored `.npz` files
- `ts_state_dict` fixture provides pretrained weights from `tests/fixtures/ts_tests/model.pt`
- Tests use `numpy.testing.assert_allclose` with configurable tolerances (`rtol`, `atol`)

**Test Categories**:

- `test_model.py`: Transformer components (attention, SwiGLU, RoPE, full model)
- `test_tokenizer.py`: BPE encoding/decoding roundtrips, tiktoken compatibility
- `test_train_bpe.py`: BPE training speed and correctness
- `test_data.py`: Batch sampling from datasets
- `test_nn_utils.py`: Softmax, cross-entropy, gradient clipping
- `test_optimizer.py`: AdamW, cosine learning rate schedule
- `test_serialization.py`: Checkpoint save/load

### Project Structure

```bash
cs336_basics/
├── tokenizer.py          # BPE_Tokenizer class
├── train_bpe.py          # FastBPETrainer class
├── model.py              # Transformer components
└── pretokenization_example.py  # Reference pretokenization

tests/
├── adapters.py           # Wire implementations to tests (main integration point)
├── conftest.py           # Fixtures: numpy_snapshot, ts_state_dict, model params
├── common.py             # Shared utilities (gpt2_bytes_to_unicode)
├── _snapshots/           # Expected test outputs (.npz files)
└── fixtures/             # Test data including pretrained weights
```

### Data Setup

TinyStories and OpenWebText sample data are expected in `data/`:

```bash
mkdir -p data && cd data
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
```

### Dependencies

Key libraries:

- `torch~=2.6.0` (or 2.2.2 on Intel Macs)
- `jaxtyping`: Type annotations for tensor shapes
- `einops`: Tensor manipulation
- `regex`: GPT-2 pretokenization pattern
- `tiktoken`: Reference tokenizer for testing
- `pytest`: Testing with snapshot fixtures
- `ruff`: Linting and formatting
