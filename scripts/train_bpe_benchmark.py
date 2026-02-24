#!/usr/bin/env python3
"""
BPE Training Benchmark Script for TinyStories and OpenWebText datasets.

Usage:
    python scripts/train_bpe_benchmark.py tinystories
    python scripts/train_bpe_benchmark.py owt
    python scripts/train_bpe_benchmark.py --profile tinystories
"""

import argparse
import cProfile
import json
import pstats
import resource
import sys
import time
import tracemalloc
from pathlib import Path

from cs336_basics.train_bpe import FastBPETrainer


def format_time(seconds: float) -> str:
    """Format seconds into human readable time."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        return f"{seconds / 60:.2f} minutes"
    else:
        return f"{seconds / 3600:.2f} hours"


def format_memory(bytes_val: int) -> str:
    """Format bytes into human readable memory."""
    return f"{bytes_val / (1024 * 1024):.2f} MB"


def find_longest_token(vocab_path: str) -> tuple[str, int]:
    """Find the longest token in the vocabulary."""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    longest_token = ""
    max_length = 0

    for token_id, token in vocab.items():
        if isinstance(token, list):
            token_bytes = bytes(token)
            try:
                token_str = token_bytes.decode('utf-8', errors='replace')
            except (UnicodeDecodeError, AttributeError):
                token_str = str(token_bytes)
        else:
            token_str = str(token)

        if len(token_str) > max_length:
            max_length = len(token_str)
            longest_token = token_str

    return longest_token, max_length


def train_tinystories(output_dir: Path, num_workers: int = 4) -> dict:
    """
    Train BPE on TinyStories dataset.
    Vocab size: 10,000 with <|endoftext|> special token.
    """
    print("=" * 60)
    print("Training BPE on TinyStories")
    print("=" * 60)

    data_path = "/Users/nashzhou/code/llm/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    vocab_path = output_dir / "TinyStoriesV2_vocab.json"
    merges_path = output_dir / "TinyStoriesV2_merges.txt"

    start_time = time.perf_counter()
    tracemalloc.start()

    trainer = FastBPETrainer(
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )

    trainer.train(data_path, num_workers)

    trainer.save_vocab_json(str(vocab_path))
    trainer.save_merges_txt(str(merges_path))

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time.perf_counter()

    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    max_rss_children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss

    elapsed = end_time - start_time

    print(f"\n{'-' * 40}")
    print(f"TinyStories Training Complete")
    print(f"{'-' * 40}")
    print(f"Time: {format_time(elapsed)}")
    print(f"Peak Python memory: {format_memory(peak_mem)}")
    print(f"Peak RSS (main): {format_memory(max_rss)}")
    print(f"Peak RSS (children): {format_memory(max_rss_children)}")

    longest_token, token_length = find_longest_token(str(vocab_path))
    print(f"Longest token ({token_length} chars): {repr(longest_token)}")
    print(f"Vocab saved to: {vocab_path}")
    print(f"Merges saved to: {merges_path}")

    return {
        "dataset": "TinyStories",
        "time_seconds": elapsed,
        "time_formatted": format_time(elapsed),
        "peak_memory_mb": peak_mem / (1024 * 1024),
        "peak_rss_main_mb": max_rss / (1024 * 1024),
        "peak_rss_children_mb": max_rss_children / (1024 * 1024),
        "longest_token": longest_token,
        "longest_token_length": token_length,
        "vocab_path": str(vocab_path),
        "merges_path": str(merges_path),
    }


def train_owt(output_dir: Path, num_workers: int = 4) -> dict:
    """
    Train BPE on OpenWebText dataset.
    Vocab size: 32,000.
    """
    print("=" * 60)
    print("Training BPE on OpenWebText")
    print("=" * 60)

    data_path = "/Users/nashzhou/code/llm/cs336/assignment1-basics/data/owt_train.txt"
    vocab_path = output_dir / "owt_vocab.json"
    merges_path = output_dir / "owt_merges.txt"

    start_time = time.perf_counter()
    tracemalloc.start()

    trainer = FastBPETrainer(
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
    )

    trainer.train(data_path, num_workers)

    trainer.save_vocab_json(str(vocab_path))
    trainer.save_merges_txt(str(merges_path))

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time.perf_counter()

    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    max_rss_children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss

    elapsed = end_time - start_time

    print(f"\n{'-' * 40}")
    print(f"OpenWebText Training Complete")
    print(f"{'-' * 40}")
    print(f"Time: {format_time(elapsed)}")
    print(f"Peak Python memory: {format_memory(peak_mem)}")
    print(f"Peak RSS (main): {format_memory(max_rss)}")
    print(f"Peak RSS (children): {format_memory(max_rss_children)}")

    longest_token, token_length = find_longest_token(str(vocab_path))
    print(f"Longest token ({token_length} chars): {repr(longest_token)}")
    print(f"Vocab saved to: {vocab_path}")
    print(f"Merges saved to: {merges_path}")

    return {
        "dataset": "OpenWebText",
        "time_seconds": elapsed,
        "time_formatted": format_time(elapsed),
        "peak_memory_mb": peak_mem / (1024 * 1024),
        "peak_rss_main_mb": max_rss / (1024 * 1024),
        "peak_rss_children_mb": max_rss_children / (1024 * 1024),
        "longest_token": longest_token,
        "longest_token_length": token_length,
        "vocab_path": str(vocab_path),
        "merges_path": str(merges_path),
    }


def profile_training(train_func, *args, **kwargs) -> dict:
    """Profile the training function and print stats."""
    profiler = cProfile.Profile()
    profiler.enable()

    result = train_func(*args, **kwargs)

    profiler.disable()

    print(f"\n{'=' * 60}")
    print("Profiling Results (Top 20 by cumulative time)")
    print(f"{'=' * 60}")
    stats = pstats.Stats(profiler, stream=sys.stdout)
    stats.sort_stats('cumulative')
    stats.print_stats(20)

    print(f"\n{'=' * 60}")
    print("Profiling Results (Top 20 by internal time)")
    print(f"{'=' * 60}")
    stats.sort_stats('time')
    stats.print_stats(20)

    return result


def compare_tokenizers(tinystories_result: dict, owt_result: dict) -> None:
    """Compare and contrast the two trained tokenizers."""
    print(f"\n{'=' * 60}")
    print("Tokenizer Comparison")
    print(f"{'=' * 60}")

    print(f"\nTinyStories:")
    print(f"  - Vocab size: 10,000")
    print(f"  - Training time: {tinystories_result['time_formatted']}")
    print(f"  - Longest token ({tinystories_result['longest_token_length']} chars): "
          f"{repr(tinystories_result['longest_token'][:50])}")

    print(f"\nOpenWebText:")
    print(f"  - Vocab size: 32,000")
    print(f"  - Training time: {owt_result['time_formatted']}")
    print(f"  - Longest token ({owt_result['longest_token_length']} chars): "
          f"{repr(owt_result['longest_token'][:50])}")

    time_ratio = owt_result['time_seconds'] / tinystories_result['time_seconds']
    print(f"\nOpenWebText training took {time_ratio:.2f}x longer than TinyStories")
    print("(considering larger dataset size and vocab size)")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark BPE training on TinyStories and OpenWebText"
    )
    parser.add_argument(
        "dataset",
        choices=["tinystories", "owt", "both"],
        help="Dataset to train on: tinystories, owt, or both"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling to find bottlenecks"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes (default: 4)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Users/nashzhou/code/llm/cs336/assignment1-basics/output",
        help="Output directory for vocab and merges"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    if args.dataset == "tinystories" or args.dataset == "both":
        if args.profile:
            results["tinystories"] = profile_training(
                train_tinystories, output_dir, args.workers
            )
        else:
            results["tinystories"] = train_tinystories(output_dir, args.workers)

    if args.dataset == "owt" or args.dataset == "both":
        if args.profile:
            results["owt"] = profile_training(
                train_owt, output_dir, args.workers
            )
        else:
            results["owt"] = train_owt(output_dir, args.workers)

    if args.dataset == "both" and len(results) == 2:
        compare_tokenizers(results["tinystories"], results["owt"])

    print(f"\n{'=' * 60}")
    print("Benchmark Complete")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
