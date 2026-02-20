import heapq
import mmap
import os
from functools import reduce
import regex as re
import multiprocessing as mp
from typing import BinaryIO, NewType
from collections import Counter, defaultdict

WordId = NewType('WordId', int)
# Regex pattern for pre-tokenization (operates on Unicode)
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

class ReverseCompare:
    """Wrapper that reverses comparison for max-heap behavior."""

    __slots__ = ['pair']

    def __init__(self, pair: tuple[bytes, bytes]):
        self.pair = pair

    def __lt__(self, other):
        return self.pair > other.pair

    def __eq__(self, other):
        return self.pair == other.pair

    def __hash__(self):
        return hash(self.pair)

class FastBPETrainer:
    def __init__(self, vocab_size: int, special_tokens: list[str]):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.vocab: dict[int, bytes] = {}
        self.merges: list[tuple[bytes, bytes]] = []

    def train(self, input_path: str, num_workers: int | None = None) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        num_workers = num_workers or min(mp.cpu_count(), 8)
        self.vocab: dict[int, bytes] = {i: token.encode("utf-8") for i, token in enumerate(self.special_tokens)}
        for i in range(256):
            self.vocab[len(self.special_tokens) + i] = bytes([i])
        word_freqs = self._pretokenize_parallel(input_path, num_workers)
        word_to_id = {word: WordId(i) for i, word in enumerate(word_freqs.keys())}
        id_to_tokens = {wid: list(word) for wid, word in ((word_to_id[w], w) for w in word_freqs)}
        id_to_freq = {word_to_id[w]: f for w, f in word_freqs.items()}

        # pair -> set of word IDs that contain this pair
        pair_locations = defaultdict(set)
        pair_counts = defaultdict(int)

        for wid, tokens in id_to_tokens.items():
            freq = id_to_freq[wid]
            word_pairs = self._compute_pair_counts(tokens)
            for pair, count in word_pairs.items():
                pair_locations[pair].add(wid)
                pair_counts[pair] += freq * count

        # Initialize max-heap for efficient max pair lookup
        # Initialize max-heap for efficient max pair lookup
        # Use negative count for max-heap behavior with heapq (min-heap)
        # Use ReverseCompare(pair) for tie-breaking: larger pair wins
        heap: list[tuple[int, ReverseCompare, tuple[bytes, bytes]]] = []
        for pair, count in pair_counts.items():
            heapq.heappush(heap, (-count, ReverseCompare(pair), pair))
        heapq.heapify(heap)

        target = self.vocab_size - len(self.vocab)
        rebuild_counter = 0
        REBUILD_EVERY = 200
        while len(self.merges) < target and heap:
            best_pair = None
            while heap:
                neg_cnt, _, pair = heapq.heappop(heap)
                current = pair_counts.get(pair, 0)
                if current == -neg_cnt:
                    best_pair = pair
                    break
            
            if best_pair is None:
                break

            pair_first, pair_second = best_pair
            merged = pair_first + pair_second
            self.vocab[len(self.vocab)] = merged
            self.merges.append(best_pair)

            # Update pair locations and counts for affected words
            affected_wids = list(pair_locations[best_pair])

            for wid in affected_wids:
                tokens = id_to_tokens[wid]
                if len(tokens) < 2:
                    continue

                freq = id_to_freq[wid]
                new_tokens = []
                i = 0

                while i < len(tokens):
                    if i < len(tokens) - 1 and tokens[i] == pair_first and tokens[i + 1] == pair_second:
                        new_tokens.append(merged)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                self._update_word_pairs(wid, tokens, new_tokens, freq, 
                                       pair_counts, pair_locations, heap)
                id_to_tokens[wid] = new_tokens
            
            rebuild_counter += 1
            if rebuild_counter >= REBUILD_EVERY:
                # Rebuild the heap to remove stale entries
                heap = [(-cnt, ReverseCompare(pair), pair) for pair, cnt in pair_counts.items() if cnt > 0]
                heapq.heapify(heap)
                rebuild_counter = 0
        
        return self.vocab, self.merges
    
    def _compute_pair_counts(self, tokens: list[bytes]) -> Counter:
        """Compute set of adjacent pairs from token list."""
        pairs = Counter()
        for i in range(len(tokens)-1):
            pairs[(tokens[i], tokens[i+1])] += 1
        return pairs
    
    def _update_word_pairs(self, wid: WordId, old_tokens: list[bytes], new_tokens: list[bytes],
                          freq: int, pair_counts: dict, pair_locations: defaultdict, heap: list) -> None:
        """
        Update global pair statistics when a word's tokenization changes.
        
        Removes contributions of old_pairs, adds contributions of new_pairs,
        and pushes updated counts to heap.
        """
        old_pairs = self._compute_pair_counts(old_tokens)
        new_pairs = self._compute_pair_counts(new_tokens)

        # Remove old pairs
        for p, count_in_word in old_pairs.items():
            pair_counts[p] -= freq * count_in_word
            pair_locations[p].discard(wid)
            if pair_counts[p] <= 0:
                del pair_counts[p]
            else:
                # Push updated (lower) count to heap
                heapq.heappush(heap, (-pair_counts[p], ReverseCompare(p), p))

        # Add new pairs
        for p, count_in_word in new_pairs.items():
            pair_counts[p] += freq * count_in_word
            pair_locations[p].add(wid)
            heapq.heappush(heap, (-pair_counts[p], ReverseCompare(p), p))

    def _pretokenize_parallel(self, input_path: str, num_workers: int) -> dict[tuple[bytes, ...], int]:
        size = os.path.getsize(input_path)
        if size < 10 * 1024 * 1024:  # For small files, do it in the main process to avoid overhead
            with open(input_path, 'rb') as f:
                return self._count_words(f.read())
            
        boundaries = self._find_chunk_boundaries(input_path, num_workers, b"<|endoftext|>")
        with mp.Pool(num_workers) as pool:
            results = pool.starmap(
                self._process_chunk,
                [(input_path, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
            )
        
        total = defaultdict(int)
        for local in results:
            for word, freq in local.items():
                total[word] += freq
        return dict(total)
    
    def _find_chunk_boundaries(self, input_path: str, num_chunks: int, split_special_token: bytes) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        with open(input_path, "rb") as file:
            # Get total file size in bytes
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)

            chunk_size = file_size // num_chunks

            # Initial guesses for chunk boundary locations, uniformly spaced
            # Chunks start on previous index, don't include last index
            chunk_boundaries = [i * chunk_size for i in range(num_chunks + 1)]
            chunk_boundaries[-1] = file_size
            mini_chunk_size = 4096  # Read ahead by 4k bytes at a time
            for bi in range(1, len(chunk_boundaries) - 1):
                initial_position = chunk_boundaries[bi]
                file.seek(initial_position)  # Start at boundary guess
                while True:
                    mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                    # If EOF, this boundary should be at the end of the file
                    if mini_chunk == b"":
                        chunk_boundaries[bi] = file_size
                        break

                    # Find the special token in the mini chunk
                    found_at = mini_chunk.find(split_special_token)
                    if found_at != -1:
                        chunk_boundaries[bi] = initial_position + found_at
                        break
                    initial_position += mini_chunk_size
            # Make sure all boundaries are unique, but might be fewer than num_chunks
            return sorted(set(chunk_boundaries))

    def _process_chunk(self, path: str, start: int, end: int) -> Counter:
        with open(path, "rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                return self._count_words(mm[start:end])
            
    def _count_words(self, data: bytes) -> Counter:
        text = data.decode('utf-8', errors='ignore')
        freqs = Counter()
        if self.special_tokens:
            pattern = re.compile("|".join(re.escape(tok) for tok in self.special_tokens))
            sub_chunks = pattern.split(text)
        else:
            sub_chunks = [text]
        for chunk in sub_chunks:
            for m in PAT.finditer(chunk):
                word = tuple(bytes([b]) for b in m.group().encode('utf-8'))
                freqs[word] += 1
        return freqs

