from collections.abc import Iterable
import regex as re

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

class BPE_Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else []
        self.special_pattern = "|".join(re.escape(s) for s in self.special_tokens)
        # Create a reverse lookup for vocab: bytes -> id
        self._bytes_to_id = {v: k for k, v in vocab.items()}
        # Create a lookup for merges: pair -> merged_bytes
        self._merge_lookup = {}
        # Create rank lookup: pair -> rank (index in merges list)
        # Lower rank = higher priority (earlier in list)
        self._rank_lookup = {}
        for rank, merge in enumerate(merges):
            first, second = merge
            self._merge_lookup[merge] = first + second
            self._rank_lookup[merge] = rank

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        raise NotImplementedError("Implement loading vocab and merges from files")

    def _get_rank(self, pair: tuple[bytes, bytes]) -> int:
        """Get the rank of a pair in the merges list. Lower rank = higher priority."""
        return self._rank_lookup.get(pair, float('inf'))  # If pair not found, return infinity

    def _bpe_encode(self, text_bytes: bytes) -> list[int]:
        """Apply BPE encoding to a sequence of bytes."""
        if not text_bytes:
            return []

        # Start with individual bytes as tokens
        tokens: list[bytes] = [bytes([b]) for b in text_bytes]

        while len(tokens) >= 2:
            # Find the pair with the lowest rank (highest priority)
            best_rank = float('inf')
            best_idx = -1

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self._get_rank(pair)
                if rank < best_rank:
                    best_rank = rank
                    best_idx = i

            # If no mergeable pair found, stop
            if best_idx == -1 or best_rank == float('inf'):
                break

            # Merge the best pair
            merged = tokens[best_idx] + tokens[best_idx + 1]
            tokens.pop(best_idx + 1)
            tokens.pop(best_idx)
            tokens.insert(best_idx, merged)

        # Convert tokens to IDs, skipping any not in vocab
        return [self._bytes_to_id[t] for t in tokens if t in self._bytes_to_id]

    def _split_text_with_special_tokens(self, text: str) -> list[tuple[str, bool]]:
        """
        Split text into segments, marking which are special tokens.
        Returns list of (segment_text, is_special_token).
        """
        # If no special tokens, return the whole text as a single non-special segment
        if not self.special_tokens:
            return [(text, False)]

        # Split while keeping delimiters (using capturing group to keep delimiters)
        # Guard against empty pattern which would cause issues with re.split
        if not self.special_pattern:
            return [(text, False)]

        parts = re.split(f"({self.special_pattern})", text)

        result = []
        for part in parts:
            is_special = part in self.special_tokens
            result.append((part, is_special))

        return result

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        token_ids: list[int] = []

        # Split text by special tokens first
        segments = self._split_text_with_special_tokens(text)

        for segment, is_special in segments:
            if not segment:
                continue

            if is_special:
                # Special tokens should be encoded as a single token
                special_bytes = segment.encode("utf-8")
                if special_bytes in self._bytes_to_id:
                    token_ids.append(self._bytes_to_id[special_bytes])
                else:
                    # If special token not in vocab, encode as regular text
                    token_ids.extend(self._bpe_encode(special_bytes))
            else:
                # Encode as regular text using regex
                words = PAT.findall(segment)
                for word in words:
                    word_bytes = word.encode("utf-8")
                    token_ids.extend(self._bpe_encode(word_bytes))

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """Encode an iterable of input texts into an iterable of token IDs."""
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        tokens = []
        for token_id in ids:
            token_bytes = self.vocab.get(token_id)
            if token_bytes is None:
                continue
            tokens.append(token_bytes)
        return b"".join(tokens).decode("utf-8", errors="replace")
