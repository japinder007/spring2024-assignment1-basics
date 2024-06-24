import os
import regex as re
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
pattern = re.compile(PAT)
from collections import Counter, defaultdict
from heapq import heappush, heappop, heappushpop, heapify
from dataclasses import dataclass, field
Part = tuple[int]
Parts = tuple[Part]
Pair: tuple[Part, Part]

@dataclass
class Word:
    text: str
    index: int
    parts: Parts
    count: int

def train_bpe(input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
        # First pre-tokenize.
        pre_tokens = pattern.findall(text)
        pre_token_counts = Counter(pre_tokens)

    # Initialize the vocabulary.
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i, st in enumerate(special_tokens):
        vocab[256 + i] = st.encode("utf-8")
    
    word_map: dict[int, Word] = {
        i: Word(text=item[0], index=i, parts=tuple([tuple([b]) for b in item[0].encode("utf-8")]), count=item[1]) 
        for i, item in enumerate(pre_token_counts.items())
    }

    # pair_counts: How often does a pair appear in the dataset.
    #   key:    a pair of bytes. Each pair is represented as a tuple of integers.
    #   value:  count
    pair_counts: dict[Pair, int] = Counter()
    # pair_to_word_map: Maps a pair to a word. Word is used in the sense of a token here. 
    # Note that tokens could be modified over time because of merging. 
    #   key:    a pair of bytes. Each pair is represented as a tuple of integers.
    #   value:  a set of tokens. Each token is represented as a tuple of bytes. bytes are represented as tuple[int]
    pair_to_word_map: dict[Pair, set[int]] = defaultdict(set)
    for w_index, word in word_map.items():
        for i in range(len(word.parts) - 1):
            pair_counts[(word.parts[i], word.parts[i+1])] += word.count
            pair_to_word_map[(word.parts[i], word.parts[i+1])].add(w_index)
    
    # token_counts
    # pair_to_word_map
    # pair_counts
    merges: list[tuple[bytes, bytes]] = []
    iteration = 0
    while len(vocab) < vocab_size:
        iteration += 1
        print(f"Iteration: {iteration}, current_vocab: {len(vocab)} of {vocab_size}")
        # Pop the most frequent pair and merge.
        max_pair, _ = max(pair_counts.items(), key=lambda item: (item[1], item[0][0], item[0][1]))
        # import pdb; pdb.set_trace()
        print(max_pair)
        merges.append((bytes(max_pair[0]), bytes(max_pair[1])))
        vocab[len(vocab)] = bytes(max_pair[0]) + bytes(max_pair[1])
        # del pair_counts[max_pair]

        # Go through all words where the pair occurs.
        word_indices = list(pair_to_word_map[max_pair])
        for w_index in word_indices:
            word = word_map[w_index]
            # Reduce the pair count for the old representation.
            for i in range(len(word.parts) - 1):
                pair = (word.parts[i], word.parts[i + 1])
                pair_counts[pair] -= word.count
                if pair in pair_to_word_map:
                    if w_index in pair_to_word_map[pair]:
                        pair_to_word_map[pair].remove(w_index)
                        if not pair_to_word_map[pair]:
                            del pair_to_word_map[pair]
                    else:
                        print(f"Warning: w_index {w_index} not found in pair_to_word_map[{pair}]")
                        # import pdb; pdb.set_trace()
                else:
                    print(f"Warning: pair {pair} not found in pair_to_word_map")
                    # import pdb; pdb.set_trace()
            
            # create a new representation. 
            i = 0
            new_token = []
            while i < len(word.parts):
                if i < (len(word.parts) - 1) and (word.parts[i], word.parts[i + 1]) == max_pair:
                    new_token.append(word.parts[i] + word.parts[i + 1])
                    i += 2
                else:
                    new_token.append(word.parts[i])
                    i += 1
            new_token = tuple(new_token)
            # Add the pair count using the new representation.
            for i in range(len(new_token) - 1):
                pair_counts[(new_token[i], new_token[i + 1])] += word.count
                pair_to_word_map[(new_token[i], new_token[i + 1])].add(w_index)

            word.parts = new_token
        
        if max_pair in pair_to_word_map:
            del pair_to_word_map[max_pair]
        
                
            
    return vocab, merges    
        



