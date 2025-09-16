"""
Byte Pair Encoding (BPE) tokenizer implementation for CS336 Assignment 1.
This module contains a BPE tokenizer and training functionality.
"""

import json
import os
import re
from collections import Counter, defaultdict
from typing import Optional


class BPETokenizer:
    """
    Byte Pair Encoding tokenizer implementation.
    """
    
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: Optional[list[str]] = None,
    ):
        """
        Initialize BPE tokenizer.
        
        Args:
            vocab: Vocabulary mapping token ID to bytes
            merges: List of BPE merge rules  
            special_tokens: List of special token strings
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        # Create reverse vocab mapping (bytes -> id)
        self.byte_to_id = {v: k for k, v in vocab.items()}
        
        # Create merge rules lookup
        self.merge_rules = {}
        for i, (first, second) in enumerate(merges):
            self.merge_rules[(first, second)] = i
            
        # Create special token patterns - sort by length (descending) to prioritize longer tokens
        if self.special_tokens:
            # Sort special tokens by length in descending order for correct priority matching
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = '|'.join(re.escape(token) for token in sorted_tokens)
            self.special_token_pattern = re.compile(f'({special_pattern})')
        else:
            self.special_token_pattern = None
    
    def _split_on_special_tokens(self, text: str) -> list[str]:
        """
        Split text on special tokens, keeping the special tokens.
        
        Args:
            text: Input text to split
            
        Returns:
            List of text segments
        """
        if not self.special_token_pattern:
            return [text]
            
        segments = []
        last_end = 0
        
        for match in self.special_token_pattern.finditer(text):
            # Add text before the special token  
            if match.start() > last_end:
                segments.append(text[last_end:match.start()])
            
            # Add the special token
            segments.append(match.group())
            last_end = match.end()
            
        # Add remaining text
        if last_end < len(text):
            segments.append(text[last_end:])
            
        return [seg for seg in segments if seg]  # Filter empty strings
    
    def _get_pairs(self, tokens: list[bytes]) -> set[tuple[bytes, bytes]]:
        """
        Get all adjacent pairs in the token sequence.
        
        Args:
            tokens: List of byte tokens
            
        Returns:
            Set of adjacent pairs  
        """
        pairs = set()
        for i in range(len(tokens) - 1):
            pairs.add((tokens[i], tokens[i + 1]))
        return pairs
    
    def _apply_bpe(self, text: str) -> list[bytes]:
        """
        Apply BPE to a text segment.
        
        Args:
            text: Input text segment
            
        Returns:
            List of BPE tokens as bytes
        """
        # Convert text to bytes and split into individual bytes
        text_bytes = text.encode('utf-8')
        tokens = [bytes([b]) for b in text_bytes]
        
        if len(tokens) <= 1:
            return tokens
            
        while True:
            pairs = self._get_pairs(tokens)
            if not pairs:
                break
                
            # Find the pair with the lowest merge priority (earliest in merge list)
            min_pair = None
            min_priority = float('inf')
            
            for pair in pairs:
                if pair in self.merge_rules:
                    priority = self.merge_rules[pair]
                    if priority < min_priority:
                        min_priority = priority
                        min_pair = pair
                        
            if min_pair is None:
                break
                
            # Apply the merge
            first, second = min_pair
            new_tokens = []
            i = 0
            
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == first and tokens[i + 1] == second:
                    # Merge the pair
                    new_tokens.append(first + second)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
                    
            tokens = new_tokens
            
        return tokens
    
    def encode(self, text: str) -> list[int]:
        """
        Encode text into token IDs.
        
        Args:
            text: Input text to encode
            
        Returns:
            List of token IDs
        """
        # Split on special tokens
        segments = self._split_on_special_tokens(text)
        
        token_ids = []
        for segment in segments:
            if segment in self.special_tokens:
                # Handle special token
                special_bytes = segment.encode('utf-8')
                if special_bytes in self.byte_to_id:
                    token_ids.append(self.byte_to_id[special_bytes])
                else:
                    # If special token not in vocab, treat as regular text
                    bpe_tokens = self._apply_bpe(segment)
                    for token in bpe_tokens:
                        if token in self.byte_to_id:
                            token_ids.append(self.byte_to_id[token])
            else:
                # Handle regular text  
                bpe_tokens = self._apply_bpe(segment)
                for token in bpe_tokens:
                    if token in self.byte_to_id:
                        token_ids.append(self.byte_to_id[token])
                        
        return token_ids
    
    def decode(self, token_ids: list[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs to decode
            
        Returns:
            Decoded text string
        """
        # Convert token IDs to bytes
        byte_tokens = []
        for token_id in token_ids:
            if token_id in self.vocab:
                byte_tokens.append(self.vocab[token_id])
                
        # Concatenate all bytes and decode
        all_bytes = b''.join(byte_tokens)
        try:
            return all_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # Handle invalid UTF-8 sequences
            return all_bytes.decode('utf-8', errors='replace')
    
    def encode_iterable(self, texts):
        """
        Encode an iterable of texts, yielding individual token IDs.
        
        Args:
            texts: Iterable of texts to encode (e.g., file object with lines)
            
        Yields:
            Individual token IDs as integers
        """
        for text in texts:
            for token_id in self.encode(text):
                yield token_id


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a Byte-Pair Encoding tokenizer.
    
    Args:
        input_path: Path to text file for training
        vocab_size: Target vocabulary size
        special_tokens: List of special tokens to add
        
    Returns:
        Tuple of (vocabulary dict, merges list)
    """
    # Read training text
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Initialize vocabulary with all bytes
    vocab = {}
    next_token_id = 0
    
    # Add all possible bytes as base tokens
    for i in range(256):
        vocab[next_token_id] = bytes([i])
        next_token_id += 1
    
    # Add special tokens to vocabulary  
    for special_token in special_tokens:
        special_bytes = special_token.encode('utf-8')
        vocab[next_token_id] = special_bytes
        next_token_id += 1
    
    # Process text: first handle special tokens properly
    # Replace special tokens with placeholders to prevent them from being split
    special_token_map = {}
    processed_text = text
    
    for i, special_token in enumerate(special_tokens):
        placeholder = f"__SPECIAL_{i}__"
        special_token_map[placeholder] = special_token
        processed_text = processed_text.replace(special_token, placeholder)
    
    # Convert to byte tokens - work on the entire text for correct statistics
    text_bytes = processed_text.encode('utf-8')
    tokens = [bytes([b]) for b in text_bytes]
    
    # Restore special tokens as single tokens
    final_tokens = []
    i = 0
    while i < len(tokens):
        # Check if we're at the start of a special token placeholder
        found_special = False
        for placeholder, special_token in special_token_map.items():
            placeholder_bytes = placeholder.encode('utf-8')
            if i + len(placeholder_bytes) <= len(tokens):
                # Check if current position matches placeholder
                match = True
                for j in range(len(placeholder_bytes)):
                    if tokens[i + j] != bytes([placeholder_bytes[j]]):
                        match = False
                        break
                
                if match:
                    # Replace with special token
                    final_tokens.append(special_token.encode('utf-8'))
                    i += len(placeholder_bytes)
                    found_special = True
                    break
        
        if not found_special:
            final_tokens.append(tokens[i])
            i += 1
    
    tokens = final_tokens
    merges = []
    
    # Optimized BPE training loop
    while len(vocab) < vocab_size:
        # Count pairs efficiently using dict
        pair_counts = {}
        
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            # Skip pairs involving special tokens
            if any(token in special_token_map.values() for token in [tokens[i].decode('utf-8', errors='ignore'), tokens[i + 1].decode('utf-8', errors='ignore')]):
                continue
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
            
        if not pair_counts:
            break
            
        # Find most frequent pair
        most_frequent_pair = max(pair_counts.items(), key=lambda x: x[1])[0]
        first, second = most_frequent_pair
        
        # Add merge rule
        merges.append((first, second))
        
        # Create new token  
        new_token = first + second
        vocab[next_token_id] = new_token
        next_token_id += 1
        
        # Apply merge efficiently
        new_tokens = []
        i = 0
        while i < len(tokens):
            if (i < len(tokens) - 1 and 
                tokens[i] == first and 
                tokens[i + 1] == second and
                # Don't merge special tokens
                tokens[i] not in [st.encode('utf-8') for st in special_tokens] and
                tokens[i + 1] not in [st.encode('utf-8') for st in special_tokens]):
                new_tokens.append(new_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        
        tokens = new_tokens
        
    return vocab, merges