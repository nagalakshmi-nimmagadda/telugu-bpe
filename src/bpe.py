from typing import Dict, List, Tuple
from collections import defaultdict
import json
import time
from tqdm import tqdm

class BPETokenizer:
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}
        self.words = []
        self.special_tokens = {'<unk>': 0, '<pad>': 1}  # Add special tokens
        
    def get_stats(self, words: List[List[str]], progress=False) -> Dict[Tuple[str, str], int]:
        pairs = defaultdict(int)
        word_iter = tqdm(words) if progress else words
        for word in word_iter:
            for i in range(len(word) - 1):
                pairs[tuple(word[i:i+2])] += 1
        return pairs
    
    def merge_vocab(self, words: List[List[str]], pair: Tuple[str, str]) -> List[List[str]]:
        first, second = pair
        new_words = []
        
        for word in words:
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_words.append(new_word)
            
        return new_words
    
    def init_vocab(self, texts: List[str]):
        """Initialize vocabulary with unique characters and special tokens"""
        # Split texts into words and characters
        self.words = [[c for c in text] for text in texts]
        
        # Get unique characters
        chars = set()
        for word in self.words:
            chars.update(word)
        
        # Create initial vocabulary with special tokens
        self.vocab = self.special_tokens.copy()
        next_id = len(self.vocab)
        for char in sorted(chars):
            self.vocab[char] = next_id
            next_id += 1
            
        print(f"Initial vocab size: {len(self.vocab)} (including {len(self.special_tokens)} special tokens)")
    
    def merge_step(self, progress=False):
        """Perform one merge operation"""
        pairs = self.get_stats(self.words, progress=progress)
        if not pairs:
            return False
            
        best_pair = max(pairs.items(), key=lambda x: x[1])[0]
        self.words = self.merge_vocab(self.words, best_pair)
        
        self.merges[f"{best_pair[0]}|{best_pair[1]}"] = len(self.merges)
        new_token = ''.join(best_pair)
        self.vocab[new_token] = len(self.vocab)
        
        return True
    
    def fit(self, texts: List[str], callback=None):
        start_time = time.time()
        print("Initializing vocabulary...")
        self.init_vocab(texts)
        
        n_merges = self.vocab_size - len(self.vocab)
        print(f"Training for {n_merges} merges...")
        
        for i in range(n_merges):
            if i % 100 == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = i / elapsed if elapsed > 0 else 0
                print(f"Merge step {i}/{n_merges} ({tokens_per_sec:.1f} tokens/sec)")
                if callback:
                    callback(i, n_merges)
                    
            if not self.merge_step(progress=(i % 1000 == 0)):
                print("No more pairs to merge")
                break
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs with unknown token handling"""
        words = [list(word) for word in text.split()]
        for merge_str in self.merges:
            first, second = merge_str.split('|')
            words = self.merge_vocab(words, (first, second))
            
        tokens = []
        for word in words:
            for token in word:
                # Use <unk> token for unknown characters
                tokens.append(self.vocab.get(token, self.vocab['<unk>']))
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to text"""
        inv_vocab = {v: k for k, v in self.vocab.items()}
        return ''.join(inv_vocab.get(t, '') for t in tokens)
    
    def save(self, path: str):
        data = {
            'vocab': self.vocab,
            'merges': self.merges,
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tokenizer = cls(vocab_size=data['vocab_size'])
        tokenizer.vocab = data['vocab']
        tokenizer.merges = data['merges']
        tokenizer.special_tokens = data.get('special_tokens', {'<unk>': 0, '<pad>': 1})
        return tokenizer 
    
    def update(self, texts: List[str]):
        """Update the tokenizer with additional texts"""
        # Convert new texts to character sequences
        new_words = [[c for c in text] for text in texts]
        
        # Add any new characters to vocabulary
        for word in new_words:
            for char in word:
                if char not in self.vocab:
                    self.vocab[char] = len(self.vocab)
        
        # Apply existing merges to new words
        for merge_str, _ in self.merges.items():
            first, second = merge_str.split('|')
            new_words = self.merge_vocab(new_words, (first, second))
        
        # Add to existing words
        self.words.extend(new_words)
        
        # Optionally perform additional merges
        while len(self.vocab) < self.vocab_size:
            if not self.merge_step():
                break 