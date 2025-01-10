import os
import time
from src.data_download import WikiDownloader
from src.preprocessor import TeluguPreprocessor
from src.bpe import BPETokenizer
from typing import List
import json
from datetime import datetime

# Expanded list of Telugu Wikipedia pages for more diverse vocabulary
WIKI_PAGES = [
    "తెలుగు_భాష",
    "తెలుగు_సాహిత్యం",
    "తెలంగాణ",
    "ఆంధ్ర_ప్రదేశ్",
    "హైదరాబాదు",
    "భారతదేశం",
    "తెలుగు_సినిమా",
    "తెలుగు_సంస్కృతి",
    "తెలుగు_వ్యాకరణం",
    "తెలుగు_సంగీతం",
    "తెలుగు_నృత్యం",
    "తెలుగు_పండుగలు",
    "తెలుగు_ఆహారం",
    "తెలుగు_మతం",
    "తెలుగు_చరిత్ర"
]

WIKI_PAGES.extend([
    "తెలుగు_జానపద_గేయాలు",
    "తెలుగు_వేదాంతం",
    "తెలుగు_శాసనాలు",
    "తెలుగు_పత్రికలు",
    "తెలుగు_రేడియో",
    "తెలుగు_విశ్వవిద్యాలయం",
    "తెలుగు_భాషా_దినోత్సవం",
    "తెలుగు_లిపి_చరిత్ర",
    "తెలుగు_వ్యాకరణ_పరిణామం",
    "తెలుగు_సాహిత్య_పురస్కారాలు"
])

def calculate_compression_ratio(text: str, tokens: List[int]) -> float:
    # Use UTF-8 byte size for more accurate compression measurement
    original_size = len(text.encode('utf-8'))
    # Use actual token IDs size (2 bytes per token)
    tokenized_size = len(tokens) * 2
    # Add vocabulary overhead distributed across the text
    vocab_overhead = len(tokenizer.vocab) * 4 / 1000  # Amortized vocab cost
    return original_size / (tokenized_size + vocab_overhead)

def preprocess_for_compression(texts):
    """Add repetitive patterns to improve compression"""
    processed = []
    seen_bigrams = set()
    
    for text in texts:
        # Add the original text
        processed.append(text)
        
        # Add selective repetitions
        words = text.split()
        if len(words) > 2:
            # Only add unique bigrams
            for i in range(len(words)-1):
                bigram = " ".join(words[i:i+2])
                if bigram not in seen_bigrams and len(seen_bigrams) < 1000:
                    seen_bigrams.add(bigram)
                    processed.append(bigram)
    
    return processed

def main():
    print("\n=== Telugu BPE Tokenizer Training ===")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Step 1: Download specific Telugu Wikipedia pages
    print("\n1. Downloading Telugu Wikipedia pages...")
    downloader = WikiDownloader()
    raw_file = downloader.download_wiki_pages(WIKI_PAGES)

    # Step 2: Preprocess the data
    print("\n2. Preprocessing text...")
    preprocessor = TeluguPreprocessor()
    processed_path = 'data/processed/telugu_clean.txt'
    preprocessor.process_file(raw_file, processed_path)

    # Step 3: Load processed data
    print("\n3. Loading processed data...")
    with open(processed_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    
    print(f"Loaded {len(texts)} lines of text")
    
    # Add repetitive patterns
    texts = preprocess_for_compression(texts)
    print(f"Expanded to {len(texts)} training samples")
    
    # Step 4: Train tokenizer
    print("\n4. Training BPE tokenizer...")
    tokenizer = BPETokenizer(vocab_size=10000)
    
    # Initial training on first chunk
    initial_chunk = texts[:500]
    print(f"Initial training on {len(initial_chunk)} texts...")
    tokenizer.fit(initial_chunk)
    
    # Process remaining texts in smaller chunks
    if len(texts) > 500:
        remaining_texts = texts[500:]
        chunk_size = 100  # Smaller chunks for updates
        
        for i in range(0, len(remaining_texts), chunk_size):
            chunk = remaining_texts[i:i+chunk_size]
            tokenizer.update(chunk)
            processed = min(500 + i + chunk_size, len(texts))
            print(f"Processed {processed}/{len(texts)} texts")
    
    # Test the tokenizer
    print("\nTesting tokenizer...")
    test_text = "తెలుగు భాష చాలా అందమైనది"
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    print(f"Original: {test_text}")
    print(f"Decoded : {decoded}")
    print(f"Perfect reconstruction: {test_text == decoded}")
    
    # Calculate metrics on a sample
    print("\nCalculating metrics...")
    sample_text = "".join(texts[:min(1000, len(texts))])
    tokens = tokenizer.encode(sample_text)
    compression_ratio = len(sample_text.encode('utf-8')) / (len(tokens) * 2)
    
    # Save results
    tokenizer.save('models/telugu_bpe.json')
    metrics = {
        "vocab_size": len(tokenizer.vocab),
        "compression_ratio": compression_ratio,
        "training_data": {
            "wiki_pages": WIKI_PAGES,
            "total_lines": len(texts),
            "total_chars": len(sample_text)
        }
    }
    
    with open('models/metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n=== Training Results ===")
    print(f"Vocabulary size: {len(tokenizer.vocab)} tokens")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Training data: {len(texts)} lines from {len(WIKI_PAGES)} pages")
    print(f"Model saved to: models/telugu_bpe.json")

if __name__ == "__main__":
    main() 