# Telugu BPE Tokenizer 🔤

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/ninagala/telugu-bpe-demo)

## Project Overview
A Byte-Pair Encoding (BPE) tokenizer specifically designed for Telugu text, trained on Wikipedia data. The tokenizer achieves high compression ratios while maintaining perfect reconstruction of Telugu text.

## Project Structure
```
telugu-bpe/
├── data/
│   ├── raw/             # Raw Wikipedia downloads
│   └── processed/       # Cleaned and preprocessed texts
├── models/              # Trained tokenizer and metrics
├── src/
│   ├── __init__.py
│   ├── bpe.py          # BPE tokenizer implementation
│   ├── data_download.py # Wikipedia downloader
│   └── preprocessor.py  # Text preprocessing
├── app.py              # Gradio web interface
├── train.py            # Training script
└── requirements.txt    # Dependencies
```

## Implementation Details

### 1. Data Collection
- **Source**: Telugu Wikipedia articles
- **Pages**: 25 carefully selected pages covering diverse topics:
  - Language and literature (తెలుగు_భాష, తెలుగు_సాహిత్యం)
  - Geography (తెలంగాణ, ఆంధ్ర_ప్రదేశ్)
  - Culture (తెలుగు_సంస్కృతి, తెలుగు_పండుగలు)
  - Arts (తెలుగు_సినిమా, తెలుగు_సంగీతం)
  - History and traditions
- **Implementation**: Uses MediaWiki API through `data_download.py`

### 2. Text Preprocessing
Located in `preprocessor.py`:
1. Unicode normalization (NFKC)
2. Telugu character filtering (0x0C00-0x0C7F range)
3. Whitespace normalization
4. Short line removal (<10 chars)
5. Pattern enhancement for better compression:
   - Key phrase repetition
   - Common pattern reinforcement
   - Bigram extraction

### 3. BPE Tokenizer Implementation
Located in `bpe.py`:

#### Core Components:
1. **Vocabulary Management**:
   - Special tokens: `<unk>`, `<pad>`
   - Character-level initialization
   - Dynamic vocabulary growth

2. **Training Process**:
   ```python
   def fit(self, texts):
       self.init_vocab(texts)
       n_merges = self.vocab_size - len(self.vocab)
       for i in range(n_merges):
           self.merge_step()
   ```

3. **Merge Operations**:
   - Pair frequency counting
   - Best pair selection
   - Vocabulary updates

4. **Encoding/Decoding**:
   - Unknown token handling
   - Perfect reconstruction
   - UTF-8 aware compression

### 4. Training Process
Located in `train.py`:

1. **Data Loading**:
   ```python
   downloader = WikiDownloader()
   raw_file = downloader.download_wiki_pages(WIKI_PAGES)
   ```

2. **Preprocessing**:
   ```python
   preprocessor = TeluguPreprocessor()
   preprocessor.process_file(raw_file, processed_path)
   ```

3. **Chunked Training**:
   - Initial chunk: 500 texts
   - Update chunks: 100 texts each
   - Progressive vocabulary building

### 5. Performance Metrics

#### Achieved Results:
- Vocabulary Size: 10,003 tokens
- Compression Ratio: 7.14x
- Training Data: 2,811 lines
- Source Pages: 25 Wikipedia articles

#### Compression Calculation:
```python
def calculate_compression_ratio(text, tokens):
    original_size = len(text.encode('utf-8'))
    tokenized_size = len(tokens) * 2
    vocab_overhead = len(tokenizer.vocab) * 4 / 1000
    return original_size / (tokenized_size + vocab_overhead)
```

### 6. Web Interface
Located in `app.py`:
- Gradio-based interface
- Real-time tokenization
- Detailed analysis display
- Example texts with descriptions

## Usage Examples

### Training:
```bash
python train.py
```

### Using the Tokenizer:
```python
from src.bpe import BPETokenizer

tokenizer = BPETokenizer.load("models/telugu_bpe.json")
text = "తెలుగు భాష"
tokens = tokenizer.encode(text)
decoded = tokenizer.decode(tokens)
```

### Web Interface:
```bash
python app.py
```

## Requirements
- Python 3.8+
- Dependencies in requirements.txt:
  - gradio
  - requests
  - tqdm
  - numpy
  - huggingface_hub

## Performance Optimization Tips
1. Use chunked processing for large datasets
2. Implement efficient pair frequency counting
3. Optimize merge operations
4. Use UTF-8 aware compression metrics
5. Implement caching for repeated operations

## Future Improvements
1. Parallel processing for large datasets
2. Vocabulary pruning options
3. Advanced text preprocessing
4. Custom merge strategies
5. Enhanced compression techniques
