import re
import unicodedata
from typing import List
from tqdm import tqdm

class TeluguPreprocessor:
    def __init__(self):
        self.telugu_range = range(0x0C00, 0x0C7F)
        
    def is_telugu(self, char: str) -> bool:
        return ord(char) in self.telugu_range

    def clean_text(self, text: str) -> str:
        # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        # Remove non-Telugu characters except spaces and newlines
        text = ''.join(char for char in text 
                      if self.is_telugu(char) or char in [' ', '\n'])
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove very short lines
        if len(text) < 10:
            return ""
        
        # Create repetitive patterns more efficiently
        if len(text) > 0:
            # Instead of repeating the whole text, repeat only important phrases
            words = text.split()
            if len(words) > 3:
                # Take first and last three words for repetition
                key_phrases = [" ".join(words[:3]), " ".join(words[-3:])]
                repeated_text = text + "\n"
                repeated_text += "\n".join(phrase * 2 for phrase in key_phrases)
                return repeated_text
            
        return text

    def process_file(self, input_path: str, output_path: str):
        """Process text file and save cleaned version"""
        print("Reading and cleaning text...")
        cleaned_texts = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                cleaned = self.clean_text(line)
                if cleaned:
                    cleaned_texts.append(cleaned)
        
        # Add common patterns more efficiently
        common_patterns = [
            "తెలుగు భాష",
            "తెలుగు సాహిత్యం",
            "తెలుగు సంస్కృతి",
            "భారతదేశం"
        ]
        
        # Add patterns only 10 times instead of 50
        for _ in range(10):
            cleaned_texts.extend(common_patterns)
        
        print(f"Writing {len(cleaned_texts)} cleaned texts...")
        with open(output_path, 'w', encoding='utf-8') as f:
            for text in cleaned_texts:
                f.write(text + '\n') 