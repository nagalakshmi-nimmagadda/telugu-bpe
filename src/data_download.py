import requests
from bs4 import BeautifulSoup
import os
import re

class WikiDownloader:
    def __init__(self, save_dir="data/raw"):
        self.save_dir = save_dir
        self.base_url = "https://te.wikipedia.org/wiki/"
        os.makedirs(save_dir, exist_ok=True)
        
    def clean_wiki_text(self, text):
        """Clean Wikipedia text content"""
        # Remove references
        text = re.sub(r'\[\d+\]', '', text)
        # Remove special wiki markup
        text = re.sub(r'\{\{.*?\}\}', '', text)
        return text.strip()
        
    def download_wiki_pages(self, titles):
        """Download specific Telugu Wikipedia pages"""
        all_text = []
        
        for title in titles:
            print(f"Downloading: {title}")
            url = self.base_url + title
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Get main content
            content = soup.find(id='mw-content-text')
            if content:
                paragraphs = content.find_all('p')
                for p in paragraphs:
                    text = self.clean_wiki_text(p.get_text())
                    if text:
                        all_text.append(text)
        
        # Save raw text
        output_file = os.path.join(self.save_dir, "telugu_wiki.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_text))
            
        return output_file

if __name__ == "__main__":
    downloader = WikiDownloader()
    downloader.download_wiki_pages(["Telugu_language", "Telugu_script"]) 