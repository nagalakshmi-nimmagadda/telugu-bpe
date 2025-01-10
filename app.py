import gradio as gr
from src.bpe import BPETokenizer
import json
from pathlib import Path

# Load Models and Metrics
try:
    tokenizer = BPETokenizer.load("models/telugu_bpe.json")
    with open("models/metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
except Exception as e:
    print(f"Error loading models: {e}")
    metrics = {"vocab_size": 10003, "compression_ratio": 7.14}

# Example texts with descriptions
EXAMPLES = [
    ["నమస్కారం! ఇది తెలుగు భాష.", "Basic greeting"],
    ["తెలుగు భారతదేశంలోని ద్రావిడ భాషల్లో ఒకటి.", "Language classification"],
    ["తెలుగు సాహిత్యం చాలా సమృద్ధమైనది.", "Literature reference"]
]

def process_text(text: str) -> tuple:
    """Process Telugu text and return tokenization analysis"""
    try:
        if not text or not text.strip():
            return (
                "Please enter Telugu text",
                0,
                "",
                0.0,
                "No text provided"
            )
        
        # Process text
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens).replace("<unk>", "")
        
        # Calculate metrics
        compression_ratio = len(text.encode('utf-8')) / (len(tokens) * 2)
        
        # Professional analysis output
        analysis = f"""
        ### 📊 Analysis Results
        
        **📝 Text Statistics**
        - Length: {len(text)} characters
        - Words: {len(text.split())}
        - Tokens: {len(tokens)}
        
        **🔄 Compression**
        - Ratio: {compression_ratio:.2f}x
        - Efficiency: {len(text.encode('utf-8'))/len(tokens):.1f} bytes/token
        
        **🔍 Token Details**
        ```python
        {str(tokens)}
        ```
        
        **✨ Text Reconstruction**
        Input:  {text}
        Output: {decoded}
        """
        
        return (
            str(tokens),
            len(tokens),
            decoded,
            float(compression_ratio),
            analysis
        )
        
    except Exception as e:
        return (
            "Error processing text",
            0,
            "Processing failed",
            0.0,
            f"### ⚠️ Error\nAn error occurred: {str(e)}"
        )

# Custom CSS for better styling
css = """
.gradio-container {
    max-width: 850px !important;
    margin: auto;
}
.output-markdown {
    background-color: #f8f9fa;
    padding: 1.5rem;
    border-radius: 0.5rem;
    border: 1px solid #dee2e6;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.example-text {
    font-style: italic;
    color: #666;
}
"""

# Create interface
demo = gr.Interface(
    fn=process_text,
    inputs=gr.Textbox(
        label="✍️ Enter Telugu Text",
        placeholder="తెలుగు వాక్యాన్ని ఇక్కడ టైప్ చేయండి...",
        lines=3,
        info="Type or paste Telugu text here for analysis"
    ),
    outputs=[
        gr.Textbox(label="🔢 Token IDs", show_copy_button=True),
        gr.Number(label="📊 Token Count"),
        gr.Textbox(label="🔄 Decoded Text", show_copy_button=True),
        gr.Number(label="📈 Compression Ratio"),
        gr.Markdown(label="📋 Analysis")
    ],
    title="Telugu BPE Tokenizer 🔤",
    description="""
    ### 🌟 About This Project
    
    A state-of-the-art Byte-Pair Encoding (BPE) tokenizer specifically designed for Telugu text processing.
    
    #### ✨ Key Features
    - 🚀 **High Efficiency**: {metrics['compression_ratio']:.1f}x compression ratio
    - 📚 **Rich Vocabulary**: {metrics['vocab_size']:,} tokens
    - 🎯 **Accurate**: Precise text reconstruction
    - ⚡ **Fast**: Real-time processing
    
    #### 📝 How to Use
    1. Enter Telugu text in the input box
    2. Get instant analysis of:
       - Token representation
       - Compression metrics
       - Text reconstruction
       - Detailed statistics
    
    #### 🔍 Examples
    Try the sample texts below or enter your own Telugu text!
    """,
    examples=EXAMPLES,
    allow_flagging="never",
    css=css,
    theme="default"
)

if __name__ == "__main__":
    demo.launch()